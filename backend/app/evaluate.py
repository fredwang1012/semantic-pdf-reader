"""Retrieval eval harness: recall@k, MRR, latency and index size per retriever over a hand-labelled question set.

Run: python -m app.evaluate <pdf folder> <questions.jsonl> [--k 1 3 5 10] [--retrievers bm25 hybrid hybrid+rerank]
     [--json out.json]
Everything is built in memory; nothing is written to the library's data folder.
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunks import Chunk, make_chunks
from .embedder import get_embedder
from .pdftext import extract
from .reranker import get_reranker
from .retrieval import Hit, Index, Reranker

DEFAULT_KS = [1, 3, 5, 10]


# --- metrics (pure) ---------------------------------------------------------------------------------------------

def rank_of(hits: list[Hit], pages: list[int]) -> int | None:
    """1-based position of the first hit that touches one of `pages` (any page in its rects); None if none does."""
    wanted = set(pages)
    for n, hit in enumerate(hits, start=1):
        if {int(r[0]) for r in hit.chunk.rects} & wanted:
            return n
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank; a question that was not found contributes 0."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = -(-len(ordered) * q // 100)  # ceil
    return ordered[int(max(1, min(len(ordered), rank))) - 1]


# --- retrievers -------------------------------------------------------------------------------------------------

class Unavailable(Exception):
    """A retriever that cannot run in this environment."""


Builder = Callable[[list[Chunk]], Index]


def _bm25(reranker: Reranker | None = None) -> Builder:
    return lambda chunks: Index(chunks, reranker=reranker)


def _hybrid(reranker: Reranker | None = None) -> Builder:
    embedder = get_embedder()
    if embedder is None:
        raise Unavailable("no embedder (embeddings are off or sentence-transformers is not installed; "
                          "uv sync --extra embed)")
    return lambda chunks: Index(chunks, embedder.embed_passages([c.text for c in chunks]) if chunks else None, embedder,
                                reranker)


def _reranked(first_stage: Callable[[Reranker], Builder]) -> Callable[[], Builder]:
    """The first stage followed by the cross-encoder reranker."""
    def factory() -> Builder:
        reranker = get_reranker()
        if reranker is None:
            raise Unavailable("no reranker (RERANK is off or sentence-transformers is not installed; set RERANK=auto "
                              "and uv sync --extra embed)")
        return first_stage(reranker)
    return factory


# Name -> factory returning a per-document index builder (or raising Unavailable). A visual retriever plugs in here.
RETRIEVERS: dict[str, Callable[[], Builder]] = {
    "bm25": _bm25, "hybrid": _hybrid, "bm25+rerank": _reranked(_bm25), "hybrid+rerank": _reranked(_hybrid),
}


# --- running ----------------------------------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    ks: list[int]
    ranks: list[int | None]
    latencies_ms: list[float]
    build_s: float
    chunk_count: int
    embedding_bytes: int

    def summary(self) -> dict:
        top = max(self.ks)
        return {
            "retriever": self.name,
            **{f"recall@{k}": recall_at_k(self.ranks, k) for k in self.ks},
            "mrr": mrr([r if r is not None and r <= top else None for r in self.ranks]),
            "latency_median_ms": statistics.median(self.latencies_ms) if self.latencies_ms else 0.0,
            "latency_p95_ms": percentile(self.latencies_ms, 95),
            "build_s": self.build_s,
            "chunks": self.chunk_count,
            "embedding_bytes": self.embedding_bytes,
        }


def load_questions(path: Path) -> list[dict]:
    questions = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        q = json.loads(line)
        if not {"question", "file", "pages"} <= q.keys():
            raise ValueError(f"{path}:{n}: expected keys question, file, pages")
        questions.append(q)
    return questions


def load_chunks(folder: Path, files: set[str]) -> dict[str, list[Chunk]]:
    """Chunks per PDF that exists and has a text layer; the others are warned about and left out."""
    out: dict[str, list[Chunk]] = {}
    for name in sorted(files):
        pdf = folder / name
        if not pdf.is_file():
            print(f"warning: {name} not found in {folder}; its questions count as not found", file=sys.stderr)
            continue
        chunks = make_chunks(extract(str(pdf))[0])
        if not chunks:
            print(f"warning: {name} has no text layer; its questions count as not found", file=sys.stderr)
            continue
        out[name] = chunks
    return out


def run_retriever(name: str, builder: Builder, chunks_by_file: dict[str, list[Chunk]], questions: list[dict],
                  ks: list[int]) -> Result:
    start = time.perf_counter()
    indexes = {f: builder(chunks) for f, chunks in chunks_by_file.items()}
    build_s = time.perf_counter() - start
    ranks: list[int | None] = []
    latencies: list[float] = []
    for q in questions:
        index = indexes.get(q["file"])
        if index is None:
            ranks.append(None)
            continue
        t = time.perf_counter()
        hits = index.search(q["question"], max(ks))
        latencies.append((time.perf_counter() - t) * 1000)
        ranks.append(rank_of(hits, q["pages"]))
    embedding_bytes = sum(ix.embeddings.nbytes for ix in indexes.values() if ix.embeddings is not None)
    return Result(name, ks, ranks, latencies, build_s, sum(len(c) for c in chunks_by_file.values()), embedding_bytes)


def markdown_table(results: list[Result]) -> str:
    ks = results[0].ks
    head = ["Retriever", *[f"R@{k}" for k in ks], "MRR", "Median ms", "p95 ms", "Build s", "Chunks", "Embedding bytes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for r in results:
        s = r.summary()
        row = [r.name, *[f"{s[f'recall@{k}']:.3f}" for k in ks], f"{s['mrr']:.3f}", f"{s['latency_median_ms']:.2f}",
               f"{s['latency_p95_ms']:.2f}", f"{s['build_s']:.2f}", str(s["chunks"]), str(s["embedding_bytes"])]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.evaluate", description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path, help="folder holding the PDFs")
    ap.add_argument("questions", type=Path, help="JSONL: {question, file, pages}")
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS), choices=list(RETRIEVERS))
    ap.add_argument("--json", type=Path, help="write raw numbers and per-question ranks here")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    chunks_by_file = load_chunks(args.folder, {q["file"] for q in questions})
    results: list[Result] = []
    for name in args.retrievers:
        try:
            builder = RETRIEVERS[name]()
        except Unavailable as e:
            print(f"warning: {name} skipped: {e}", file=sys.stderr)
            continue
        results.append(run_retriever(name, builder, chunks_by_file, questions, sorted(set(args.k))))
    if not results:
        print("error: no retriever ran", file=sys.stderr)
        return 1

    print(markdown_table(results))
    if args.json:
        payload = [{**r.summary(), "ranks": [{"question": q["question"], "rank": rk}
                                             for q, rk in zip(questions, r.ranks)]} for r in results]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
