import json

import pytest

from app import evaluate
from app.chunks import Chunk
from app.retrieval import Hit

from .conftest import FakeEmbedder, FakeReranker, make_pdf


def hit(*pages: int) -> Hit:
    return Hit(Chunk("c", pages[0], "t", [[p, 0, 0, 1, 1] for p in pages]), 1.0)


def test_recall_at_k():
    ranks = [1, 3, None, 2]
    assert evaluate.recall_at_k(ranks, 1) == 0.25
    assert evaluate.recall_at_k(ranks, 2) == 0.5
    assert evaluate.recall_at_k(ranks, 10) == 0.75


def test_metrics_all_missing_and_empty():
    assert evaluate.recall_at_k([None, None], 5) == 0.0
    assert evaluate.mrr([None, None]) == 0.0
    assert evaluate.recall_at_k([], 5) == 0.0
    assert evaluate.mrr([]) == 0.0


def test_mrr():
    assert evaluate.mrr([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_percentile():
    assert evaluate.percentile([], 95) == 0.0
    assert evaluate.percentile([5.0], 95) == 5.0
    assert evaluate.percentile(list(map(float, range(1, 101))), 95) == 95.0


def test_rank_of_uses_any_page_in_rects():
    hits = [hit(1), hit(2), hit(3, 4)]
    assert evaluate.rank_of(hits, [1]) == 1
    assert evaluate.rank_of(hits, [4]) == 3          # second page of a chunk that spans pages 3-4
    assert evaluate.rank_of(hits, [2, 4]) == 2       # any listed page counts; the first match wins
    assert evaluate.rank_of(hits, [9]) is None
    assert evaluate.rank_of([], [1]) is None


def _setup(tmp_path, extra=()):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    (folder / "sample.pdf").write_bytes(make_pdf())
    rows = [
        {"question": "How does the Calvin cycle fix carbon dioxide?", "file": "sample.pdf", "pages": [1]},
        {"question": "What does Adam do with running averages of the gradient?", "file": "sample.pdf", "pages": [2]},
        *extra,
    ]
    questions = tmp_path / "q.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return folder, questions


def test_main_bm25_end_to_end(tmp_path, capsys):
    folder, questions = _setup(tmp_path)
    out = tmp_path / "out.json"
    assert evaluate.main([str(folder), str(questions), "--k", "1", "3", "--retrievers", "bm25", "--json", str(out)]) == 0
    stdout = capsys.readouterr().out
    header = stdout.splitlines()[0]
    for col in ("Retriever", "R@1", "R@3", "MRR", "Median ms", "p95 ms", "Build s", "Chunks", "Embedding bytes"):
        assert col in header
    row = stdout.splitlines()[2].split("|")
    assert row[1].strip() == "bm25" and row[2].strip() == "1.000"
    (result,) = json.loads(out.read_text(encoding="utf-8"))
    assert result["recall@1"] == 1.0 and result["embedding_bytes"] == 0
    assert [r["rank"] for r in result["ranks"]] == [1, 1]


def test_missing_pdf_counts_as_not_found(tmp_path, capsys):
    extra = [{"question": "anything", "file": "gone.pdf", "pages": [1]}]
    folder, questions = _setup(tmp_path, extra)
    assert evaluate.main([str(folder), str(questions), "--k", "1", "--retrievers", "bm25"]) == 0
    captured = capsys.readouterr()
    assert "gone.pdf not found" in captured.err
    assert captured.out.splitlines()[2].split("|")[2].strip() == "0.667"


def test_hybrid_skipped_without_embedder(tmp_path, capsys):
    folder, questions = _setup(tmp_path)
    assert evaluate.main([str(folder), str(questions), "--k", "1"]) == 0
    captured = capsys.readouterr()
    assert "hybrid skipped" in captured.err
    assert "bm25+rerank skipped: no reranker" in captured.err and "hybrid+rerank skipped" in captured.err
    assert [r.split("|")[1].strip() for r in captured.out.splitlines()[2:]] == ["bm25"]


def test_hybrid_with_injected_embedder(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(evaluate, "get_embedder", FakeEmbedder)
    folder, questions = _setup(tmp_path)
    assert evaluate.main([str(folder), str(questions), "--k", "1"]) == 0
    rows = capsys.readouterr().out.splitlines()[2:]
    assert [r.split("|")[1].strip() for r in rows] == ["bm25", "hybrid"]
    assert int(rows[1].split("|")[-2]) > 0   # embedding bytes


def test_rerank_retrievers_run_with_injected_models(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(evaluate, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(evaluate, "get_reranker", lambda: FakeReranker("momentum"))  # always prefers page 2
    folder, questions = _setup(tmp_path)
    out = tmp_path / "out.json"
    assert evaluate.main([str(folder), str(questions), "--k", "1", "3", "--json", str(out)]) == 0
    rows = capsys.readouterr().out.splitlines()[2:]
    assert [r.split("|")[1].strip() for r in rows] == ["bm25", "hybrid", "bm25+rerank", "hybrid+rerank"]
    ranks = {r["retriever"]: [q["rank"] for q in r["ranks"]] for r in json.loads(out.read_text(encoding="utf-8"))}
    assert ranks["bm25"] == [1, 1]
    assert ranks["bm25+rerank"] == ranks["hybrid+rerank"] == [2, 1]


def test_hybrid_rerank_needs_the_embedder_too(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(evaluate, "get_reranker", lambda: FakeReranker("momentum"))
    folder, questions = _setup(tmp_path)
    assert evaluate.main([str(folder), str(questions), "--retrievers", "hybrid+rerank"]) == 1
    err = capsys.readouterr().err
    assert "hybrid+rerank skipped: no embedder" in err and "no retriever ran" in err
