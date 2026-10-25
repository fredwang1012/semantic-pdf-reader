"""The document library on disk: one folder per PDF, named by a hash of its bytes (so re-uploading is a no-op).

data/docs/<id>/doc.pdf, meta.json, chunks.json and, when an embedder is available, embeddings.npy.
"""
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunks import Chunk, make_chunks
from .pdftext import extract, title_of
from .retrieval import Embedder, Hit, Index

DOC_ID = re.compile(r"^[0-9a-f]{16}$")


class NoTextLayer(ValueError):
    """The PDF has no extractable text (a scan); OCR is not supported yet."""


@dataclass
class DocMeta:
    id: str
    title: str
    filename: str
    pages: int
    chunks: int
    page_sizes: list[list[float]]   # [width, height] per page, in points
    embedded: bool
    added: float

    def to_json(self) -> dict:
        return asdict(self)


def doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class Library:
    def __init__(self, root: Path, embedder: Embedder | None = None):
        self.root = root
        self.embedder = embedder
        self._indexes: dict[str, Index] = {}

    def _dir(self, id_: str) -> Path:
        if not DOC_ID.match(id_):
            raise KeyError(id_)
        return self.root / id_

    def list(self) -> list[DocMeta]:
        if not self.root.is_dir():
            return []
        metas = []
        for d in self.root.iterdir():
            path = d / "meta.json"
            if DOC_ID.match(d.name) and path.is_file():
                metas.append(DocMeta(**json.loads(path.read_text(encoding="utf-8"))))
        return sorted(metas, key=lambda m: m.added, reverse=True)

    def get(self, id_: str) -> DocMeta:
        path = self._dir(id_) / "meta.json"
        if not path.is_file():
            raise KeyError(id_)
        return DocMeta(**json.loads(path.read_text(encoding="utf-8")))

    def pdf_path(self, id_: str) -> Path:
        path = self._dir(id_) / "doc.pdf"
        if not path.is_file():
            raise KeyError(id_)
        return path

    def add(self, data: bytes, filename: str) -> DocMeta:
        """Extract, chunk, embed and save a PDF; returns the existing entry if the same bytes were added before."""
        id_ = doc_id(data)
        try:
            return self.get(id_)
        except KeyError:
            pass
        blocks, sizes = extract(data)
        chunks = make_chunks(blocks)
        if not chunks:
            raise NoTextLayer("no text layer found (a scanned PDF?); OCR is not supported yet")
        embeddings = None
        if self.embedder is not None:
            embeddings = np.asarray(self.embedder.embed_passages([c.text for c in chunks]), dtype=np.float32)
        stem = Path(filename).stem or "Untitled"
        meta = DocMeta(id=id_, title=title_of(data, stem), filename=Path(filename).name, pages=len(sizes),
                       chunks=len(chunks), page_sizes=[[s.width, s.height] for s in sizes],
                       embedded=embeddings is not None, added=time.time())
        # write into a temp folder and rename, so a crash never leaves a half-written doc that list() would show
        final = self._dir(id_)
        tmp = self.root / f".tmp-{id_}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "doc.pdf").write_bytes(data)
        (tmp / "chunks.json").write_text(json.dumps([c.to_json() for c in chunks]), encoding="utf-8")
        if embeddings is not None:
            np.save(tmp / "embeddings.npy", embeddings)
        (tmp / "meta.json").write_text(json.dumps(meta.to_json()), encoding="utf-8")
        os.replace(tmp, final)
        return meta

    def delete(self, id_: str) -> None:
        d = self._dir(id_)
        if not d.is_dir():
            raise KeyError(id_)
        self._indexes.pop(id_, None)
        shutil.rmtree(d)

    def index(self, id_: str) -> Index:
        if id_ not in self._indexes:
            d = self._dir(id_)
            if not (d / "chunks.json").is_file():
                raise KeyError(id_)
            chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
            emb_path = d / "embeddings.npy"
            emb = np.load(emb_path) if emb_path.is_file() else None
            self._indexes[id_] = Index(chunks, emb, self.embedder)
        return self._indexes[id_]

    def search_all(self, query: str, k_per_doc: int = 3, max_docs: int = 10) -> "list[tuple[DocMeta, list[Hit]]]":
        """Search every document; keep each one's top hits, drop those with none, best document first.

        Documents are scored independently: hybrid and BM25-only scores both lie in [0, 1] but are not
        calibrated across documents, so the ordering between documents is a rough guide only.
        """
        groups = []
        for meta in self.list():
            hits = self.index(meta.id).search(query, k_per_doc)
            if hits:
                groups.append((meta, hits))
        groups.sort(key=lambda g: g[1][0].score, reverse=True)
        return groups[:max_docs]
