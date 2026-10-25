"""Hybrid search over one document's chunks: BM25 (scaled by the document's best raw score) blended with cosine similarity.

Falls back to BM25 alone when the document has no embeddings or no embedder is loaded.
"""
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Plus

from .chunks import Chunk

DEFAULT_K = 8
W_EMBED, W_BM25 = 0.6, 0.4

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None, embedder: Embedder | None = None):
        self.chunks = chunks
        usable = embeddings is not None and embedder is not None and len(embeddings) == len(chunks)
        self.embeddings = embeddings if usable else None
        self.embedder = embedder if usable else None
        # BM25Plus rather than Okapi: Okapi's idf is 0 for a term in half the chunks, so a short document (a few
        # chunks) would score its own words 0 and never match.
        tokens = [tokenize(c.text) for c in chunks]
        self._words = [set(t) for t in tokens]
        self._bm25 = BM25Plus(tokens) if chunks else None

    @property
    def mode(self) -> str:
        return "hybrid" if self.embeddings is not None else "bm25"

    def scores(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        terms = tokenize(query)
        bm = np.asarray(self._bm25.get_scores(terms), dtype=float) if self._bm25 and terms else np.zeros(n)
        hi = bm.max() if n else 0.0
        bm_norm = bm / hi if hi > 0 else np.zeros(n)
        if self.mode == "bm25" or not query.strip():
            return bm_norm
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        cos = self.embeddings @ (q / (np.linalg.norm(q) or 1.0))
        return W_EMBED * cos + W_BM25 * bm_norm

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        """Top-k chunks, best first. BM25-only search returns only chunks that contain a query word."""
        if not self.chunks or not query.strip():
            return []
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        if self.mode == "bm25":
            # BM25Plus gives absent terms a constant positive score, so a score says nothing about a match
            terms = set(tokenize(query))
            order = [i for i in order if self._words[i] & terms]
        return [Hit(self.chunks[i], float(s[i])) for i in order]
