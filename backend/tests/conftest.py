import numpy as np
import pymupdf
import pytest

from app.config import settings
from app.retrieval import tokenize

# Two pages on unrelated topics, so a search can be checked by the page it lands on.
PAGE_TEXTS = [
    [
        "Photosynthesis converts light energy into chemical energy stored in glucose. Chlorophyll in the "
        "chloroplast absorbs mostly red and blue light and reflects green.",
        "The Calvin cycle fixes carbon dioxide into sugar using ATP and NADPH produced by the light reactions.",
        "Stomata on the underside of a leaf open to let carbon dioxide in and water vapour out.",
    ],
    [
        "Gradient descent minimises a loss by stepping against its gradient. The learning rate controls the "
        "step size: too large and training diverges, too small and it crawls.",
        "Adam keeps running averages of the gradient and its square to adapt the step size per parameter.",
        "Momentum smooths noisy updates by adding a fraction of the previous update to the current one.",
    ],
]


def make_pdf(pages: list[list[str]] = PAGE_TEXTS, title: str = "") -> bytes:
    doc = pymupdf.open()
    for n, paras in enumerate(pages, start=1):
        page = doc.new_page(width=612, height=792)
        y = 72
        for para in paras:
            box = pymupdf.Rect(72, y, 540, y + 120)
            page.insert_textbox(box, para, fontsize=11)
            y += 140
        page.insert_text((300, 760), str(n), fontsize=9)  # a page number footer, which extraction drops
    if title:
        doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf()


class FakeEmbedder:
    """Concept vectors: words map to shared dimensions, so "plants making food" lands near photosynthesis without
    sharing a single word with it (what BM25 alone would miss)."""
    CONCEPTS = [
        {"photosynthesis", "plants", "plant", "chlorophyll", "light", "glucose", "food", "leaves", "sugar"},
        {"gradient", "descent", "optimizer", "optimiser", "learning", "rate", "adam", "training", "step"},
    ]

    def _vec(self, text: str) -> np.ndarray:
        words = set(text.lower().replace(".", " ").replace(",", " ").split())
        v = np.array([len(words & c) for c in self.CONCEPTS], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class FakeReranker:
    """Scores a passage by how often `word` occurs in it, whatever the query, so it can disagree with the first stage.
    Records every passage it is shown."""
    def __init__(self, word: str):
        self.word = word
        self.seen: list[str] = []

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        return np.array([tokenize(t).count(self.word) for t in texts], dtype=float)


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch, tmp_path):
    """Every test gets its own library folder and never loads the real embedding or reranker model."""
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "embed", "off")
    monkeypatch.setattr(settings, "rerank", "off")
    from app import main
    monkeypatch.setattr(main, "_library", None)
