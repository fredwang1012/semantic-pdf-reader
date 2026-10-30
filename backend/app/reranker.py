"""bge-reranker-v2-m3 cross-encoder scores for (query, passage) pairs. Opt-in (RERANK=auto) and imported lazily;
search skips the rerank stage if it is unavailable."""
import logging
import threading

import numpy as np

from .config import settings

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
log = logging.getLogger(__name__)


class BgeReranker:
    def __init__(self):
        from sentence_transformers import CrossEncoder  # heavy import: only when actually used
        self.model = CrossEncoder(MODEL_NAME)

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
        # one output label, so predict() applies a sigmoid: relevance in (0, 1), independent of the other passages
        return self.model.predict([(query, t) for t in texts], batch_size=batch_size)


_loaded: BgeReranker | None = None
_failed = False
_lock = threading.Lock()


def get_reranker() -> BgeReranker | None:
    """The shared reranker, loaded on first use; None when reranking is off or the model cannot be loaded."""
    global _loaded, _failed
    if settings.rerank == "off":
        return None
    with _lock:
        if _failed:
            return None
        if _loaded is None:
            try:
                _loaded = BgeReranker()
            except ImportError:
                log.info("sentence-transformers not installed (uv sync --extra embed): no reranking")
                _failed = True
            except Exception as e:  # e.g. the model cannot be downloaded offline
                log.warning("could not load %s (%s): no reranking", MODEL_NAME, e)
                _failed = True
        return _loaded
