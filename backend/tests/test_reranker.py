import pytest

from app import reranker

from .conftest import FakeReranker


@pytest.fixture
def fresh(monkeypatch):
    """A reranker module with nothing loaded yet, whose model constructor records its calls."""
    monkeypatch.setattr(reranker, "_loaded", None)
    monkeypatch.setattr(reranker, "_failed", False)
    calls = []

    def construct(make):
        def build():
            calls.append(1)
            return make()
        monkeypatch.setattr(reranker, "BgeReranker", build)
        return calls

    return construct


def test_off_returns_none_and_never_loads(fresh, monkeypatch):
    monkeypatch.setattr(reranker.settings, "rerank", "off")
    calls = fresh(lambda: FakeReranker("x"))
    assert reranker.get_reranker() is None
    assert calls == []


def test_auto_loads_once_and_shares_the_model(fresh, monkeypatch):
    monkeypatch.setattr(reranker.settings, "rerank", "auto")
    calls = fresh(lambda: FakeReranker("x"))
    first = reranker.get_reranker()
    assert isinstance(first, FakeReranker)
    assert reranker.get_reranker() is first
    assert len(calls) == 1


@pytest.mark.parametrize("error", [ImportError("no sentence_transformers"), OSError("offline")])
def test_a_model_that_cannot_load_gives_none_and_is_not_retried(fresh, monkeypatch, error):
    monkeypatch.setattr(reranker.settings, "rerank", "auto")

    def boom():
        raise error

    calls = fresh(boom)
    assert reranker.get_reranker() is None
    assert reranker.get_reranker() is None
    assert len(calls) == 1
