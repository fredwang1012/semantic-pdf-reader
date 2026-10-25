import pytest

from app.chunks import make_chunks
from app.pdftext import extract
from app.retrieval import Index

from .conftest import FakeEmbedder


@pytest.fixture
def chunks(sample_pdf):
    return make_chunks(extract(sample_pdf)[0])


@pytest.fixture
def bm25_index(chunks):
    return Index(chunks)


@pytest.fixture
def hybrid_index(chunks):
    emb = FakeEmbedder()
    return Index(chunks, emb.embed_passages([c.text for c in chunks]), emb)


def test_bm25_ranks_the_matching_page_first(bm25_index):
    hits = bm25_index.search("Adam optimizer")
    assert bm25_index.mode == "bm25"
    assert hits[0].chunk.page == 2


@pytest.mark.parametrize("index_fixture", ["bm25_index", "hybrid_index"])
def test_exact_term_finds_its_page_in_both_modes(index_fixture, request):
    index = request.getfixturevalue(index_fixture)
    assert index.search("NADPH")[0].chunk.page == 1


def test_hybrid_finds_a_paraphrase_that_bm25_misses(bm25_index, hybrid_index):
    query = "how do plants make food"
    assert hybrid_index.mode == "hybrid"
    assert hybrid_index.search(query)[0].chunk.page == 1
    assert [h for h in bm25_index.search(query) if h.chunk.page == 1] == []


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_query_returns_nothing(hybrid_index, bm25_index, query):
    assert hybrid_index.search(query) == []
    assert bm25_index.search(query) == []


def test_mismatched_embeddings_fall_back_to_bm25(chunks):
    emb = FakeEmbedder()
    index = Index(chunks, emb.embed_passages(["only one"]), emb)
    assert index.mode == "bm25"
    assert index.search("NADPH")[0].chunk.page == 1


def test_empty_index_searches_to_nothing():
    assert Index([]).search("anything") == []
