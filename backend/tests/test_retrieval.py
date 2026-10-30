import pytest

from app.chunks import make_chunks
from app.pdftext import extract
from app.chunks import Chunk
from app.retrieval import Index, match_terms

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


def _chunk(i, text):
    return Chunk(id=f"c{i}", page=1, text=text, rects=[[1, 0, 0, 10, 10]])


def test_bm25_single_chunk_document_matches_its_own_word():
    hits = Index([_chunk(0, "Rockets burn fuel.")]).search("rockets")
    assert len(hits) == 1 and hits[0].score == 1.0


def test_bm25_chunks_that_all_contain_the_term_all_match():
    index = Index([_chunk(0, "rockets burn fuel"), _chunk(1, "rockets reach orbit")])
    assert len(index.search("rockets")) == 2


def test_bm25_query_matching_no_word_returns_nothing():
    index = Index([_chunk(0, "rockets burn fuel"), _chunk(1, "plants need light")])
    assert index.search("zebra") == []


def test_bm25_chunk_sharing_only_stopwords_is_not_a_match():
    index = Index([_chunk(0, "the rockets burn fuel"), _chunk(1, "the plants need light")])
    assert [h.chunk.id for h in index.search("why do the rockets fly")] == ["c0"]


def test_bm25_query_of_only_stopwords_still_matches_them():
    index = Index([_chunk(0, "to be or not to be"), _chunk(1, "rockets burn fuel")])
    assert [h.chunk.id for h in index.search("to be")] == ["c0"]


def test_match_terms_drop_stopwords_unless_nothing_is_left():
    assert match_terms("Why does THE rocket fly?") == {"rocket", "fly"}
    assert match_terms("to be or not") == {"to", "be", "or", "not"}
