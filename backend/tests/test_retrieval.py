import pytest

from app.chunks import make_chunks
from app.pdftext import extract
from app.chunks import Chunk
from app.retrieval import RERANK_TOP, Index

from .conftest import FakeEmbedder, FakeReranker


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


def test_reranker_reorders_hits_and_its_scores_are_returned():
    chunks = [_chunk(0, "rockets rockets rockets burn fuel"), _chunk(1, "rockets reach orbit")]
    assert Index(chunks).search("rockets")[0].chunk.id == "c0"
    hits = Index(chunks, reranker=FakeReranker("orbit")).search("rockets")
    assert [h.chunk.id for h in hits] == ["c1", "c0"]
    assert [h.score for h in hits] == [1.0, 0.0]


@pytest.mark.parametrize("k", [3, RERANK_TOP + 5])
def test_reranker_sees_the_top_candidates_and_returns_k(k):
    chunks = [_chunk(i, "rockets " * (i + 1) + ("orbit" if i % 2 else "fuel")) for i in range(RERANK_TOP + 10)]
    reranker = FakeReranker("orbit")
    hits = Index(chunks, reranker=reranker).search("rockets", k)
    assert len(hits) == k
    assert len(reranker.seen) == max(k, RERANK_TOP)
    assert all("orbit" in h.chunk.text for h in hits[:3])


def test_reranker_keeps_the_bm25_term_filter():
    chunks = [_chunk(0, "rockets burn fuel"), _chunk(1, "plants need light in orbit")]
    reranker = FakeReranker("orbit")
    index = Index(chunks, reranker=reranker)
    assert [h.chunk.id for h in index.search("rockets")] == ["c0"]
    assert reranker.seen == ["rockets burn fuel"]
    assert index.search("zebra") == []


def test_reranker_works_on_top_of_hybrid(chunks):
    emb = FakeEmbedder()
    index = Index(chunks, emb.embed_passages([c.text for c in chunks]), emb, FakeReranker("momentum"))
    assert index.mode == "hybrid"
    assert index.search("how do plants make food")[0].chunk.page == 2
