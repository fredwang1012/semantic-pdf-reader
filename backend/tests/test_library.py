import time

import pytest

from app.config import settings
from app.library import Library, NoTextLayer

from .conftest import FakeEmbedder, FakeReranker, make_pdf


# Each page is padded past the chunk minimum so it becomes its own chunk.
def _two_pages(first: str, second: str = "Nothing relevant here.") -> bytes:
    pad = " The quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget at length again and again." * 2
    return make_pdf([[first + pad], [second + pad]])


@pytest.fixture
def lib():
    return Library(settings.docs_dir)


def _blank_pdf() -> bytes:
    import pymupdf
    doc = pymupdf.open()
    doc.new_page().draw_rect(pymupdf.Rect(100, 100, 300, 300))
    data = doc.tobytes()
    doc.close()
    return data


def test_add_then_list_and_get(lib, sample_pdf):
    meta = lib.add(sample_pdf, "bio.pdf")
    assert meta.title == "bio"
    assert meta.pages == 2 and meta.chunks >= 2 and meta.embedded is False
    assert lib.get(meta.id) == meta
    assert lib.list() == [meta]
    assert lib.pdf_path(meta.id).read_bytes() == sample_pdf


def test_adding_the_same_bytes_twice_is_a_no_op(lib, sample_pdf):
    first = lib.add(sample_pdf, "a.pdf")
    second = lib.add(sample_pdf, "b.pdf")
    assert second.id == first.id
    assert [p.name for p in settings.docs_dir.iterdir()] == [first.id]


def test_pdf_without_text_is_rejected_and_leaves_nothing(lib):
    with pytest.raises(NoTextLayer):
        lib.add(_blank_pdf(), "scan.pdf")
    assert not settings.docs_dir.exists() or list(settings.docs_dir.iterdir()) == []


def test_delete_removes_folder_and_cached_index(lib, sample_pdf):
    meta = lib.add(sample_pdf, "a.pdf")
    lib.index(meta.id)
    lib.delete(meta.id)
    assert not (settings.docs_dir / meta.id).exists()
    assert meta.id not in lib._indexes
    with pytest.raises(KeyError):
        lib.get(meta.id)
    with pytest.raises(KeyError):
        lib.delete(meta.id)


@pytest.mark.parametrize("bad", ["../etc", "short", "G" * 16, "0123456789abcdef0", ""])
def test_malformed_ids_raise_key_error(lib, bad):
    for call in (lib.get, lib.pdf_path, lib.index, lib.delete):
        with pytest.raises(KeyError):
            call(bad)


def test_embedded_documents_search_in_hybrid_mode(sample_pdf):
    lib = Library(settings.docs_dir, FakeEmbedder())
    meta = lib.add(sample_pdf, "a.pdf")
    assert meta.embedded is True
    assert lib.index(meta.id).mode == "hybrid"
    assert lib.index(meta.id).search("how do plants make food")[0].chunk.page == 1


def test_a_fresh_library_on_the_same_root_still_searches(sample_pdf):
    meta = Library(settings.docs_dir, FakeEmbedder()).add(sample_pdf, "a.pdf")
    again = Library(settings.docs_dir, FakeEmbedder())
    assert again.list() == [meta]
    index = again.index(meta.id)
    assert index.mode == "hybrid"
    assert index.search("Adam optimizer")[0].chunk.page == 2


def test_search_all_groups_by_document(lib):
    plants = lib.add(_two_pages("Photosynthesis lets plants turn sunlight into food."), "plants.pdf")
    rockets = lib.add(_two_pages("Plants and rockets.", "Rockets burn fuel to reach orbit."), "rockets.pdf")
    assert [m.id for m, _, _ in lib.search_all("orbit")] == [rockets.id]
    assert {m.id for m, _, _ in lib.search_all("plants")} == {plants.id, rockets.id}
    assert len(lib.search_all("plants", max_docs=1)) == 1
    assert all(len(hits) <= 1 for _, hits, _ in lib.search_all("plants", k_per_doc=1))
    assert lib.search_all("zebra") == []


def test_search_all_orders_by_query_term_coverage(lib):
    full = lib.add(_two_pages("Rockets burn fuel to reach orbit."), "full.pdf")
    time.sleep(0.01)  # the partial match is newer, so list order alone would put it first
    partial = lib.add(_two_pages("Rockets are loud."), "partial.pdf")
    assert [m.id for m, _, _ in lib.search_all("rockets orbit")] == [full.id, partial.id]


def test_search_all_breaks_coverage_ties_by_reranker_score():
    older = Library(settings.docs_dir).add(_two_pages("Rockets reach orbit."), "older.pdf")
    time.sleep(0.01)
    newer = Library(settings.docs_dir).add(_two_pages("Rockets burn fuel."), "newer.pdf")
    # both top hits score 1.0 after per-document scaling, so without a reranker list order (newest first) decides
    assert [m.id for m, _, _ in Library(settings.docs_dir).search_all("rockets")] == [newer.id, older.id]
    groups = Library(settings.docs_dir, reranker=FakeReranker("orbit")).search_all("rockets")
    assert [m.id for m, _, _ in groups] == [older.id, newer.id]
    assert [hits[0].score for _, hits, _ in groups] == [1.0, 0.0]


def test_search_all_skips_a_document_deleted_mid_search(lib, monkeypatch):
    a = lib.add(_two_pages("Rockets burn fuel."), "a.pdf")
    b = lib.add(_two_pages("Rockets reach orbit."), "b.pdf")
    real = lib.index

    def flaky(id_):
        if id_ == a.id:
            raise KeyError(id_)
        return real(id_)

    monkeypatch.setattr(lib, "index", flaky)
    assert [m.id for m, _, _ in lib.search_all("rockets")] == [b.id]


def test_search_all_on_an_empty_library(lib):
    assert lib.search_all("anything") == []


def test_concurrent_adds_of_the_same_pdf_leave_one_complete_folder(lib):
    import threading

    class SlowEmbedder(FakeEmbedder):
        def embed_passages(self, texts):
            time.sleep(0.2)
            return super().embed_passages(texts)

    lib.embedder = SlowEmbedder()
    data = _two_pages("Plants make food from light.")
    results, errors = [], []

    def work():
        try:
            results.append(lib.add(data, "a.pdf"))
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len({m.id for m in results}) == 1 and len(results) == 2
    folders = list(lib.root.iterdir())
    assert [f.name for f in folders] == [results[0].id]
    assert {p.name for p in folders[0].iterdir()} >= {"meta.json", "chunks.json", "doc.pdf"}


def test_add_discards_its_temp_folder_when_the_final_one_appears_first(lib, monkeypatch):
    data = _two_pages("Plants make food from light.")
    first = Library(lib.root).add(data, "a.pdf")
    real_get = Library.get
    calls = {"n": 0}

    def blind_get(self, id_):  # the first check misses, as if another process had not finished yet
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError(id_)
        return real_get(self, id_)

    monkeypatch.setattr(Library, "get", blind_get)
    assert lib.add(data, "b.pdf").id == first.id
    assert [f.name for f in lib.root.iterdir()] == [first.id]
