import pytest

from app.config import settings
from app.library import Library, NoTextLayer

from .conftest import FakeEmbedder, make_pdf


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
