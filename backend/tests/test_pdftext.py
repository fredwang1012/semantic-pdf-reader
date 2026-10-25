
from app.pdftext import extract, join_lines, title_of

from .conftest import make_pdf


def test_page_sizes(sample_pdf):
    _, sizes = extract(sample_pdf)
    assert [(s.width, s.height) for s in sizes] == [(612, 792), (612, 792)]


def test_blocks_are_on_real_pages_and_footer_is_dropped(sample_pdf):
    blocks, _ = extract(sample_pdf)
    assert {b.page for b in blocks} == {1, 2}
    assert all(b.text not in {"1", "2"} for b in blocks)
    assert any("Calvin cycle" in b.text for b in blocks)


def test_line_boxes_lie_inside_their_page(sample_pdf):
    blocks, sizes = extract(sample_pdf)
    for b in blocks:
        for ln in b.lines:
            x0, y0, x1, y1 = ln.bbox
            size = sizes[ln.page - 1]
            assert 0 <= x0 < x1 <= size.width
            assert 0 <= y0 < y1 <= size.height


def test_join_lines_repairs_hyphenated_words_only():
    assert join_lines(["retrie-", "val works"]) == "retrieval works"
    assert join_lines(["BM-", "25"]) == "BM-25"


def test_title_of_uses_metadata_or_fallback():
    assert title_of(make_pdf(title="Attention Is All You Need"), "fallback") == "Attention Is All You Need"
    for bad in ("", "untitled", "paper.pdf"):
        assert title_of(make_pdf(title=bad), "fallback") == "fallback"


def test_extract_accepts_a_path(tmp_path, sample_pdf):
    path = tmp_path / "a.pdf"
    path.write_bytes(sample_pdf)
    blocks, sizes = extract(str(path))
    assert blocks and len(sizes) == 2


def test_password_protected_pdf_is_rejected():
    import pymupdf
    import pytest

    from app.pdftext import PasswordProtected

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text", fontsize=11)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    with pytest.raises(PasswordProtected, match="password-protected"):
        extract(data)
