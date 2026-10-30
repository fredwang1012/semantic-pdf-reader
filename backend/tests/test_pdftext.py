
import pymupdf
import pytest

from app.pdftext import extract, join_lines, title_of

from .conftest import make_pdf


def _one_line_pdf(rotation: int, cropbox: pymupdf.Rect | None = None) -> bytes:
    """A 612 x 792 page with one line of text that starts at (72, 100) in the unrotated page."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Rotated pages keep their highlights", fontsize=12)
    if cropbox:
        page.set_cropbox(cropbox)
    page.set_rotation(rotation)
    data = doc.tobytes()
    doc.close()
    return data


def _displayed(bbox, rotation: int, w: float = 612, h: float = 792) -> tuple[float, float, float, float]:
    """Where a box in an unrotated w x h page is shown once /Rotate turns the page clockwise."""
    x0, y0, x1, y1 = bbox
    return {0: (x0, y0, x1, y1), 90: (h - y1, x0, h - y0, x1),
            180: (w - x1, h - y1, w - x0, h - y0), 270: (y0, w - x1, y1, w - x0)}[rotation]


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


# The text's start point (72, 100) shows at `start` in the displayed page; a turned line runs down, left or up from it.
@pytest.mark.parametrize("rotation, size, start", [
    (0, (612, 792), (72, 100)), (90, (792, 612), (692, 72)), (180, (612, 792), (540, 692)), (270, (792, 612), (100, 540)),
])
def test_line_box_is_where_a_rotated_page_shows_the_text(rotation, size, start):
    blocks, sizes = extract(_one_line_pdf(rotation))
    assert (sizes[0].width, sizes[0].height) == size
    [line] = blocks[0].lines
    x0, y0, x1, y1 = line.bbox
    assert x0 - 0.5 <= start[0] <= x1 + 0.5 and y0 - 0.5 <= start[1] <= y1 + 0.5
    assert (x1 - x0 > y1 - y0) == (rotation in (0, 180))  # a line is long along the direction it is read
    assert 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_every_line_box_turns_with_the_page(rotation):
    upright, _ = extract(make_pdf())
    turned, _ = extract(make_pdf(rotation=rotation))
    pairs = [(a, b) for p, q in zip(upright, turned, strict=True) for a, b in zip(p.lines, q.lines, strict=True)]
    assert pairs and all(a.text == b.text for a, b in pairs)
    for a, b in pairs:
        assert b.bbox == pytest.approx(_displayed(a.bbox, rotation), abs=0.11)


def test_line_boxes_are_relative_to_the_cropbox_of_a_rotated_page():
    # cropped to x 50..562 and y 30..742 (top-left origin): the start point is (22, 70) in the 512 x 712 crop
    blocks, sizes = extract(_one_line_pdf(90, cropbox=pymupdf.Rect(50, 30, 562, 742)))
    assert (sizes[0].width, sizes[0].height) == (712, 512)
    x0, y0, x1, y1 = blocks[0].lines[0].bbox
    assert x0 <= 712 - 70 <= x1 and y0 == pytest.approx(22, abs=0.5)


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
