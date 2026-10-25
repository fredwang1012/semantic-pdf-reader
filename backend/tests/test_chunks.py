from app.chunks import CHUNK_MAX, CHUNK_MIN, make_chunks
from app.pdftext import Block, Line, extract


def _line(page: int, text: str, y: float = 10) -> Line:
    return Line(page=page, bbox=(0, y, 100, y + 10), text=text)


def _block(page: int, texts: list[str]) -> Block:
    return Block(page=page, lines=[_line(page, t, 10 + 12 * i) for i, t in enumerate(texts)])


def test_chunk_sizes_ids_and_rects(sample_pdf):
    blocks, sizes = extract(sample_pdf)
    chunks = make_chunks(blocks)
    assert [c.id for c in chunks] == [f"c{i}" for i in range(len(chunks))]
    for c in chunks:
        assert len(c.text) <= CHUNK_MAX or len(c.rects) == 1
        for r in c.rects:
            assert len(r) == 5
            assert 1 <= r[0] <= len(sizes)


def test_chunks_cover_every_word_in_order(sample_pdf):
    blocks, _ = extract(sample_pdf)
    chunks = make_chunks(blocks)
    assert " ".join(c.text for c in chunks).split() == " ".join(b.text for b in blocks).split()


def test_long_block_is_split_without_losing_lines():
    lines = [f"line number {i} " + "x" * 80 for i in range(40)]  # ~3600 chars in one block
    chunks = make_chunks([_block(1, lines)])
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_MAX for c in chunks)
    assert " ".join(c.text for c in chunks) == " ".join(lines)


def test_chunk_closes_at_a_page_break_once_long_enough():
    long_line = "a" * (CHUNK_MIN + 10)
    chunks = make_chunks([_block(1, [long_line]), _block(2, ["next page text"])])
    assert [c.page for c in chunks] == [1, 2]
    assert {r[0] for r in chunks[0].rects} == {1}


def test_short_chunk_carries_over_a_page_break():
    chunks = make_chunks([_block(1, ["A heading"]), _block(2, ["The body that follows it."])])
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert {r[0] for r in chunks[0].rects} == {1, 2}
