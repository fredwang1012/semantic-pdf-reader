"""Search chunks: runs of PDF lines that keep each line's page and box, so a hit can be highlighted in place.

Chunks are small (about CHUNK_TARGET characters) because a hit is shown as a highlight: a long chunk would light up
half a page. They break at block (paragraph) boundaries where they can, and between lines when one block is too long.
A chunk closes at a page break once it holds `min_chars`; a shorter one (a heading at the bottom of a page)
continues onto the next page, and its rects then span both pages.
"""
from dataclasses import asdict, dataclass

from .pdftext import Block, Line, join_lines

CHUNK_TARGET, CHUNK_MIN, CHUNK_MAX = 600, 250, 1000   # characters


@dataclass
class Chunk:
    id: str
    page: int                          # page of the first line
    text: str
    rects: list[list[float]]           # [page, x0, y0, x1, y1] per line

    def to_json(self) -> dict:
        return asdict(self)


def _pieces(block: Block, max_chars: int) -> list[list[Line]]:
    """The block's lines, cut into runs of at most `max_chars` (a single over-long line stays whole)."""
    runs: list[list[Line]] = [[]]
    size = 0
    for ln in block.lines:
        n = len(ln.text) + 1
        if runs[-1] and size + n > max_chars:
            runs.append([])
            size = 0
        runs[-1].append(ln)
        size += n
    return [r for r in runs if r]


def make_chunks(blocks: list[Block], target: int = CHUNK_TARGET, min_chars: int = CHUNK_MIN,
                max_chars: int = CHUNK_MAX) -> list[Chunk]:
    """Greedy packing of block pieces: a chunk closes once it has `min_chars` and the next piece would pass `target`,
    or when the next piece would pass `max_chars`, or at a page break once it has `min_chars`."""
    chunks: list[Chunk] = []
    current: list[Line] = []
    size = 0

    def close():
        nonlocal current, size
        if current:
            chunks.append(Chunk(id=f"c{len(chunks)}", page=current[0].page, text=join_lines([ln.text for ln in current]),
                                rects=[[ln.page, *ln.bbox] for ln in current]))
        current, size = [], 0

    for block in blocks:
        for piece in _pieces(block, max_chars):
            n = sum(len(ln.text) + 1 for ln in piece)
            page_break = bool(current) and piece[0].page != current[-1].page
            if current and (size + n > max_chars or (size >= min_chars and (size + n > target or page_break))):
                close()
            current.extend(piece)
            size += n
    close()
    return chunks
