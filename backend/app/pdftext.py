"""Text lines with their page and bounding box, grouped into blocks, from a PDF's text layer (PyMuPDF).

Coordinates are PDF points with a top-left origin, relative to the page's visible area: the same space pdf.js uses
for a viewport at scale 1, so the frontend can draw a highlight by multiplying by its render scale.
"""
import re
from dataclasses import dataclass, field

import pymupdf

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-$")


@dataclass
class Line:
    page: int                                  # 1-based, as pdf.js numbers pages
    bbox: tuple[float, float, float, float]    # x0, y0, x1, y1
    text: str


@dataclass
class Block:
    page: int
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return join_lines([ln.text for ln in self.lines])


@dataclass
class PageSize:
    width: float
    height: float


def join_lines(texts: list[str]) -> str:
    """Lines joined with spaces, except that a word hyphenated across a line break is joined back together.

    Only a lowercase continuation is joined ("retrie-" + "val" -> "retrieval"); "BM-" + "25" keeps its hyphen.
    """
    out = ""
    for t in texts:
        t = " ".join(t.split())
        if not t:
            continue
        if out and _HYPHEN_BREAK.search(out) and t[0].islower():
            out = out[:-1] + t
        elif out and _HYPHEN_BREAK.search(out) and t[0].isalnum():
            out += t
        else:
            out = f"{out} {t}" if out else t
    return out


def extract(source: bytes | str) -> tuple[list[Block], list[PageSize]]:
    """(text blocks in reading order, page sizes) from a PDF given as bytes or a path.

    Blocks that are only a page number are dropped. A scanned PDF without a text layer gives no blocks.
    """
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, bytes) else pymupdf.open(source)
    blocks: list[Block] = []
    sizes: list[PageSize] = []
    with doc:
        for pno, page in enumerate(doc, start=1):
            rect = page.rect
            sizes.append(PageSize(width=rect.width, height=rect.height))
            data = page.get_text("dict", sort=True)
            for b in data["blocks"]:
                if b.get("type") != 0:  # 1 = image block
                    continue
                block = Block(page=pno)
                for ln in b["lines"]:
                    text = "".join(span["text"] for span in ln["spans"])
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = ln["bbox"]
                    block.lines.append(Line(page=pno, bbox=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                                            text=text))
                if block.lines and not _PAGE_NUMBER.match(block.text):
                    blocks.append(block)
    return blocks, sizes


def title_of(source: bytes | str, fallback: str) -> str:
    """The PDF's metadata title when it looks like a real one, else `fallback` (the file name)."""
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, bytes) else pymupdf.open(source)
    with doc:
        title = " ".join((doc.metadata or {}).get("title", "").split())
    if len(title) < 4 or title.lower() in {"untitled", "microsoft word"} or title.lower().endswith((".pdf", ".doc", ".docx", ".dvi")):
        return fallback
    return title
