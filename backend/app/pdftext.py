"""Text lines with their page and bounding box, grouped into blocks, from a PDF's text layer (PyMuPDF).

Coordinates are PDF points with a top-left origin, in the page as displayed: relative to the CropBox's top-left corner
and after the page's /Rotate. That is the space of `page.rect` (the page sizes) and of pdf.js's viewport at scale 1
(`page.getViewport({ scale: 1 })`), so the frontend draws a highlight by multiplying by its render scale. PyMuPDF
reports text boxes in the unrotated page, so `display_bbox` turns them with the page's rotation matrix. Converting here
rather than in the viewer keeps the viewer synchronous: it lays out pages and highlights from these numbers alone,
before pdf.js has loaded a page.
"""
import re
from dataclasses import dataclass, field

import pymupdf

_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-$")


class PasswordProtected(ValueError):
    """The PDF is encrypted and needs a password."""


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


def display_bbox(bbox, rotation_matrix: pymupdf.Matrix) -> tuple[float, float, float, float]:
    """A box in PyMuPDF's unrotated page space, turned into the displayed page (see above) and rounded to 0.1 pt."""
    r = pymupdf.Rect(bbox) * rotation_matrix
    return round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)


def extract(source: bytes | str) -> tuple[list[Block], list[PageSize]]:
    """(text blocks in reading order, page sizes) from a PDF given as bytes or a path.

    Blocks that are only a page number are dropped. A scanned PDF without a text layer gives no blocks.
    """
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, bytes) else pymupdf.open(source)
    blocks: list[Block] = []
    sizes: list[PageSize] = []
    with doc:
        if doc.needs_pass:
            raise PasswordProtected("password-protected PDFs are not supported")
        for pno, page in enumerate(doc, start=1):
            rect, rotation = page.rect, page.rotation_matrix
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
                    block.lines.append(Line(page=pno, bbox=display_bbox(ln["bbox"], rotation), text=text))
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
