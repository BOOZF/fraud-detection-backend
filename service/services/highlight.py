"""Serve a PDF with the cited chunk highlighted, so a citation shows exactly the passage it points at.

The stored chunk text is cleaned (headers removed, lines re-flowed), so it is located on the page by searching short
runs of consecutive words, not the whole chunk at once. Highlights are real PDF annotations: any PDF viewer shows them."""
from functools import lru_cache
from pathlib import Path

import pymupdf

from . import rag

WINDOW_WORDS = 4  # long enough to be specific, short enough to survive line breaks and cleaning differences
MAX_PAGES = 3
MIN_HIT_POINTS = 12  # ignore slivers (list numbers and stray glyphs); a real phrase is much wider


@lru_cache(maxsize=64)
def _render(path: str, mtime: float, pages: tuple[int, ...], text: str) -> bytes:
    words = text.split()
    phrases = [" ".join(words[i:i + WINDOW_WORDS]) for i in range(0, len(words) - WINDOW_WORDS + 1, WINDOW_WORDS)]
    with pymupdf.open(path) as pdf:
        for number in pages:
            if not 1 <= number <= len(pdf):
                continue
            page = pdf[number - 1]
            for phrase in phrases:
                for quad in page.search_for(phrase, quads=True):
                    if quad.rect.width >= MIN_HIT_POINTS:
                        page.add_highlight_annot(quad).update()
        return pdf.tobytes(garbage=0, deflate=True)


def highlighted_pdf(path: Path, section: str, text: str) -> bytes:
    """`section` is the chunk's citation label ('p.5' or 'pp.5-6'); the highlight goes on those pages."""
    first = rag.page_of(section) or 1
    last = int(section.split("-")[-1]) if "-" in section else first
    pages = tuple(range(first, min(last, first + MAX_PAGES - 1) + 1))
    return _render(str(path), path.stat().st_mtime, pages, text)
