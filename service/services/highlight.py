"""Serve a PDF with the cited chunk highlighted, so a citation shows exactly the passage it points at.

The stored chunk text is cleaned (headers removed, lines re-flowed), so it is located on the page by searching short
runs of consecutive words, not the whole chunk at once. Highlights are real PDF annotations: any PDF viewer shows them."""
import re
from functools import lru_cache
from pathlib import Path

import pymupdf

from . import rag

WINDOW_WORDS = 4  # long enough to be specific, short enough to survive line breaks and cleaning differences
MAX_PAGES = 3
KEY_COLOUR = (1.0, 0.6, 0.0)  # the sentence the answer relies on; the paragraph around it keeps the default yellow
MIN_SENTENCE_WORDS = 4  # fewer matched words than this is not evidence of where a sentence is
MAX_STRAY_WORDS = 4  # page words tolerated between two matched words (footnote marks, hyphenation)
MIN_HIT_POINTS = 12  # ignore slivers (list numbers and stray glyphs); a real phrase is much wider


def _phrases(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + WINDOW_WORDS]) for i in range(0, len(words) - WINDOW_WORDS + 1, WINDOW_WORDS)]


def _token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _same(page_token: str, target: str) -> bool:
    """Equal, or equal once a superscript footnote number glued to the word on the page ('alerts31') is ignored."""
    return page_token == target or (page_token.startswith(target) and page_token[len(target):].isdigit())


def _sentence_rects(page, sentence: str) -> list:
    """One rectangle per line of `sentence` where it really sits on the page, found by walking the page's words in
    reading order. Unlike searching for short phrases this cannot light up other clauses that repeat some of the same
    words, and it tolerates stray tokens in between (footnote marks) and a heading glued to the front."""
    wanted = [t for t in (_token(w) for w in sentence.split()) if t]
    page_words = [(_token(w[4]), pymupdf.Rect(w[:4])) for w in page.get_text("words")]
    page_words = [(t, r) for t, r in page_words if t]
    best: list[int] = []
    for skip in range(0, min(4, len(wanted))):  # the sentence may start with a heading or a clause number
        target = wanted[skip:]
        if len(target) < MIN_SENTENCE_WORDS:
            break
        for start, (token, _) in enumerate(page_words):
            if not _same(token, target[0]):
                continue
            matched, j, misses = [], 0, 0
            for k in range(start, min(len(page_words), start + 2 * len(target) + 6)):
                if _same(page_words[k][0], target[j]):
                    matched.append(k)
                    j, misses = j + 1, 0
                elif j + 1 < len(target) and _same(page_words[k][0], target[j + 1]):
                    matched.append(k)  # the sentence has a token the page does not (a footnote number): skip it
                    j, misses = j + 2, 0
                else:
                    misses += 1
                    if misses > MAX_STRAY_WORDS:
                        break
                if j >= len(target):
                    break
            if len(matched) > len(best):
                best = matched
    if len(best) < max(MIN_SENTENCE_WORDS, 0.6 * len(wanted)):
        return []
    lines: list[pymupdf.Rect] = []
    for k in best:
        rect = page_words[k][1]
        if lines and abs(lines[-1].y0 - rect.y0) < 3:
            lines[-1] |= rect
        else:
            lines.append(pymupdf.Rect(rect))
    return lines


@lru_cache(maxsize=64)
def _render(path: str, mtime: float, pages: tuple[int, ...], text: str, focus: str) -> bytes:
    phrases = _phrases(text)
    with pymupdf.open(path) as pdf:
        for number in pages:
            if not 1 <= number <= len(pdf):
                continue
            page = pdf[number - 1]
            for phrase in phrases:
                for quad in page.search_for(phrase, quads=True):
                    if quad.rect.width >= MIN_HIT_POINTS:
                        page.add_highlight_annot(quad).update()
            for rect in _sentence_rects(page, focus):  # drawn last, so the key sentence stands out from the paragraph
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=KEY_COLOUR)
                annot.update()
        return pdf.tobytes(garbage=0, deflate=True)


def highlighted_pdf(path: Path, section: str, text: str, focus: str = "") -> bytes:
    """`section` is the chunk's citation label ('p.5' or 'pp.5-6'); the highlight goes on those pages."""
    first = rag.page_of(section) or 1
    last = int(section.split("-")[-1]) if "-" in section else first
    pages = tuple(range(first, min(last, first + MAX_PAGES - 1) + 1))
    return _render(str(path), path.stat().st_mtime, pages, text, focus)
