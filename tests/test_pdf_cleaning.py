import re
from pathlib import Path

from service.services import documents

REAL_PDF = Path(__file__).resolve().parent.parent / "docs" / "Fraud_Detection_SOP.pdf"

HEADER = ["Fraud Detection Branch", "Standard Operating Procedure"]


def synthetic_pages(n=12):
    """n pages with the same running header/footer, plus body text that is unique per page."""
    pages = []
    for i in range(1, n + 1):
        pages.append("\n".join([
            "US.Cm:twhlp",            # OCR-garbled logo that precedes the running header
            "mdl mmlgmioll",
            *HEADER,
            f"Section {i}. The officer carefully reviews case number {i} and records the final result in the file.",
            f"The result for case {i} is sent to the senior officer for super-",
            f"vision within {i} working days of receipt and is then archived for later audit.",
            "(b )(7)( e)",
            str(i),                   # printed page number
            "AILA InfoNet Doc. No. 14092342. (Posted 9/23/14)",
        ]))
    return pages


def test_running_headers_and_footers_are_removed_but_body_text_is_kept():
    out = documents.clean_pdf_pages(synthetic_pages())
    joined = "\n".join(out)
    for line in [*HEADER, "AILA InfoNet"]:
        assert line not in joined
    assert "Section 7. The officer carefully reviews case number 7" in out[6]


def test_redaction_markers_are_removed_whatever_the_ocr_spacing():
    pages = synthetic_pages(10)
    pages[0] += "\nSensitive step (b)(7)(e) continues (b )(5) here (b) (7) (e) and ends."
    out = documents.clean_pdf_pages(pages)
    assert not re.search(r"\(\s*b\s*\)\s*\(\s*\d\s*\)", "\n".join(out))
    assert "Sensitive step" in out[0] and "and ends." in out[0]


def test_ocr_garbage_before_the_running_header_is_dropped():
    out = documents.clean_pdf_pages(synthetic_pages())
    assert "twhlp" not in "\n".join(out) and "mmlgmioll" not in "\n".join(out)


def test_hyphenated_line_breaks_are_joined_and_lines_reflowed():
    out = documents.clean_pdf_pages(synthetic_pages())
    assert "sent to the senior officer for supervision within 1 working days of receipt" in out[0]
    assert "super-" not in out[0]


def test_bare_page_numbers_are_removed():
    out = documents.clean_pdf_pages(synthetic_pages())
    assert not any(line.strip().isdigit() for page in out for line in page.split("\n"))


def test_ocr_variants_and_changing_page_suffixes_of_a_running_line_are_removed_too():
    pages = synthetic_pages(12)
    for i, page in enumerate(pages):
        link = "Return to Table of Contents." if i % 2 else "Return to Table of Contems."  # OCR typo on half the pages
        pages[i] = page + f"\n{link}\nVersion 3.0 For Official Use Only M-{i + 2}"
    joined = "\n".join(documents.clean_pdf_pages(pages))
    assert "Return to Table of" not in joined
    assert "Version 3.0 For Official Use Only" not in joined


def test_boilerplate_glued_to_the_end_of_a_content_line_is_removed():
    pages = synthetic_pages(12)
    for i, page in enumerate(pages):
        pages[i] = page + "\nVersion 3.0 For Official Use Only"  # the footer on its own line, on every page
    pages[3] += "\nContents entry about the review procedure ........ 18 Version 3.0 For Official Use Only iii"
    joined = "\n".join(documents.clean_pdf_pages(pages))
    assert "Contents entry about the review procedure" in joined and "18" in joined
    assert "Version 3.0 For Official Use Only" not in joined


def test_short_symbol_heavy_lines_are_dropped_but_short_real_headings_survive():
    pages = synthetic_pages(12)
    pages[5] += "\nU.S.CJtl=$hl.fc 'e' \" Oil\n2.26 Request Documentary Evidence\nA legitimate sentence follows the heading and carries the content."
    out = documents.clean_pdf_pages(pages)[5]
    assert "CJtl" not in out
    assert "2.26 Request Documentary Evidence" in out and "A legitimate sentence follows" in out


def test_ocr_junk_token_glued_to_the_start_of_a_content_line_is_stripped_but_real_abbreviations_survive():
    pages = synthetic_pages(12)
    pages[7] += ("\nU.S.CJtl=$hl.fc Section 4. Request for Assistance The objective of this section is to outline the steps."
                 "\nU.S. Citizenship and Immigration Services cites 8 CFR 292.3 and 8 U.S.C. 1324c(e) in this paragraph of the manual.")
    out = documents.clean_pdf_pages(pages)[7]
    assert "CJtl" not in out and "Section 4. Request for Assistance The objective" in out
    assert "U.S. Citizenship and Immigration Services cites 8 CFR 292.3 and 8 U.S.C. 1324c(e)" in out


def test_a_junk_token_on_its_own_line_is_dropped_instead_of_being_glued_to_the_next_sentence():
    pages = synthetic_pages(12)
    pages[9] += ("\nU.S.CJtl=$hl.fc\nSection 4. Request for Assistance The objective of this section is to outline the proper method.")
    out = documents.clean_pdf_pages(pages)[9]
    assert "CJtl" not in out
    assert "Section 4. Request for Assistance The objective of this section" in out


def test_a_short_document_is_not_mistaken_for_having_running_headers():
    pages = ["Fraud Detection Branch\nIntro text about the policy.", "Other page.\nFraud Detection Branch is named here."]
    out = documents.clean_pdf_pages(pages)
    assert "Fraud Detection Branch" in "\n".join(out)  # with 2 pages nothing is 'repeated on many pages'


def test_the_real_pdf_yields_clean_readable_chunks():
    chunks, pages = documents.extract("pdf", REAL_PDF.read_bytes())
    assert pages == 131 and 100 < len(chunks) < 400
    text = "\n".join(c.text for c in chunks)
    assert not re.search(r"\(\s*b\s*\)\s*\(\s*\d\s*\)", text), "redaction marks must be gone"
    assert "AILA InfoNet" not in text and "Return to Table of Contents" not in text
    assert all(len(c.text) <= documents.CHUNK_CHARS for c in chunks)
    assert "Return to Table of Conte" not in text and "Version 3.0 For Official Use Only" not in text
    letters = sorted(sum(ch.isalpha() for ch in c.text) / len(c.text) for c in chunks)
    # Measured after cleaning: mean 0.78, 5th percentile 0.69. Tables and citations legitimately have more digits.
    assert sum(letters) / len(letters) > 0.75
    assert sum(r < 0.7 for r in letters) / len(letters) < 0.10
    firsts = [documents_page(c.section) for c in chunks]
    assert firsts == sorted(firsts), "chunks stay in page order"
    assert len(set(firsts)) > 60, "the chunks span the whole document, not just its start"


def documents_page(section: str) -> int:
    from service.services import rag
    return rag.page_of(section)
