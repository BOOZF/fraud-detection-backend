"""Clicking a citation opens the PDF at the cited page WITH the cited passage highlighted."""
import pymupdf
import pytest

from service import db
from service.services import rag

PDFS = ["Fraud_Detection_SOP.pdf", "uj-fraud-prevention-procedures-sop-mar-2021.pdf"]


def _chunk(doc: str, offset: int = 3):
    rows = db.query(f"SELECT chunk_id, section FROM policy_chunks WHERE doc = '{doc}' ORDER BY chunk_id")
    return rows[offset]


def _highlights(pdf_bytes: bytes, page_number: int) -> list:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return [a for a in pdf[page_number - 1].annots() if a.type[1] == "Highlight"]


@pytest.mark.parametrize("doc", PDFS)
def test_cited_chunk_is_highlighted_on_its_page(client, doc):
    chunk_id, section = _chunk(doc)
    page = rag.page_of(section)
    r = client.get(f"/api/documents/{doc}/file", params={"chunk": chunk_id})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert len(_highlights(r.content, page)) >= 2  # several phrases of the chunk, not a stray single hit


def test_without_a_chunk_the_original_file_is_served_untouched(client):
    doc = PDFS[1]
    chunk_id, section = _chunk(doc)
    plain = client.get(f"/api/documents/{doc}/file").content
    assert _highlights(plain, rag.page_of(section)) == []


def test_chunk_of_another_document_is_rejected(client):
    chunk_id, _ = _chunk(PDFS[0])
    assert client.get(f"/api/documents/{PDFS[1]}/file", params={"chunk": chunk_id}).status_code == 404
    assert client.get(f"/api/documents/{PDFS[1]}/file", params={"chunk": 999999999}).status_code == 404
