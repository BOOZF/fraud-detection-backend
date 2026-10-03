from pathlib import Path

import pytest

from service.services import rag

DOCS = Path(__file__).resolve().parent.parent / "docs"
PROBE = "pytest_purple_elephant.md"
PROBE_TEXT = b"""# Purple Elephant Handling Manual

## 9.1 Purple elephant protocol
When a purple elephant is detected near a vault, the duty officer must ring the amber bell exactly
seven times, then file form PE-77 with the zoology desk within twelve minutes. Never feed the elephant
any customer records.

## 9.2 Escalation of striped giraffe sightings
Striped giraffe sightings are escalated to the savanna supervisor and logged in the giraffe register.
"""


@pytest.fixture
def probe_cleanup(client):
    yield
    client.delete(f"/api/documents/{PROBE}")


def _upload(client, name: str, data: bytes, mime: str = "text/markdown"):
    return client.post("/api/documents", files={"file": (name, data, mime)})


def test_upload_lists_and_removes_a_document(client, probe_cleanup):
    r = _upload(client, PROBE, PROBE_TEXT)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["doc"] == PROBE and doc["kind"] == "md"
    assert doc["chunks"] == 2  # one chunk per numbered section
    assert doc["bytes"] == len(PROBE_TEXT)

    listed = {d["doc"]: d for d in client.get("/api/documents").json()}
    assert listed[PROBE]["chunks"] == 2
    assert "uploaded_at" in listed[PROBE]

    removed = client.delete(f"/api/documents/{PROBE}")
    assert removed.status_code == 200 and removed.json()["chunks"] == 2
    assert PROBE not in {d["doc"] for d in client.get("/api/documents").json()}
    assert client.delete(f"/api/documents/{PROBE}").status_code == 404


def test_uploading_the_same_name_replaces_instead_of_duplicating(client, probe_cleanup):
    _upload(client, PROBE, PROBE_TEXT)
    again = _upload(client, PROBE, PROBE_TEXT)
    assert again.status_code == 200
    listed = {d["doc"]: d for d in client.get("/api/documents").json()}
    assert listed[PROBE]["chunks"] == 2


def test_uploaded_document_becomes_searchable_by_the_copilot_retrieval(client, probe_cleanup):
    _upload(client, PROBE, PROBE_TEXT)
    hits = rag.retrieve("How many times must the amber bell be rung for a purple elephant?", k=2)
    assert hits[0]["doc"] == PROBE
    assert "amber bell" in hits[0]["text"]


def test_rejects_unsupported_empty_and_oversized_uploads(client):
    assert _upload(client, "malware.exe", b"MZ\x90", "application/octet-stream").status_code == 415
    assert _upload(client, "empty.md", b"", "text/markdown").status_code == 400
    assert _upload(client, "blank.txt", b"   \n\n  ", "text/plain").status_code == 400
    assert _upload(client, "huge.txt", b"a" * (26 * 1024 * 1024), "text/plain").status_code == 413


def test_pdf_is_extracted_page_by_page(client):
    name = "pytest_Fraud_Detection_SOP.pdf"
    try:
        r = client.post("/api/documents", files={"file": (name, (DOCS / "Fraud_Detection_SOP.pdf").read_bytes(), "application/pdf")})
        assert r.status_code == 200, r.text
        doc = r.json()
        assert doc["kind"] == "pdf" and doc["pages"] == 131
        assert doc["chunks"] > 100
        hits = rag.retrieve("Standard Operating Procedures for the Fraud Detection and National Security Branch", k=3)
        assert any(h["doc"] == name and h["section"].startswith("p") for h in hits)
    finally:
        client.delete(f"/api/documents/{name}")
