import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from service.services import llm, rag

QUESTIONS = [
    "Why was this transaction flagged?",
    "How urgent is it, and what is the response deadline?",
    "What should the analyst do next?",
    "Does this need a regulator report?",
    "How should the customer be contacted?",
]


@pytest.fixture(scope="module")
def top_txn(client):
    return client.get("/api/alerts", params={"limit": 1}).json()[0]["txn_id"]


PDF = "Fraud_Detection_SOP.pdf"
KNOWLEDGE_BASE = {p.name for p in (Path(__file__).resolve().parent.parent / "docs").glob("*.pdf")}
Q_VOLUNTARY = "Is an alien's participation in an administrative investigation voluntary?"  # only page 45 says so
Q_REFERRAL = "What is the Fraud Referral Sheet used for?"  # pages 6, 15, 56, 71, 106 of the PDF mention it
REFERRAL_PAGES = {6, 15, 56, 71, 106}


def pages_of(hit):
    first = rag.page_of(hit["section"])
    last = int(hit["section"].split("-")[-1]) if "-" in hit["section"] else first
    return set(range(first, last + 1))


def test_retrieval_is_safe_when_requests_overlap(client):
    def run(q):
        return rag.retrieve_many([q], k=3)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(run, [Q_VOLUNTARY, Q_REFERRAL, Q_VOLUNTARY, Q_REFERRAL]))
    for hits, want in zip(got, [{45}, REFERRAL_PAGES, {45}, REFERRAL_PAGES]):
        assert any(pages_of(h) & want for h in hits), [h["section"] for h in hits]


def test_retrieve_many_returns_one_ranked_list_per_question(client):
    lists = rag.retrieve_many([Q_VOLUNTARY, Q_REFERRAL], k=3)
    assert len(lists) == 2 and all(len(l) == 3 for l in lists)
    assert 45 in pages_of(lists[0][0])
    assert any(pages_of(h) & REFERRAL_PAGES for h in lists[1])
    for hits in lists:
        assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
        assert {h["doc"] for h in hits} == {PDF}


def test_brief_answers_the_standard_questions_with_pdf_page_citations(client, top_txn):
    r = client.post(f"/api/alerts/{top_txn}/brief")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["txn_id"] == top_txn and b["priority"] == "P1"  # the highest-probability alert is >= 90%
    assert b["headline"].strip()
    assert [i["question"] for i in b["items"]] == QUESTIONS
    for item in b["items"]:
        assert item["answer"].strip()
        assert len(item["answer"].split()) <= 90, "answers are meant to be brief (prompt asks for <= 60 words)"
        if item["verdict"] == "Not covered by policies":
            assert item["citations"] == []  # never a page under an answer that page does not support
        else:
            assert item["citations"] or item["question"] == QUESTIONS[0], f"no citation for: {item['question']}"
        for c in item["citations"]:
            assert c["doc"] in KNOWLEDGE_BASE  # only the ingested PDFs are knowledge sources
            assert isinstance(c["chunk_id"], int) and c["text"].strip()
            assert 1 <= c["page"] <= 131 and c["section"].startswith("p")
            # the evidence shown with an answer really is on the cited page
            assert "".join(ch for ch in item["evidence"].lower() if ch.isalnum()) in "".join(ch for ch in c["text"].lower() if ch.isalnum())


def test_pdf_citations_carry_the_page_number_to_open():
    assert rag.page_of("p.12") == 12
    assert rag.page_of("pp.54-55") == 54
    assert rag.page_of("3.1") is None  # section labels of non-PDF documents have no page
    assert rag.page_of("part 2") is None


def test_brief_is_cached_after_the_first_call(client, top_txn):
    first = client.post(f"/api/alerts/{top_txn}/brief").json()
    t0 = time.perf_counter()
    again = client.post(f"/api/alerts/{top_txn}/brief").json()
    assert time.perf_counter() - t0 < 0.5
    assert again == first


def test_brief_for_an_unknown_transaction_is_404(client):
    assert client.post("/api/alerts/999999999/brief").status_code == 404


def test_brief_reports_502_when_the_llm_is_down(client, monkeypatch):
    from service import db

    def boom(system, user, model=None):
        raise RuntimeError("network down")

    db.clear_cache()
    monkeypatch.setattr(llm, "complete_json", boom)
    top = client.get("/api/alerts", params={"limit": 1}).json()[0]["txn_id"]
    db.execute(f"DELETE FROM copilot_briefs WHERE txn_id = {top}")  # an alert whose brief was never generated
    assert client.post(f"/api/alerts/{top}/brief").status_code == 502


def test_the_pdf_is_embedded_well_enough_to_find_a_passage_from_one_of_its_sentences(client):
    """Quality gate for chunking + embedding + search: quote one sentence from the middle of sampled chunks and
    the chunk it came from must come back in the top 3 (measured on this PDF: ~0.99; vector-only was ~0.87)."""
    import random
    import re

    from service import db

    rows = db.query_df("SELECT chunk_id, text FROM policy_chunks ORDER BY chunk_id")
    rng = random.Random(7)
    sample = rng.sample(list(rows.itertuples(index=False)), 40)
    queries, gold = [], []
    for r in sample:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", r.text) if 60 <= len(s.strip()) <= 220]
        if sentences:
            queries.append(sentences[len(sentences) // 2])
            gold.append(int(r.chunk_id))
    assert len(queries) >= 30
    found = [g in {h["chunk_id"] for h in hits} for g, hits in zip(gold, rag.retrieve_many(queries, k=3))]
    assert sum(found) / len(found) >= 0.95, f"recall@3 = {sum(found) / len(found):.2f}"
