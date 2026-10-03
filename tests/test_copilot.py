import pytest

from service.services import llm

SCRIPTED = "Why was this flagged?"


@pytest.fixture(scope="module")
def txn_id(client):
    return client.get("/api/alerts", params={"min_prob": 0.5, "limit": 1}).json()[0]["txn_id"]


def test_copilot_cites_the_matching_sop_section(client, txn_id):
    r = client.post("/api/copilot", json={
        "txn_id": txn_id, "question": "What are the card-not-present amount thresholds?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].strip()
    assert body["citations"], "copilot must return the retrieved SOP chunks"
    assert body["citations"][0]["doc"] == "Fraud_Operations_SOP.md"
    assert "3.1" in body["citations"][0]["text"]  # in-DB VectorDistance ranks section 3.1 first
    assert body["retrieval_ms"] > 0 and body["llm_ms"] > 0


def test_copilot_refuses_out_of_scope_questions(client, txn_id):
    r = client.post("/api/copilot", json={
        "txn_id": txn_id, "question": "What is the best recipe for nasi lemak?"})
    assert r.status_code == 200
    assert "cannot answer" in r.json()["answer"].lower()


def test_copilot_unknown_transaction_is_404(client):
    r = client.post("/api/copilot", json={"txn_id": 999999999, "question": SCRIPTED})
    assert r.status_code == 404


def test_demo_cache_answers_scripted_question_when_llm_is_down(client, txn_id, monkeypatch):
    def boom(system, user):
        raise RuntimeError("network down")

    monkeypatch.setattr(llm, "complete", boom)
    r = client.post("/api/copilot", json={"txn_id": txn_id, "question": SCRIPTED})
    assert r.status_code == 200
    assert r.json()["answer"].strip()
    off_script = client.post("/api/copilot", json={"txn_id": txn_id, "question": "Something unscripted?"})
    assert off_script.status_code == 502
