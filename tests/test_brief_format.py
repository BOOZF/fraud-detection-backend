"""The brief is what a banker reads first: fixed structure, short, and identical every time it is opened."""
import pytest

from service import data, db, reasons


@pytest.fixture(scope="module")
def top_txn(client):
    return client.get("/api/alerts", params={"limit": 1}).json()[0]["txn_id"]


@pytest.fixture(scope="module")
def brief(client, top_txn):
    r = client.post(f"/api/alerts/{top_txn}/brief")
    assert r.status_code == 200, r.text
    return r.json()


def test_every_answer_is_a_short_verdict_plus_a_few_bullet_points(brief):
    assert len(brief["items"]) == 5
    for item in brief["items"]:
        assert 1 <= len(item["verdict"].split()) <= 10, item["verdict"]
        assert 1 <= len(item["points"]) <= 3
        assert all(1 <= len(p.split()) <= 30 for p in item["points"])
        assert item["answer"].strip()  # plain-text form kept for existing consumers


def test_headline_is_one_short_sentence(brief):
    assert 3 <= len(brief["headline"].split()) <= 25 and brief["headline"].count("\n") == 0


def test_risk_indicators_come_from_the_rules_not_the_language_model(client, brief, top_txn):
    txn, _, _ = data.get_txn(top_txn)
    assert brief["indicators"] == reasons.for_txn(txn)
    assert brief["facts"]["amount_myr"] == txn["amount_myr"] and brief["facts"]["channel"] == txn["channel"]


def test_reopening_an_alert_always_gives_the_same_brief_even_after_a_restart(client, top_txn, brief):
    db.clear_cache()  # what a service restart does to the in-memory cache
    again = client.post(f"/api/alerts/{top_txn}/brief").json()
    assert again == brief


def test_changing_the_knowledge_base_invalidates_stored_briefs(client, top_txn, brief):
    from service.services import brief as brief_service
    key_before = brief_service.kb_key()
    client.post("/api/documents", files={"file": ("pytest_kb_probe.md", b"# Probe\n\n## 1.1 Probe rule\nA probe rule long enough to be kept as a chunk of the knowledge base for this test only.\n" * 3, "text/markdown")})
    try:
        assert brief_service.kb_key() != key_before
    finally:
        client.delete("/api/documents/pytest_kb_probe.md")


# ---- an incomplete model reply must never be shown (or stored) as "not covered by the policies" ----

SENTENCE = "A standard response period does not exceed twelve weeks as outlined in the regulation text."
HIT = {"doc": "Fraud_Detection_SOP.pdf", "chunk_id": 78, "section": "p.45", "page": 45, "text": SENTENCE, "distance": 0.4, "score": 0.5}


def _policy_reply(brief_service, only=None):
    qs = only or brief_service.QUESTIONS[1:]
    return {"items": [{"question": q, "answerable": True, "verdict": f"Verdict {brief_service.QUESTIONS.index(q)}",
                       "points": ["Point."], "quote": SENTENCE, "sources": [78]} for q in qs]}


def _fresh_txn(client, monkeypatch):
    """An alert with no stored brief yet, so the model really is called; retrieval returns one fixed excerpt."""
    from service.services import rag
    monkeypatch.setattr(rag, "retrieve_many", lambda queries, k=4: [[dict(HIT)] for _ in queries])
    txn = client.get("/api/alerts", params={"limit": 3}).json()[2]["txn_id"]
    db.execute(f"DELETE FROM copilot_briefs WHERE txn_id = {txn}")
    db.clear_cache()
    return txn


def test_each_policy_question_is_asked_on_its_own_and_an_unusable_reply_is_asked_again(client, monkeypatch):
    from service.services import brief as brief_service, llm
    txn, calls = _fresh_txn(client, monkeypatch), {}

    def flaky(system, user, model=None):
        if system == brief_service.VERIFY_SYSTEM:
            return {"applies": True}
        q = next(q for q in brief_service.QUESTIONS[1:] if q in user)
        assert sum(other in user for other in brief_service.QUESTIONS) == 1  # never several questions in one request
        calls[q] = calls.get(q, 0) + 1
        return {"items": []} if calls[q] == 1 else _policy_reply(brief_service, [q])  # first reply per question is unusable

    monkeypatch.setattr(llm, "complete_json", flaky)
    items = client.post(f"/api/alerts/{txn}/brief").json()["items"]
    assert sorted(calls.values()) == [2, 2, 2, 2]
    assert [i["verdict"] for i in items[1:]] == [f"Verdict {n}" for n in range(1, 5)]
    assert not any(i["verdict"] == "Not covered by policies" for i in items)


def test_a_model_that_keeps_returning_incomplete_replies_is_an_error_and_nothing_is_stored(client, monkeypatch):
    from service.services import llm
    txn = _fresh_txn(client, monkeypatch)
    monkeypatch.setattr(llm, "complete_json", lambda s, u, model=None: {"items": []})
    assert client.post(f"/api/alerts/{txn}/brief").status_code == 502
    assert db.query(f"SELECT COUNT(*) FROM copilot_briefs WHERE txn_id = {txn}")[0][0] == 0


def test_when_the_model_says_a_question_is_not_covered_that_is_kept_and_the_rest_is_unaffected(client, monkeypatch):
    from service.services import brief as brief_service, llm
    txn = _fresh_txn(client, monkeypatch)
    reply = _policy_reply(brief_service)
    reply["items"][2] = {"question": brief_service.QUESTIONS[3], "answerable": False}
    monkeypatch.setattr(llm, "complete_json", lambda s, u, model=None: {"applies": True} if s == brief_service.VERIFY_SYSTEM else reply)
    items = client.post(f"/api/alerts/{txn}/brief").json()["items"]
    assert items[3]["verdict"] == "Not covered by policies" and items[3]["citations"] == []
    assert items[1]["verdict"] == "Verdict 1" and items[1]["citations"][0]["chunk_id"] == 78


def test_stored_briefs_are_invalidated_when_the_model_or_the_pipeline_changes(monkeypatch):
    from service.config import Settings
    from service.services import brief as brief_service
    base = brief_service.kb_key()
    monkeypatch.setattr(brief_service, "PIPELINE", "some-other-version")
    assert brief_service.kb_key() != base
    monkeypatch.undo()
    other = Settings(td_host="h", td_user="u", td_password="p", llm_model="another-model")
    monkeypatch.setattr(brief_service, "get_settings", lambda: other)
    assert brief_service.kb_key() != base
