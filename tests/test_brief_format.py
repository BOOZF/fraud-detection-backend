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

def _full_reply(brief_service):
    return {"headline": "Escalate to a fraud lead now",
            "items": [{"question": q, "verdict": f"Verdict {n}", "points": [f"Point {n}"], "sources": []}
                      for n, q in enumerate(brief_service.QUESTIONS, 1)]}


def _fresh_txn(client):
    """An alert with no stored brief yet, so the model really is called."""
    txn = client.get("/api/alerts", params={"limit": 3}).json()[2]["txn_id"]
    db.execute(f"DELETE FROM copilot_briefs WHERE txn_id = {txn}")
    db.clear_cache()
    return txn


def test_an_incomplete_model_reply_is_asked_again_instead_of_shown_as_not_covered(client, monkeypatch):
    from service.services import brief as brief_service, llm
    txn, calls = _fresh_txn(client), []

    def flaky(system, user):
        calls.append(1)
        full = _full_reply(brief_service)
        return {**full, "items": full["items"][-1:]} if len(calls) == 1 else full  # first reply: only the last question

    monkeypatch.setattr(llm, "complete_json", flaky)
    items = client.post(f"/api/alerts/{txn}/brief").json()["items"]
    assert len(calls) == 1 + 4  # the whole set once, then each of the four missing questions on its own
    assert [i["verdict"] for i in items] == [f"Verdict {n}" for n in range(1, 6)]
    assert not any("Not covered" in i["verdict"] for i in items)


def test_a_model_that_keeps_returning_incomplete_replies_is_an_error_and_nothing_is_stored(client, monkeypatch):
    from service.services import brief as brief_service, llm
    txn = _fresh_txn(client)
    monkeypatch.setattr(llm, "complete_json", lambda s, u: {**_full_reply(brief_service), "items": []})
    assert client.post(f"/api/alerts/{txn}/brief").status_code == 502
    assert db.query(f"SELECT COUNT(*) FROM copilot_briefs WHERE txn_id = {txn}")[0][0] == 0


def test_when_the_model_says_a_question_is_not_covered_that_is_kept(client, monkeypatch):
    from service.services import brief as brief_service, llm
    txn = _fresh_txn(client)
    reply = _full_reply(brief_service)
    reply["items"][3] = {"question": brief_service.QUESTIONS[3], "verdict": "Not covered by policies",
                         "points": [brief_service.NOT_COVERED], "sources": []}
    monkeypatch.setattr(llm, "complete_json", lambda s, u: reply)
    items = client.post(f"/api/alerts/{txn}/brief").json()["items"]
    assert items[3]["verdict"] == "Not covered by policies" and items[0]["verdict"] == "Verdict 1"
