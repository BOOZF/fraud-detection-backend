"""Chat answers reach the user in the fixed structure, however the model replied."""
import json
from types import SimpleNamespace

import pytest

from service import guardrails
from service.services import llm, rag

GOOD = {"kind": "alert", "summary": "This P1 alert looks like card-not-present fraud.",
        "sections": [{"heading": "Why it was flagged", "points": ["Foreign transaction."]}, {"heading": "Key facts", "points": ["RM 1,028.23"]}]}


def msg(content):
    return SimpleNamespace(content=content, tool_calls=None)


@pytest.fixture
def ask(client, monkeypatch):
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")

    def go(replies, text="explain this"):
        it = iter(replies)
        calls = []
        monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: (calls.append((messages, kw)), next(it))[1])
        body = client.post("/api/chat", json={"messages": [{"role": "user", "content": text}]}).json()
        return body, calls
    return go


def test_the_json_the_model_returns_is_rendered_into_the_fixed_layout(ask):
    body, _ = ask([msg(json.dumps(GOOD))])
    assert body["answer"] == "This P1 alert looks like card-not-present fraud.\n\n**Key facts**\n- RM 1,028.23\n\n**Why it was flagged**\n- Foreign transaction."


def test_the_model_is_asked_for_json_every_turn(ask):
    _, calls = ask([msg(json.dumps(GOOD))])
    assert all(kw.get("json_mode") for _, kw in calls)


def test_a_reply_that_is_not_json_is_asked_again_once(ask):
    body, calls = ask([msg("Here is a friendly paragraph."), msg(json.dumps(GOOD))])
    assert len(calls) == 2 and body["answer"].startswith("This P1 alert")


def test_if_it_still_is_not_json_the_answer_is_still_in_the_structure_as_a_summary(ask):
    body, _ = ask([msg("A friendly paragraph."), msg("Another friendly paragraph.")])
    assert body["answer"] == "Another friendly paragraph."
    assert "|" not in body["answer"] and "**" not in body["answer"]


def test_citations_still_follow_the_references_inside_the_rendered_points(client, monkeypatch, ask):
    hit = {"doc": "a.pdf", "chunk_id": 9, "section": "p.3", "page": 3, "text": "t", "distance": 0.3, "score": 0.5}
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [dict(hit)])
    tool = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(name="search_policies", arguments='{"query": "x"}'))])
    reply = {"kind": "policy", "summary": "Yes.", "sections": [{"heading": "What the policy says", "points": ["Acknowledge within 3 days [a.pdf p.3]."]}]}
    body, _ = ask([tool, msg(json.dumps(reply))])
    assert [c["chunk_id"] for c in body["citations"]] == [9]


# ---- facts about the focused alert are filled in by the code, not left to the model ----

def test_key_facts_and_reason_codes_of_the_focused_alert_come_from_the_data_every_time(client, monkeypatch):
    from service import data, reasons
    top = client.get("/api/alerts", params={"limit": 1}).json()[0]["txn_id"]
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")
    only_policy = {"kind": "alert", "summary": "Looks like card-not-present fraud.",
                   "sections": [{"heading": "Key facts", "points": ["the model made this up"]},
                                {"heading": "Policy guidance", "points": ["Alert the cardholder [a.pdf p.1]."]}]}
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: msg(json.dumps(only_policy)))
    answer = client.post("/api/chat", json={"messages": [{"role": "user", "content": "explain this alert for me"}], "alert_id": top}).json()["answer"]
    txn, prob, _ = data.get_txn(top)
    labels = [l for l in answer.splitlines() if l.startswith("**")]
    assert labels == ["**Key facts**", "**Why it was flagged**", "**Policy guidance**"]
    assert "the model made this up" not in answer
    assert f"RM {txn['amount_myr']:,.2f}" in answer and txn["channel"] in answer and f"{prob:.3f}" in answer
    for reason in reasons.for_txn(txn):
        assert reason in answer


def test_a_data_answer_is_not_changed_by_a_focused_alert(client, monkeypatch):
    top = client.get("/api/alerts", params={"limit": 1}).json()[0]["txn_id"]
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")
    reply = {"kind": "data", "summary": "Alerts by channel.", "table": {"headers": ["Channel", "Alerts"], "rows": [["ATM", "1"]]}}
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: msg(json.dumps(reply)))
    answer = client.post("/api/chat", json={"messages": [{"role": "user", "content": "alerts per channel"}], "alert_id": top}).json()["answer"]
    assert "Key facts" not in answer and "| ATM | 1 |" in answer


def test_the_prompt_asks_for_readable_table_headers_and_formatted_figures():
    from service.services import chat
    assert "Total amount (RM)" in chat.SYSTEM and "thousands separators" in chat.SYSTEM
