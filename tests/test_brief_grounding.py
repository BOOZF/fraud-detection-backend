"""The brief may only cite a page for what that page really says.

A fake model and fixed excerpts make these deterministic. They reproduce the failure seen with the real corpus: an
excerpt about an applicant's evidence-request deadline sitting under an answer 'respond immediately'."""
import pytest

from service import data, db, reasons
from service.services import brief, llm, rag

TWELVE_WEEKS = ("A standard response period does not exceed twelve weeks as outlined in 8 CFR. If the deadline for "
                "response to the request for evidence is not met, the officer will record the fact and decide.")
HIT = {"doc": "Fraud_Detection_SOP.pdf", "chunk_id": 78, "section": "p.45", "page": 45, "text": TWELVE_WEEKS,
       "distance": 0.46, "score": 0.48}


@pytest.fixture
def txn_id(client):
    return client.get("/api/alerts", params={"limit": 1}).json()[0]["txn_id"]


@pytest.fixture
def one_excerpt(monkeypatch):
    monkeypatch.setattr(rag, "retrieve_many", lambda queries, k=4: [[dict(HIT)] for _ in queries])


def run(monkeypatch, txn_id, item_for, applies=True):
    """Build the brief with a model that answers every LLM question with item_for(question_index). The relevance
    verifier (a separate model call) answers `applies`."""
    def fake(system, user, model=None):
        if system == brief.VERIFY_SYSTEM:
            return {"applies": applies, "why": "test"}
        asked = [q for q in brief.QUESTIONS if q in user] or brief.QUESTIONS
        return {"items": [{"question": q, **item_for(brief.QUESTIONS.index(q))} for q in asked]}

    monkeypatch.setattr(llm, "complete_json", fake)
    return brief._generate(txn_id)


def covered(quote, verdict="Standard response period is twelve weeks"):
    return {"answerable": True, "verdict": verdict, "points": ["The SOP sets a response period."], "quote": quote, "sources": [78]}


def test_a_quote_that_is_not_in_the_cited_excerpt_is_rejected_and_no_page_is_cited(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: covered("Respond to every fraud alert within fifteen minutes of detection."))
    for item in b["items"][1:]:
        assert item["verdict"] == "Not covered by policies"
        assert item["citations"] == []  # nothing is cited for an answer the page does not support
        assert brief.NOT_COVERED in item["points"]


def test_a_quote_found_in_the_excerpt_keeps_its_citation_and_is_shown_as_evidence(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: covered("A standard response period does not exceed twelve weeks"))
    item = b["items"][1]
    assert item["verdict"] == "Standard response period is twelve weeks"
    assert [c["chunk_id"] for c in item["citations"]] == [78]
    assert "twelve weeks" in item["evidence"]


def test_quote_matching_ignores_case_spacing_and_punctuation(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: covered("a STANDARD response  period does not exceed twelve-weeks, as outlined in 8 CFR"))
    assert b["items"][1]["citations"]


def test_a_model_that_says_not_answerable_is_believed_even_with_a_quote(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: {**covered("A standard response period does not exceed twelve weeks"), "answerable": False})
    assert all(i["verdict"] == "Not covered by policies" and i["citations"] == [] for i in b["items"][1:])


def test_a_very_short_quote_is_not_accepted_as_evidence(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: covered("twelve weeks"))  # 'twelve weeks' appears in the text but proves nothing
    assert b["items"][1]["citations"] == []


def test_why_flagged_comes_from_the_reason_codes_and_the_model_is_not_asked_it(one_excerpt, monkeypatch, txn_id):
    asked = []

    def fake(system, user, model=None):
        if system == brief.VERIFY_SYSTEM:
            return {"applies": True}
        asked.append(user)
        return {"items": [{"question": q, **covered("A standard response period does not exceed twelve weeks")} for q in brief.QUESTIONS[1:]]}

    monkeypatch.setattr(llm, "complete_json", fake)
    b = brief._generate(txn_id)
    txn, prob, _ = data.get_txn(txn_id)
    first = b["items"][0]
    assert first["question"] == brief.QUESTIONS[0]
    assert reasons.for_txn(txn)[0] in first["points"][0]
    assert f"{prob:.0%}" in first["verdict"]
    assert first["citations"] == []
    assert all(brief.QUESTIONS[0] not in u for u in asked)


def test_an_uncovered_urgency_question_still_tells_the_analyst_the_model_priority(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: {"answerable": False})
    txn, prob, _ = data.get_txn(txn_id)
    urgency = b["items"][1]
    assert any(f"Model priority: {brief.priority_of(prob)}" in p for p in urgency["points"])


def test_the_headline_is_built_from_the_alert_not_invented_by_the_model(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: {"answerable": False})
    txn, prob, _ = data.get_txn(txn_id)
    assert brief.priority_of(prob) in b["headline"]
    assert 3 <= len(b["headline"].split()) <= 25


def test_a_verified_quote_that_does_not_apply_to_this_alert_is_not_covered(one_excerpt, monkeypatch, txn_id):
    """The failure from the field: a real sentence (an applicant's twelve-week evidence deadline) used to answer 'how
    urgent is this card-fraud alert'. The quote is genuine, so only a relevance check can catch it."""
    b = run(monkeypatch, txn_id, lambda i: covered("A standard response period does not exceed twelve weeks"), applies=False)
    for item in b["items"][1:]:
        assert item["verdict"] == "Not covered by policies" and item["citations"] == [] and item["covered"] is False


def test_the_verifier_sees_the_question_the_transaction_and_the_quoted_passage_only_for_answers_that_passed_the_quote_check(one_excerpt, monkeypatch, txn_id):
    seen = []

    def fake(system, user, model=None):
        if system == brief.VERIFY_SYSTEM:
            seen.append(user)
            return {"applies": True}
        q = next(q for q in brief.QUESTIONS[1:] if q in user)
        return {"items": [{"question": q, **(covered("A standard response period does not exceed twelve weeks") if "urgent" in q else covered("A made up sentence that is nowhere in the excerpt at all"))}]}

    monkeypatch.setattr(llm, "complete_json", fake)
    brief._generate(txn_id)
    assert len(seen) == 1  # only the urgency answer had a real quote
    assert "How urgent is it" in seen[0] and "twelve weeks" in seen[0] and "card payment" in seen[0].lower()


def test_a_verifier_that_cannot_be_reached_is_an_error_not_a_silent_accept(one_excerpt, monkeypatch, txn_id):
    def fake(system, user, model=None):
        if system == brief.VERIFY_SYSTEM:
            raise RuntimeError("network down")
        return {"items": [{"question": q, **covered("A standard response period does not exceed twelve weeks")} for q in brief.QUESTIONS[1:]]}

    monkeypatch.setattr(llm, "complete_json", fake)
    with pytest.raises(RuntimeError):
        brief._generate(txn_id)


def test_each_item_says_whether_it_is_covered_so_the_page_can_explain_gaps(one_excerpt, monkeypatch, txn_id):
    b = run(monkeypatch, txn_id, lambda i: covered("A standard response period does not exceed twelve weeks") if i == 1 else {"answerable": False})
    assert [i["covered"] for i in b["items"]] == [True, True, False, False, False]
    assert b["coverage"] == {"covered": 1, "total": 4}  # questions 2-5 are the ones that need a policy


def test_a_reply_that_is_not_valid_json_is_asked_again(one_excerpt, monkeypatch, txn_id):
    import json

    tries = {}

    def fake(system, user, model=None):
        if system == brief.VERIFY_SYSTEM:
            return {"applies": True}
        q = next(q for q in brief.QUESTIONS[1:] if q in user)
        tries[q] = tries.get(q, 0) + 1
        if tries[q] == 1:  # the model rambles on and the JSON is cut off
            raise json.JSONDecodeError("Unterminated string", "{", 1)
        return {"items": [{"question": q, **covered("A standard response period does not exceed twelve weeks")}]}

    monkeypatch.setattr(llm, "complete_json", fake)
    b = brief._generate(txn_id)
    assert sorted(tries.values()) == [2, 2, 2, 2]
    assert all(i["covered"] for i in b["items"])
