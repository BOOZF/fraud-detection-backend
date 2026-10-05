"""Chat shows a citation only for a passage the answer actually refers to."""
import json
from types import SimpleNamespace

import pytest

from service import guardrails
from service.services import llm, rag

HITS = [
    {"doc": "Fraud_Detection_SOP.pdf", "chunk_id": 1, "section": "p.45", "page": 45, "text": "Deadline text.", "distance": 0.4, "score": 0.5},
    {"doc": "Fraud_Detection_SOP.pdf", "chunk_id": 2, "section": "pp.58-59", "page": 58, "text": "Other text.", "distance": 0.5, "score": 0.4},
    {"doc": "uj-fraud-prevention.pdf", "chunk_id": 3, "section": "p.6", "page": 6, "text": "University text.", "distance": 0.5, "score": 0.4},
]


def chat_with(client, monkeypatch, final_text: str):
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [dict(h) for h in HITS])
    turns = iter([
        SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="search_policies", arguments=json.dumps({"query": "deadline"})))]),
        SimpleNamespace(content=json.dumps({"kind": "policy", "summary": "Answer.", "sections": [{"heading": "What the policy says", "points": [final_text]}]}), tool_calls=None),
    ])
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: next(turns))
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": "What is the deadline?"}]}).json()


def test_no_citation_chips_when_the_answer_cites_nothing(client, monkeypatch):
    body = chat_with(client, monkeypatch, "I cannot find a response deadline in the uploaded policies.")
    assert body["citations"] == []


def test_only_the_cited_passage_is_shown(client, monkeypatch):
    body = chat_with(client, monkeypatch, "The policy sets twelve weeks [Fraud_Detection_SOP.pdf p.45].")
    assert [c["chunk_id"] for c in body["citations"]] == [1]


def test_a_page_inside_a_cited_range_matches_the_chunk_that_spans_it(client, monkeypatch):
    body = chat_with(client, monkeypatch, "See [Fraud_Detection_SOP.pdf p.59] and [uj-fraud-prevention.pdf p.6].")
    assert sorted(c["chunk_id"] for c in body["citations"]) == [2, 3]


def test_the_same_page_of_a_different_document_does_not_match(client, monkeypatch):
    body = chat_with(client, monkeypatch, "See [uj-fraud-prevention.pdf p.45].")
    assert body["citations"] == []


def test_two_chunks_from_the_same_page_are_one_citation_chip(client, monkeypatch):
    twin = dict(HITS[0], chunk_id=11)  # another chunk of the same document and page
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [dict(HITS[0]), twin])
    reply = {"kind": "policy", "summary": "Answer.", "sections": [{"heading": "What the policy says", "points": ["Twelve weeks [Fraud_Detection_SOP.pdf p.45]."]}]}
    turns = iter([SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(name="search_policies", arguments='{"query": "x"}'))]),
                  SimpleNamespace(content=json.dumps(reply), tool_calls=None)])
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: next(turns))
    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "deadline?"}]}).json()
    assert len(body["citations"]) == 1
