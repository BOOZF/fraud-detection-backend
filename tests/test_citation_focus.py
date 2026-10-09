"""A citation shows the paragraph it comes from and pinpoints the sentence the answer relies on."""
import json
import re
from types import SimpleNamespace

import pymupdf
import pytest

from service import db, guardrails
from service.services import brief, chat, llm, rag

ORANGE = (1.0, 0.6, 0.0)


def _annots(pdf_bytes: bytes, page: int):
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return [tuple(round(c, 2) for c in (a.colors["stroke"] or ())) for a in pdf[page - 1].annots() if a.type[1] == "Highlight"]


@pytest.fixture(scope="module")
def card_chunk():
    rows = db.query("SELECT chunk_id, doc, section, text FROM policy_chunks WHERE doc LIKE 'credit_card%' AND text LIKE '%shall provide transaction alerts%' ORDER BY chunk_id")
    if not rows:
        pytest.skip("the BNM credit card document is not in the knowledge base")
    chunk_id, doc, section, text = rows[0]
    sentence = next(s for s in re.split(r"(?<=[.!?])\s+", text) if "shall provide transaction alerts" in s and len(s) > 60)
    return {"id": int(chunk_id), "doc": doc, "page": rag.page_of(section), "sentence": sentence}


def test_the_key_sentence_gets_a_stronger_highlight_on_top_of_the_paragraph(client, card_chunk):
    r = client.get(f"/api/documents/{card_chunk['doc']}/file", params={"chunk": card_chunk["id"], "quote": card_chunk["sentence"]})
    colours = _annots(r.content, card_chunk["page"])
    assert any(c == ORANGE for c in colours), "the key sentence has its own colour"
    assert any(c != ORANGE for c in colours), "the paragraph around it is still highlighted"


def test_without_a_quote_only_the_paragraph_is_highlighted(client, card_chunk):
    r = client.get(f"/api/documents/{card_chunk['doc']}/file", params={"chunk": card_chunk["id"]})
    assert not any(c == ORANGE for c in _annots(r.content, card_chunk["page"]))


def test_a_quote_that_is_not_on_the_page_still_gives_the_paragraph(client, card_chunk):
    r = client.get(f"/api/documents/{card_chunk['doc']}/file", params={"chunk": card_chunk["id"], "quote": "this sentence is nowhere in the document at all"})
    colours = _annots(r.content, card_chunk["page"])
    assert r.status_code == 200 and colours and not any(c == ORANGE for c in colours)


# ---- chat: which sentence of the cited chunk does the answer rely on? ----

CHUNK_TEXT = ("Issuers must keep a register of cards. Issuer shall provide transaction alerts to cardholders through prominent channels "
              "such as SMS or in-app notification. Branch opening hours are set by the issuer.")
HIT = {"doc": "a.pdf", "chunk_id": 5, "section": "p.3", "page": 3, "text": CHUNK_TEXT, "distance": 0.3, "score": 0.5}


def chat_with(client, monkeypatch, point):
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [dict(HIT)])
    reply = {"kind": "policy", "summary": "Yes.", "sections": [{"heading": "What the policy says", "points": [point]}]}
    turns = iter([SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(name="search_policies", arguments='{"query": "alerts"}'))]),
                  SimpleNamespace(content=json.dumps(reply), tool_calls=None)])
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: next(turns))
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": "how are cardholders alerted?"}]}).json()


def test_a_chat_citation_points_at_the_sentence_the_answer_relies_on(client, monkeypatch):
    body = chat_with(client, monkeypatch, "Issuers send transaction alerts to cardholders by SMS or in-app notification [a.pdf p.3].")
    assert body["citations"][0]["focus"] == "Issuer shall provide transaction alerts to cardholders through prominent channels such as SMS or in-app notification."


def test_the_focus_is_always_a_sentence_of_the_cited_chunk(client, monkeypatch):
    body = chat_with(client, monkeypatch, "Cardholders are told about activity [a.pdf p.3].")
    focus = body["citations"][0]["focus"]
    assert focus is None or focus in CHUNK_TEXT


def test_no_focus_when_nothing_in_the_chunk_resembles_the_point(client, monkeypatch):
    body = chat_with(client, monkeypatch, "Zebras migrate seasonally [a.pdf p.3].")
    assert body["citations"][0]["focus"] is None


# ---- brief: the evidence sentence is the focus ----

def test_a_brief_citation_focuses_on_the_evidence_sentence():
    chunk = {"doc": "a.pdf", "chunk_id": 5, "section": "p.3", "page": 3, "text": CHUNK_TEXT, "distance": 0.3, "score": 0.5}
    item = {"quote": "Issuer shall provide transaction alerts to cardholders through prominent channels such as SMS or in-app notification."}
    out = brief._covered_item(4, item, chunk, "Via SMS", ["Alerts."])
    assert out["citations"][0]["focus"] == out["evidence"] == item["quote"]


def test_the_key_sentence_is_highlighted_only_where_the_whole_sentence_is_not_in_clauses_that_repeat_its_words(client, card_chunk):
    r = client.get(f"/api/documents/{card_chunk['doc']}/file", params={"chunk": card_chunk["id"], "quote": card_chunk["sentence"]})
    with pymupdf.open(stream=r.content, filetype="pdf") as pdf:
        key = [a.rect for a in pdf[card_chunk["page"] - 1].annots() if a.type[1] == "Highlight" and tuple(round(c, 2) for c in a.colors["stroke"]) == ORANGE]
    assert key
    assert max(k.y1 for k in key) - min(k.y0 for k in key) < 90  # one sentence of a few lines, not three clauses 200pt apart


def test_when_several_chunks_of_one_page_were_retrieved_the_card_shows_the_one_the_answer_is_about(client, monkeypatch):
    unrelated = dict(HIT, chunk_id=11, text="The issuer keeps a register of every card issued. Branch hours are published yearly.")
    relevant = dict(HIT, chunk_id=12, text="Staff are trained annually. Issuer shall provide transaction alerts through SMS or in-app notification.")
    monkeypatch.setattr(guardrails, "topic_of", lambda *a, **k: "on_topic")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=3: [dict(unrelated), dict(relevant)])  # the unrelated chunk comes first
    reply = {"kind": "policy", "summary": "Yes.", "sections": [{"heading": "What the policy says", "points": ["Issuers send transaction alerts through SMS or in-app notification [a.pdf p.3]."]}]}
    turns = iter([SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(name="search_policies", arguments='{"query": "alerts"}'))]),
                  SimpleNamespace(content=json.dumps(reply), tool_calls=None)])
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, **kw: next(turns))
    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "how are cardholders alerted?"}]}).json()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == 12
    assert body["citations"][0]["focus"] == "Issuer shall provide transaction alerts through SMS or in-app notification."
