"""Structured copilot brief for one alert: the questions a fraud analyst must settle, answered from the
uploaded policies with citations (document, section, PDF page). One retrieval query and one LLM call."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import time
from functools import lru_cache

from .. import data, db, reasons
from ..services import llm, rag

QUESTIONS = [
    "Why was this transaction flagged?",
    "How urgent is it, and what is the response deadline?",
    "What should the analyst do next?",
    "Does this need a regulator report?",
    "How should the customer be contacted?",
]

# What to search for each question, in the vocabulary a procedures document uses. The display questions above stay
# as they are; these queries only steer the vector + keyword search.
RETRIEVAL = [
    "indicators of fraud: reasons a case is flagged, suspicious indicators and red flags for fraud review",
    "priority and urgency of a fraud case: timeframe, deadline and response time for handling it",
    "next steps and procedure when fraud is suspected: investigation, referral and what the officer does next",
    "when a fraud case must be reported or referred to law enforcement or another agency",
    "how to contact the subject: interview, request for evidence and communication with the person concerned",
]

SYSTEM = (
    "You are the fraud-operations copilot for Malaysia XX Bank. For each numbered question, answer ONLY from the "
    "CONTEXT of that question (policy excerpts, each starting with [chunk_id]) plus the TRANSACTION facts. "
    "Write for a bank fraud analyst who skims: every answer has a VERDICT of at most 8 words (a direct answer such as "
    "'Yes, report to law enforcement' or 'High risk: respond today') and 2 or 3 POINTS, each one short sentence of at "
    "most 20 words, the first point being the reason. No filler, no repetition of the question. "
    "When a policy sets a threshold or condition (for example an amount, a probability or a pattern), state the rule "
    "and say whether THIS transaction meets it, using the TRANSACTION facts: a clear 'No, because ...' or "
    "'Yes, because ...' is a valid answer and counts as covered. "
    "Never claim a condition is met unless the TRANSACTION facts show it: compare every number in the policy with "
    "the actual figures and mention only the conditions that really apply. "
    "Only if the excerpts say nothing relevant to the question, answer exactly: "
    "Not covered by the uploaded policies; escalate to a fraud supervisor. and use no sources. "
    "Never reveal customer identifiers. Return JSON: "
    '{"headline": "recommended action as one imperative sentence of at most 14 words", "items": [{"question": "<the question>", '
    '"verdict": "<verdict>", "points": ["<point>", "<point>"], "sources": [<chunk_id numbers actually used>]}]} '
    'with one item per question, in order.'
)


NOT_COVERED = "Not covered by the uploaded policies; escalate to a fraud supervisor."
MAX_VERDICT_WORDS, MAX_POINTS, MAX_POINT_WORDS, MAX_HEADLINE_WORDS = 10, 3, 30, 25


def _words(text: str, limit: int) -> str:
    """One line, at most `limit` words: the layout never has to cope with a runaway answer."""
    words = text.split()
    return " ".join(words[:limit]) + ("..." if len(words) > limit else "")


SINGLE_ATTEMPTS = 2


def _usable(item) -> bool:
    return isinstance(item, dict) and (bool(item.get("points")) or bool(str(item.get("answer", "")).strip()))


def _answers(result: dict, asked: list[int]) -> dict[int, dict]:
    """The usable items of a reply, keyed by question index. A reply may skip questions (models sometimes answer
    just one), so the caller asks again for whatever is still missing."""
    items = result.get("items")
    if not isinstance(items, list):
        return {}
    by_question = {str(i.get("question", "")).strip(): i for i in items if _usable(i)}
    found = {i: by_question[QUESTIONS[i]] for i in asked if QUESTIONS[i] in by_question}
    if not found and len(items) == len(asked) and all(_usable(i) for i in items):
        found = dict(zip(asked, items))  # same count, question text reworded: trust the order
    return found


def _shape(item: dict) -> tuple[str, list[str]]:
    points = item.get("points")
    if not isinstance(points, list):
        points = [item["answer"]] if str(item.get("answer", "")).strip() else []  # model fell back to the old shape
    points = [_words(str(p).strip(), MAX_POINT_WORDS) for p in points if str(p).strip()][:MAX_POINTS]
    verdict = _words(str(item.get("verdict", "")).strip(), MAX_VERDICT_WORDS)
    if not points:
        return "Not covered by policies", [NOT_COVERED]
    return verdict or _words(points[0], 6), points


def priority_of(prob: float) -> str | None:
    return "P1" if prob >= 0.9 else "P2" if prob >= db.ALERT_THRESHOLD else None


def _prompt(txn: dict, prob: float, rs: list[str], hits: list[list[dict]], only: list[int] | None = None) -> str:
    facts = (f"amount RM{txn['amount_myr']:.2f}, channel {txn['channel']}, merchant {txn['merchant_cat']}, "
             f"new_device={txn['device_new']}, foreign={txn['is_foreign']}, hour={txn['hour_of_day']}, "
             f"amount_vs_30d_avg={txn['amt_ratio_30d']:.1f}x, txns_last_hour={txn['txn_count_1h']}, "
             f"account_age_days={txn['account_age_days']}")
    blocks = []
    wanted = range(len(QUESTIONS)) if only is None else only
    for n, i in enumerate(wanted, start=1):
        q, chunk_hits = QUESTIONS[i], hits[i]
        ctx = "\n".join(f"[{h['chunk_id']}] ({h['doc']} {h['section']}) {h['text']}" for h in chunk_hits)
        blocks.append(f"QUESTION {n}: {q}\nCONTEXT {n}:\n{ctx}")
    return (f"TRANSACTION: {facts}\nMODEL: fraud probability {prob:.2f} (priority {priority_of(prob)}). "
            f"REASON CODES: {', '.join(rs)}\n\n" + "\n\n".join(blocks))


def _citations(source_ids, retrieved: list[dict]) -> list[dict]:
    by_id = {h["chunk_id"]: h for h in retrieved}
    wanted = [by_id[i] for i in dict.fromkeys(source_ids) if i in by_id]
    return [{k: h[k] for k in ("doc", "chunk_id", "section", "page", "text")} for h in wanted]


def _generate(txn_id: int) -> dict | None:
    found = data.get_txn(txn_id)
    if found is None:
        return None
    txn, prob, _ = found
    rs = reasons.for_txn(txn)

    t0 = time.perf_counter()
    queries = [f"{r} {', '.join(rs)}" if i == 0 else r for i, r in enumerate(RETRIEVAL)]
    hits = rag.retrieve_many(queries, k=4)
    retrieval_ms = max(1, round((time.perf_counter() - t0) * 1000))

    t1 = time.perf_counter()
    result = llm.complete_json(SYSTEM, _prompt(txn, prob, rs, hits))  # one call for all five questions
    headline = str(result.get("headline", "")).strip()
    chosen = _answers(result, list(range(len(QUESTIONS))))
    # Models sometimes answer only one question of several (and a retry for "the rest" peels off one more each time),
    # so every question still missing is asked on its own, in parallel: a one-question request cannot come back partial.
    for _ in range(SINGLE_ATTEMPTS):
        missing = [i for i in range(len(QUESTIONS)) if i not in chosen]
        if not missing:
            break
        with ThreadPoolExecutor(max_workers=len(missing)) as pool:
            replies = list(pool.map(lambda i: llm.complete_json(SYSTEM, _prompt(txn, prob, rs, hits, [i])), missing))
        for i, reply in zip(missing, replies):
            headline = headline or str(reply.get("headline", "")).strip()
            chosen.update(_answers(reply, [i]))
    if len(chosen) < len(QUESTIONS):  # an incomplete brief is an error, never "not covered by the policies"
        raise RuntimeError(f"the model did not answer all {len(QUESTIONS)} questions")
    llm_ms = max(1, round((time.perf_counter() - t1) * 1000))

    items = []
    for n, (q, retrieved) in enumerate(zip(QUESTIONS, hits)):
        item = chosen[n]
        verdict, points = _shape(item)
        sources = [s for s in item.get("sources", []) if isinstance(s, int)]
        cites = _citations(sources, retrieved) or _citations([retrieved[0]["chunk_id"]], retrieved)  # fall back to closest chunk
        items.append({"question": q, "verdict": verdict, "points": points, "answer": " ".join(points), "citations": cites})
    facts = {k: txn[k] for k in ("amount_myr", "channel", "merchant_cat", "hour_of_day", "txn_ts")}
    return {"txn_id": txn_id, "prob": prob, "priority": priority_of(prob),
            "headline": _words(headline, MAX_HEADLINE_WORDS) or "Review this alert per the fraud SOP.",
            "facts": facts, "indicators": rs, "items": items, "retrieval_ms": retrieval_ms, "llm_ms": llm_ms}


# ---------- stored briefs: an alert's brief is generated once and then read back, so it never changes ----------

CACHE_DDL = ("CREATE TABLE copilot_briefs (txn_id INTEGER NOT NULL, kb_key VARCHAR(64), "
             "payload CLOB CHARACTER SET UNICODE, created_at TIMESTAMP(0)) PRIMARY INDEX (txn_id)")


def kb_key() -> str:
    """Fingerprint of the knowledge base. A stored brief is only valid for the documents it was written from."""
    rows = db.query("SELECT doc, chunks, CAST(uploaded_at AS VARCHAR(19)) FROM policy_docs ORDER BY doc")
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()[:32]


@lru_cache(maxsize=1)
def _ensure_cache_table() -> bool:
    try:
        db.execute(CACHE_DDL)
    except Exception:
        pass  # already exists
    return True


def build(txn_id: int) -> dict | None:
    _ensure_cache_table()
    key = kb_key()
    rows = db.query(f"SELECT payload FROM copilot_briefs WHERE txn_id = {int(txn_id)} AND kb_key = '{key}'")
    if rows:
        return json.loads(rows[0][0])
    result = _generate(txn_id)
    if result is not None:
        db.execute(f"DELETE FROM copilot_briefs WHERE txn_id = {int(txn_id)}")
        db.execute("INSERT INTO copilot_briefs VALUES (?, ?, ?, CURRENT_TIMESTAMP(0))",
                   [(int(txn_id), key, json.dumps(result))])
    return result
