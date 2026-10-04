"""Structured copilot brief for one alert: the questions a fraud analyst must settle, answered from the
uploaded policies with citations (document, section, PDF page). One retrieval query and one LLM call."""
import time

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
    "Be concrete and brief: at most 60 words per answer, plain sentences. "
    "When a policy sets a threshold or condition (for example an amount, a probability or a pattern), state the rule "
    "and say whether THIS transaction meets it, using the TRANSACTION facts: a clear 'No, because ...' or "
    "'Yes, because ...' is a valid answer and counts as covered. "
    "Never claim a condition is met unless the TRANSACTION facts show it: compare every number in the policy with "
    "the actual figures and mention only the conditions that really apply. "
    "Only if the excerpts say nothing relevant to the question, answer exactly: "
    "Not covered by the uploaded policies; escalate to a fraud supervisor. and use no sources. "
    "Never reveal customer identifiers. Return JSON: "
    '{"headline": "one-sentence recommended action", "items": [{"question": "<the question>", '
    '"answer": "<answer>", "sources": [<chunk_id numbers actually used>]}]} with one item per question, in order.'
)


def priority_of(prob: float) -> str | None:
    return "P1" if prob >= 0.9 else "P2" if prob >= db.ALERT_THRESHOLD else None


def _prompt(txn: dict, prob: float, rs: list[str], hits: list[list[dict]]) -> str:
    facts = (f"amount RM{txn['amount_myr']:.2f}, channel {txn['channel']}, merchant {txn['merchant_cat']}, "
             f"new_device={txn['device_new']}, foreign={txn['is_foreign']}, hour={txn['hour_of_day']}, "
             f"amount_vs_30d_avg={txn['amt_ratio_30d']:.1f}x, txns_last_hour={txn['txn_count_1h']}, "
             f"account_age_days={txn['account_age_days']}")
    blocks = []
    for n, (q, chunk_hits) in enumerate(zip(QUESTIONS, hits), start=1):
        ctx = "\n".join(f"[{h['chunk_id']}] ({h['doc']} {h['section']}) {h['text']}" for h in chunk_hits)
        blocks.append(f"QUESTION {n}: {q}\nCONTEXT {n}:\n{ctx}")
    return (f"TRANSACTION: {facts}\nMODEL: fraud probability {prob:.2f} (priority {priority_of(prob)}). "
            f"REASON CODES: {', '.join(rs)}\n\n" + "\n\n".join(blocks))


def _citations(source_ids, retrieved: list[dict]) -> list[dict]:
    by_id = {h["chunk_id"]: h for h in retrieved}
    wanted = [by_id[i] for i in dict.fromkeys(source_ids) if i in by_id]
    return [{k: h[k] for k in ("doc", "chunk_id", "section", "page", "text")} for h in wanted]


def build(txn_id: int) -> dict | None:
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
    result = llm.complete_json(SYSTEM, _prompt(txn, prob, rs, hits))
    llm_ms = max(1, round((time.perf_counter() - t1) * 1000))

    answers = {str(i.get("question", "")).strip(): i for i in result.get("items", []) if isinstance(i, dict)}
    ordered = list(result.get("items", []))
    items = []
    for n, (q, retrieved) in enumerate(zip(QUESTIONS, hits)):
        item = answers.get(q) or (ordered[n] if n < len(ordered) and isinstance(ordered[n], dict) else {})
        answer = str(item.get("answer", "")).strip() or "Not covered by the uploaded policies; escalate to a fraud supervisor."
        sources = [s for s in item.get("sources", []) if isinstance(s, int)]
        cites = _citations(sources, retrieved) or _citations([retrieved[0]["chunk_id"]], retrieved)  # fall back to closest chunk
        items.append({"question": q, "answer": answer, "citations": cites})
    return {"txn_id": txn_id, "prob": prob, "priority": priority_of(prob),
            "headline": str(result.get("headline", "")).strip() or "Review this alert per the fraud SOP.",
            "items": items, "retrieval_ms": retrieval_ms, "llm_ms": llm_ms}
