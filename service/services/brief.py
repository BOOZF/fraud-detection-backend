"""Structured copilot brief for one alert: the questions a fraud analyst must settle.

* Question 1 (why flagged) and the headline come from the rule-based reason codes and the model score: facts, no LLM.
* Questions 2-5 are answered from the uploaded policies. The model must say whether an excerpt really answers the
  question and quote the supporting sentence word for word; the code checks the quote is in the cited chunk. An answer
  without a verified quote is shown as "Not covered", with no citation, rather than a page that does not support it."""
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from .. import data, db, reasons
from ..config import get_settings
from ..services import llm, rag

QUESTIONS = [
    "Why was this transaction flagged?",
    "How urgent is it, and what is the response deadline?",
    "What should the analyst do next?",
    "Does this need a regulator report?",
    "How should the customer be contacted?",
]
FACTS_QUESTION = 0  # answered from reason codes; the rest need a policy
POLICY_QUESTIONS = [1, 2, 3, 4]

# What to search for each question. These are written in the vocabulary of a CARD ISSUER's procedures (cards, issuer,
# cardholder, unauthorised transaction), not of any one document: queries written for an immigration SOP ("referral",
# "request for evidence") ranked that document's chunks above a bank's own card policy and left every question uncovered.
RETRIEVAL = [
    "indicators of fraud: reasons a card transaction is flagged, suspicious indicators and red flags for fraud review",
    "time limit or deadline in days for the card issuer to investigate, respond to or resolve a suspected fraudulent or unauthorised card transaction",
    "what the card issuer must do when a suspicious or unauthorised card transaction is detected: block the transaction, kill switch, hotline, fraud case investigation procedures",
    "must the card issuer report a fraud case or suspicious transaction to Bank Negara Malaysia, the police or another authority",
    "how and when the card issuer must notify or contact the cardholder about a suspicious or blocked card transaction: transaction alerts, SMS, hotline",
]
EXCERPTS_PER_QUESTION = 6

SYSTEM = (
    "You are the fraud-operations copilot for Malaysia XX Bank. Each numbered question is about ONE flagged bank card "
    "payment. Answer ONLY from the CONTEXT of that question (policy excerpts, each starting with [chunk_id]); the "
    "TRANSACTION facts tell you whether a rule applies to this payment. "
    "The excerpts may come from documents written for another organisation or another kind of case (for example "
    "immigration applications or a university). A rule that is about another party or case, such as a deadline for an "
    "applicant, does NOT answer a question about this card payment. Do not stretch it. "
    "Set answerable=true only if an excerpt states what is needed to answer THIS question for THIS payment. Then give "
    "'quote': ONE sentence copied word for word from that excerpt (at least 8 words), and 'sources': its chunk_id. "
    "Never state a deadline, threshold, amount or procedure that is not in your quote. If no excerpt answers the "
    "question, set answerable=false and leave verdict, points and quote empty. "
    "When answerable: a VERDICT of at most 10 words (a direct answer; if it contains a time limit it must say what the limit applies to, for example 'Acknowledge a cardholder dispute within 3 working days') and 2 or 3 POINTS, each one short sentence of at "
    "most 20 words, that only restate what the quoted sentence says, in plain words: do not add steps, comparisons, "
    "deadlines, conditions or advice of your own, and say what a time limit applies to. "
    "Never reveal customer identifiers. Return JSON: "
    '{"items": [{"question": "<the question>", "answerable": true or false, "verdict": "<verdict>", '
    '"points": ["<point>", "<point>"], "quote": "<verbatim sentence>", "sources": [<chunk_id>]}]} '
    "with EXACTLY one item per question, in order: every question gets an item, never skip one."
)

NOT_COVERED = "Not covered by the uploaded policies."
NOT_COVERED_VERDICT = "Not covered by policies"
MAX_VERDICT_WORDS, MAX_POINTS, MAX_POINT_WORDS = 10, 3, 30
MIN_QUOTE_CHARS = 25  # shorter than this and a "quote" proves nothing
ATTEMPTS = 2
PIPELINE = "grounded-v8"  # bump when the way briefs are built changes, so stored briefs from before are not served


def priority_of(prob: float) -> str | None:
    return "P1" if prob >= 0.9 else "P2" if prob >= db.ALERT_THRESHOLD else None


def _words(text: str, limit: int) -> str:
    """One line, at most `limit` words: the layout never has to cope with a runaway answer."""
    words = text.split()
    return " ".join(words[:limit]) + ("..." if len(words) > limit else "")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


VERIFY_SYSTEM = (
    "You check one answer produced by a bank's fraud copilot. The alert is a flagged bank CARD payment. Decide whether "
    "the PASSAGE states what is needed to answer the QUESTION for this card payment, and whether the PROPOSED ANSWER is "
    "what the passage says. Reply applies=false if the passage is about another party, organisation or kind of case "
    "(for example an immigration applicant, a university, a government agency's own staff), if the answer stretches it "
    "by analogy, if it adds steps, deadlines or conditions the passage does not state, or if the passage only shares "
    'words with the question. Also compare WHAT STARTS the duty or deadline in the passage with the situation here: this '
    "alert was raised by the bank's own fraud model, not by a customer report. A duty or time limit that only starts when a "
    "customer lodges a dispute or report (acknowledge it, request information within N days) does NOT answer how urgent a "
    "bank-detected alert is or what the analyst's response deadline is: reply applies=false. A passage that states how the "
    "issuer must alert or notify the cardholder about suspicious or unauthorised transactions (channels, contact details) "
    'DOES answer how to contact the customer. Return JSON: {"applies": true or false, "why": "<one sentence>"}.'
)


def _applies(qi: int, txn: dict, prob: float, rs: list[str], chunk: dict, item: dict, model: str | None) -> bool:
    facts = (f"bank card payment RM{txn['amount_myr']:.2f}, channel {txn['channel']}, merchant {txn['merchant_cat']}, "
             f"fraud probability {prob:.2f}, reason codes: {', '.join(rs)}")
    user = (f"QUESTION: {QUESTIONS[qi]}\nTRANSACTION: {facts}\nPASSAGE ({chunk['doc']} {chunk['section']}):\n{chunk['text']}\n"
            f"QUOTED SENTENCE: {item.get('quote', '')}\nPROPOSED ANSWER: {item.get('verdict', '')}. {' '.join(map(str, item.get('points') or []))}")
    return _is_true(llm.complete_json(VERIFY_SYSTEM, user, model or get_settings().verify_model).get("applies"))


def _ask(system: str, user: str, model: str | None) -> dict:
    """One JSON reply from the model; text that is not valid JSON (a reply that ran on and was cut off) counts as no reply."""
    try:
        return llm.complete_json(system, user, model)
    except ValueError:  # json.JSONDecodeError
        return {}


def _prompt(txn: dict, prob: float, rs: list[str], hits: list[list[dict]], only: list[int]) -> str:
    facts = (f"amount RM{txn['amount_myr']:.2f}, channel {txn['channel']}, merchant {txn['merchant_cat']}, "
             f"new_device={txn['device_new']}, foreign={txn['is_foreign']}, hour={txn['hour_of_day']}, "
             f"amount_vs_30d_avg={txn['amt_ratio_30d']:.1f}x, txns_last_hour={txn['txn_count_1h']}, "
             f"account_age_days={txn['account_age_days']}")
    blocks = []
    for n, i in enumerate(only, start=1):
        ctx = "\n".join(f"[{h['chunk_id']}] ({h['doc']} {h['section']}) {h['text']}" for h in hits[i])
        blocks.append(f"QUESTION {n}: {QUESTIONS[i]}\nCONTEXT {n}:\n{ctx}")
    return (f"TRANSACTION: {facts}\nMODEL: fraud probability {prob:.2f} (priority {priority_of(prob)}). "
            f"REASON CODES: {', '.join(rs)}\n\n" + "\n\n".join(blocks))


def _citation(h: dict, focus: str | None = None) -> dict:
    return {**{k: h[k] for k in ("doc", "chunk_id", "section", "page", "text")}, "focus": focus}


def _verified_chunk(quote: str, retrieved: list[dict]) -> dict | None:
    """The retrieved chunk that really contains the quoted sentence (ignoring case, spacing and punctuation)."""
    q = _norm(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return None
    return next((h for h in retrieved if q in _norm(h["text"])), None)


def _usable(item) -> bool:
    return isinstance(item, dict) and "answerable" in item


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


def _is_true(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _not_covered(qi: int, prob: float) -> dict:
    points = [NOT_COVERED]
    if qi == 1:  # urgency: the one thing the system itself knows is the model's priority band
        points.append(f"Model priority: {priority_of(prob)} ({prob:.0%} fraud probability).")
    return {"question": QUESTIONS[qi], "verdict": NOT_COVERED_VERDICT, "points": points, "answer": " ".join(points),
            "evidence": None, "citations": [], "covered": False}


def _candidate(item: dict, retrieved: list[dict]):
    """(chunk, verdict, points) if the model says an excerpt answers the question AND its quote is really in a retrieved
    chunk; otherwise None."""
    chunk = _verified_chunk(item.get("quote", ""), retrieved) if _is_true(item.get("answerable")) else None
    points = [_words(str(p).strip(), MAX_POINT_WORDS) for p in (item.get("points") or []) if str(p).strip()][:MAX_POINTS]
    if chunk is None or not points:
        return None
    verdict = _words(str(item.get("verdict", "")).strip(), MAX_VERDICT_WORDS) or _words(points[0], 6)
    return chunk, verdict, points


def _covered_item(qi: int, item: dict, chunk: dict, verdict: str, points: list[str]) -> dict:
    return {"question": QUESTIONS[qi], "verdict": verdict, "points": points, "answer": " ".join(points),
            "evidence": str(item["quote"]).strip(), "citations": [_citation(chunk, str(item["quote"]).strip())], "covered": True}


def _reasons_item(prob: float, rs: list[str]) -> dict:
    """Why it was flagged: the rule-based reason codes and the model score, exactly as the alert page shows them."""
    points = [f"{r}." for r in rs[:MAX_POINTS]]
    return {"question": QUESTIONS[FACTS_QUESTION], "verdict": f"{priority_of(prob)} alert: {prob:.0%} fraud probability",
            "points": points, "answer": " ".join(points), "evidence": None, "citations": [], "covered": True}


def _headline(prob: float, rs: list[str]) -> str:
    return _words(f"Review this {priority_of(prob)} alert: " + ", ".join(r.lower() for r in rs[:3]), 25)


def answer_policy_questions(txn: dict, prob: float, rs: list[str], hits: list[list[dict]], model: str | None = None) -> list[dict]:
    """Questions 2-5, answered by the model and checked against the excerpts. Used by the brief and by the benchmark."""
    # Each question is asked on its own, in parallel. Measured: gpt-5-mini answers a question well when it is the only
    # one in the request, and skips or refuses some of them when four are in one request.
    def ask_one(qi: int) -> dict | None:
        for _ in range(ATTEMPTS):
            item = _answers(_ask(SYSTEM, _prompt(txn, prob, rs, hits, [qi]), model), [qi]).get(qi)
            if item is not None:
                return item
        return None

    with ThreadPoolExecutor(max_workers=len(POLICY_QUESTIONS)) as pool:
        chosen = dict(zip(POLICY_QUESTIONS, pool.map(ask_one, POLICY_QUESTIONS)))
    if any(item is None for item in chosen.values()):  # an incomplete brief is an error, never "not covered"
        raise RuntimeError("the model did not answer every policy question")
    candidates = {i: _candidate(chosen[i], hits[i]) for i in POLICY_QUESTIONS}
    # A real quote can still be the wrong answer (a rule for another party or case), so each surviving answer is
    # checked by an independent model call that sees only the question, the transaction and the passage.
    checked = [i for i in POLICY_QUESTIONS if candidates[i]]
    with ThreadPoolExecutor(max_workers=max(1, len(checked))) as pool:
        verdicts = dict(zip(checked, pool.map(lambda i: _applies(i, txn, prob, rs, candidates[i][0], chosen[i], None), checked)))
    return [_covered_item(i, chosen[i], *candidates[i]) if verdicts.get(i) else _not_covered(i, prob) for i in POLICY_QUESTIONS]


def _generate(txn_id: int) -> dict | None:
    found = data.get_txn(txn_id)
    if found is None:
        return None
    txn, prob, _ = found
    rs = reasons.for_txn(txn)

    t0 = time.perf_counter()
    queries = [f"{r} {', '.join(rs)}" if i == 0 else r for i, r in enumerate(RETRIEVAL)]
    hits = rag.retrieve_many(queries, k=EXCERPTS_PER_QUESTION)
    retrieval_ms = max(1, round((time.perf_counter() - t0) * 1000))

    t1 = time.perf_counter()
    policy_items = answer_policy_questions(txn, prob, rs, hits)
    llm_ms = max(1, round((time.perf_counter() - t1) * 1000))

    facts = {k: txn[k] for k in ("amount_myr", "channel", "merchant_cat", "hour_of_day", "txn_ts")}
    return {"txn_id": txn_id, "prob": prob, "priority": priority_of(prob), "headline": _headline(prob, rs),
            "facts": facts, "indicators": rs, "items": [_reasons_item(prob, rs), *policy_items],
            "coverage": {"covered": sum(i["covered"] for i in policy_items), "total": len(policy_items)},
            "retrieval_ms": retrieval_ms, "llm_ms": llm_ms}


# ---------- stored briefs: an alert's brief is generated once and then read back, so it never changes ----------

CACHE_DDL = ("CREATE TABLE copilot_briefs (txn_id INTEGER NOT NULL, kb_key VARCHAR(64), "
             "payload CLOB CHARACTER SET UNICODE, created_at TIMESTAMP(0)) PRIMARY INDEX (txn_id)")


def kb_key() -> str:
    """Fingerprint of everything a brief depends on: the documents, the model and the pipeline version. A stored
    brief is only valid for the combination it was written from."""
    rows = db.query("SELECT doc, chunks, CAST(uploaded_at AS VARCHAR(19)) FROM policy_docs ORDER BY doc")
    return hashlib.sha256(json.dumps([rows, get_settings().llm_model, PIPELINE], default=str).encode()).hexdigest()[:32]


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
