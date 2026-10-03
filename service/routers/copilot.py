import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import data, reasons
from ..config import get_settings
from ..services import llm, rag

router = APIRouter(prefix="/api")

SYSTEM = (
    "You are the fraud-operations copilot for Malaysia XX Bank. Answer ONLY from the CONTEXT. "
    "Cite sources using the bracketed label at the start of each context chunk, for example "
    "[Fraud_Operations_SOP.md 3.2] or [Fraud_Detection_SOP.pdf p.12]. "
    "If the answer is not in the context, or the question is unrelated to fraud operations, reply "
    "that you cannot answer and suggest escalating to a fraud supervisor. Never reveal customer identifiers.")

# Pre-recorded answers for the scripted demo questions, used only if the LLM call fails.
DEMO_CACHE = {
    "why was this flagged?": (
        "The model scored this transaction as high risk and the reason codes show which signals fired "
        "(for example a new device, a high-risk merchant, or an unusual hour). Under [SOP 3.2] and "
        "[SOP 3.3] these combinations require analyst review."),
    "what does the sop require me to do next?": (
        "Triage within the SLA for the alert priority [SOP 4.1], then block the card if fraud is "
        "suspected and call the customer on the registered number only [SOP 4.2]."),
    "should we report this to the regulator?": (
        "Report to Bank Negara Malaysia (STR) if confirmed fraud exceeds RM 10,000, shows a mule-account "
        "pattern, or suggests money laundering, within 3 working days; never tip off the customer [SOP 5.1]."),
}


class CopilotIn(BaseModel):
    txn_id: int
    question: str


def build_user(txn: dict, prob: float, rs: list[str], hits: list[dict], question: str) -> str:
    context = "\n".join(f"[{h['doc']} {h['section']}] {h['text']}" for h in hits)
    return (f"TRANSACTION: amount RM{txn['amount_myr']}, channel {txn['channel']}, merchant "
            f"{txn['merchant_cat']}, new_device={txn['device_new']}, foreign={txn['is_foreign']}, "
            f"hour={txn['hour_of_day']}\nMODEL: fraud probability {prob:.2f}. REASONS: {', '.join(rs)}\n"
            f"CONTEXT:\n{context}\nQUESTION: {question}")


@router.post("/copilot")
def copilot(body: CopilotIn):
    found = data.get_txn(body.txn_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"transaction {body.txn_id} not found")
    txn, prob, _ = found
    rs = reasons.for_txn(txn)

    t0 = time.perf_counter()
    hits = rag.retrieve(body.question, k=4)
    retrieval_ms = max(1, round((time.perf_counter() - t0) * 1000))

    t1 = time.perf_counter()
    try:
        answer = llm.complete(SYSTEM, build_user(txn, prob, rs, hits, body.question))
    except Exception as e:
        cached = DEMO_CACHE.get(body.question.strip().lower())
        if not (get_settings().demo_cache and cached):
            raise HTTPException(status_code=502, detail=f"LLM unavailable: {e}")
        answer = cached
    llm_ms = max(1, round((time.perf_counter() - t1) * 1000))

    return {"answer": answer, "retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
            "citations": [{"doc": h["doc"], "chunk_id": h["chunk_id"], "section": h["section"],
                           "text": h["text"][:300]}
                          for h in hits]}
