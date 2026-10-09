"""Answer benchmark on two alert sets so a fix is not graded on the alerts it was designed on.
  dev      = the 10 highest-probability alerts (the set the fixes were designed on)
  heldout  = 10 other alerts drawn at random (seed 7) from the remaining open alerts, never looked at while fixing
Uses the production answer path and the same judge as eval_generation.py.
Run:  python scripts/eval_heldout.py --set dev|heldout --tag baseline [--model gpt-5-mini]"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_generation as eg  # noqa: E402
from common import connect  # noqa: E402
from service import data, db, reasons  # noqa: E402
from service.services import brief, rag  # noqa: E402

SET, TAG, MODEL = eg.arg("--set", "dev"), eg.arg("--tag", "run"), eg.arg("--model", "gpt-5-mini")

if __name__ == "__main__":
    connect()
    ranked = [int(r[0]) for r in db.query(f"SELECT txn_id FROM txn_scores WHERE Prob_1 >= {db.ALERT_THRESHOLD} ORDER BY Prob_1 DESC, txn_id")]
    ids = ranked[:10] if SET == "dev" else sorted(random.Random(7).sample(ranked[10:], 10))
    cases = []
    for t in ids:
        txn, prob, _ = data.get_txn(t)
        rs = reasons.for_txn(txn)
        queries = [f"{r} {', '.join(rs)}" if i == 0 else r for i, r in enumerate(brief.RETRIEVAL)]
        hits = rag.retrieve_many(queries, k=brief.EXCERPTS_PER_QUESTION)
        cases.append((t, txn, prob, rs, hits))

    def run(c):
        t, txn, prob, rs, hits = c
        items = brief.answer_policy_questions(txn, prob, rs, hits, MODEL)
        facts = (f"amount RM{txn['amount_myr']:.2f}, channel {txn['channel']}, merchant {txn['merchant_cat']}, fraud probability {prob:.2f}, "
                 f"model priority {brief.priority_of(prob)} (P1 = probability 0.90+, P2 = 0.80-0.89, set by the bank's scoring system), "
                 f"reason codes: {', '.join(rs)}")
        return [{"txn": t, "channel": txn["channel"], "question": brief.QUESTIONS[qi], "verdict": it["verdict"], "points": it["points"],
                 "judge": eg.judge(brief.QUESTIONS[qi], facts, hits[qi], it)}
                for qi, it in zip(brief.POLICY_QUESTIONS, items)]

    with ThreadPoolExecutor(5) as pool:
        out = [o for batch in pool.map(run, cases) for o in batch]
    j = [o["judge"] for o in out]
    bad = lambda x: x["contradicts"] or x["wrongly_asserts"] or not x["grounded"]
    n = len(out)
    ans = [o for o in out if o["verdict"] != brief.NOT_COVERED_VERDICT]
    summary = {"set": SET, "tag": TAG, "n": n, "acceptable": sum(1 for x in j if not bad(x) and not x["wrongly_refuses"]),
               "answerable": sum(x["answerable"] for x in j), "answered": len(ans),
               "answered_bad": sum(1 for o in ans if bad(o["judge"])), "wrongly_refused": sum(x["wrongly_refuses"] for x in j),
               "answerable_and_answered_ok": sum(1 for o in ans if o["judge"]["answerable"] and not bad(o["judge"]))}
    print(json.dumps(summary))
    (ROOT / "reports" / f"heldout_{TAG}_{SET}.json").write_text(json.dumps({"summary": summary, "cases": out}, indent=1))
