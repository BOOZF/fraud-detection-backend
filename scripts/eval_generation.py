"""Answer-stage benchmark: are the brief's answers faithful to the excerpts they were given?

Retrieval is run once per (alert, question) with the production code, so every model sees IDENTICAL excerpts. A stronger
judge model then reads the excerpts + transaction facts + the answer and labels it. Run from this folder:
    python scripts/eval_generation.py [--alerts 10] [--models gpt-4o-mini,gpt-5-mini] [--judge gpt-5]"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from common import connect  # noqa: E402
from service import data, db, reasons  # noqa: E402
from service.services import brief, rag  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180)


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


N_ALERTS = int(arg("--alerts", 10))
MODELS = arg("--models", "gpt-4o-mini,gpt-5-mini").split(",")
JUDGE = arg("--judge", "gpt-5")


def generate(model: str, system: str, user: str) -> dict:
    kw = {"reasoning_effort": "low", "max_completion_tokens": 6000} if model.startswith("gpt-5") else {"temperature": 0}
    r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    return json.loads(r.choices[0].message.content)


JUDGE_SYSTEM = (
    "You are a strict auditor of a RAG system for a bank's fraud analysts. You get a QUESTION, the TRANSACTION facts, the "
    "EXCERPTS the system retrieved (the only knowledge it may use), and the ANSWER it produced. The alert is a card payment "
    "flagged by an ML model at a bank. Judge only from the excerpts and facts, never from your own knowledge. Return JSON:\\n"
    '{"answerable": bool, "grounded": bool, "contradicts": bool, "wrongly_asserts": bool, "wrongly_refuses": bool, "reason": str}\\n'
    "- answerable: the excerpts really contain what is needed to answer THIS question for THIS card-fraud alert (not merely "
    "related words, and not a rule for a different kind of case or party, e.g. a deadline for an applicant).\\n"
    "- grounded: every policy/process claim in the answer is stated in the excerpts, and every claim about the transaction "
    "matches the facts (the fraud probability and reason codes are facts).\\n"
    "- contradicts: some claim in the answer conflicts with an excerpt.\\n"
    "- wrongly_asserts: the excerpts do NOT answer the question, yet the answer states policy/process/deadline as if they did, "
    "or presents an excerpt about something else as the answer.\\n"
    "- wrongly_refuses: the excerpts DO answer it, yet the answer says it is not covered.\\n"
    "Answers such as 'Not covered by the uploaded policies.' is a correct answer when the question is not answerable from the excerpts."
)


def judge(question: str, facts: str, excerpts: list[dict], answer: dict) -> dict:
    ex = "\n".join(f"[{h['chunk_id']}] ({h['doc']} {h['section']}) {h['text']}" for h in excerpts)
    text = f"QUESTION: {question}\nTRANSACTION: {facts}\nEXCERPTS:\n{ex}\n\nANSWER verdict: {answer['verdict']}\nANSWER points: {answer['points']}"
    r = client.chat.completions.create(
        model=JUDGE, response_format={"type": "json_object"}, reasoning_effort="low", max_completion_tokens=6000,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": text}])
    return json.loads(r.choices[0].message.content)


if __name__ == "__main__":
    connect()
    ids = [int(r[0]) for r in db.query(f"SELECT TOP {N_ALERTS} txn_id FROM txn_scores ORDER BY Prob_1 DESC")]
    cases = []
    for t in ids:
        txn, prob, _ = data.get_txn(t)
        rs = reasons.for_txn(txn)
        queries = [f"{r} {', '.join(rs)}" if i == 0 else r for i, r in enumerate(brief.RETRIEVAL)]
        hits = rag.retrieve_many(queries, k=4)
        for qi in brief.POLICY_QUESTIONS:
            cases.append({"txn": t, "qi": qi, "question": brief.QUESTIONS[qi], "hits": hits[qi], "txn_data": txn, "prob": prob, "rs": rs, "all_hits": hits})
    print(f"{len(cases)} cases ({N_ALERTS} alerts x {len(brief.POLICY_QUESTIONS)} policy questions), generators {MODELS}, judge {JUDGE}")
    by_alert = {}
    for c in cases:
        by_alert.setdefault(c["txn"], []).append(c)

    def judge_case(c, item):
        t = c["txn_data"]
        facts = (f"amount RM{t['amount_myr']:.2f}, channel {t['channel']}, merchant {t['merchant_cat']}, fraud probability {c['prob']:.2f}, "
                 f"model priority {brief.priority_of(c['prob'])} (P1 = probability 0.90+, P2 = 0.80-0.89, set by the bank's scoring system), "
                 f"reason codes: {', '.join(c['rs'])}")
        return {"verdict": item["verdict"], "points": item["points"], "cited": [x["chunk_id"] for x in item["citations"]],
                "judge": judge(c["question"], facts, c["hits"], item)}

    results = {}
    for model in MODELS:
        def per_alert(txn):
            cs = by_alert[txn]
            items = brief.answer_policy_questions(cs[0]["txn_data"], cs[0]["prob"], cs[0]["rs"], cs[0]["all_hits"], model)  # the FINAL product
            return [judge_case(c, items[brief.POLICY_QUESTIONS.index(c["qi"])]) for c in cs]
        with ThreadPoolExecutor(5) as pool:
            out = [o for batch in pool.map(per_alert, list(by_alert)) for o in batch]
        results[model] = out
        n = len(out)
        j = [o["judge"] for o in out]
        rate = lambda f: sum(1 for x in j if f(x)) / n
        bad = lambda x: x["contradicts"] or x["wrongly_asserts"] or not x["grounded"]
        shown_not_covered = sum(o["verdict"] == brief.NOT_COVERED_VERDICT for o in out) / n
        print(f"\n{model}: {n} answers")
        print(f"  not answerable from the excerpts : {rate(lambda x: not x['answerable']):.0%}")
        print(f"  shown to the analyst as 'Not covered': {shown_not_covered:.0%}")
        print(f"  grounded                          : {rate(lambda x: x['grounded']):.0%}")
        print(f"  contradicts an excerpt            : {rate(lambda x: x['contradicts']):.0%}")
        print(f"  asserts policy it was not given   : {rate(lambda x: x['wrongly_asserts']):.0%}")
        print(f"  wrongly says 'not covered'        : {rate(lambda x: x['wrongly_refuses']):.0%}")
        print(f"  ACCEPTABLE (no problem above)     : {rate(lambda x: not bad(x) and not x['wrongly_refuses']):.0%}")
    (ROOT / "reports").mkdir(exist_ok=True)
    slim = {m: [{"question": cases[i]["question"], "txn": cases[i]["txn"], **o} for i, o in enumerate(out)] for m, out in results.items()}
    (ROOT / "reports" / "generation_eval_final.json").write_text(json.dumps(slim, indent=1))
