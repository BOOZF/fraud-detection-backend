"""Retrieval benchmark: does the right chunk come back for a natural question?

For a sample of stored chunks an LLM writes the question an analyst would ask that ONLY that chunk answers. Each
embedding setup then has to rank the source chunk among all chunks. Measures retrieval alone (no answer generation).
Run from this folder:  python scripts/eval_retrieval.py [--n 80]"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from common import connect  # noqa: E402
from service import db  # noqa: E402
from service.services import rag  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120)
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 80
QGEN_MODEL = "gpt-5-mini"
CONFIGS = [  # (label, model, dimensions)
    ("3-small @384 (current)", "text-embedding-3-small", 384),
    ("3-small @1536", "text-embedding-3-small", 1536),
    ("3-large @1024", "text-embedding-3-large", 1024),
    ("3-large @3072", "text-embedding-3-large", 3072),
]


def make_question(text: str) -> str:
    r = client.chat.completions.create(
        model=QGEN_MODEL, reasoning_effort="minimal", max_completion_tokens=400,
        messages=[{"role": "user", "content":
                   "Below is one passage from a fraud-procedures document. Write ONE natural question that a fraud analyst "
                   "might type and that this passage answers. Use your own words (do not copy phrases of 4+ words), do not "
                   "mention 'the passage' or 'the document'. Reply with the question only.\n\nPASSAGE:\n" + text[:1500]}])
    return r.choices[0].message.content.strip()


def embed(texts: list[str], model: str, dims: int) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 128):
        r = client.embeddings.create(model=model, input=texts[i:i + 128], dimensions=dims)
        out += [d.embedding for d in r.data]
    m = np.array(out, dtype=np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def metrics(ranks: list[int], pages_ok: list[bool]) -> dict:
    r = np.array(ranks)
    return {"hit@1": float((r <= 1).mean()), "hit@3": float((r <= 3).mean()), "hit@5": float((r <= 5).mean()),
            "hit@10": float((r <= 10).mean()), "mrr": float((1 / r).mean()), "same_page@3": float(np.mean(pages_ok))}


if __name__ == "__main__":
    connect()
    rows = db.query("SELECT chunk_id, doc, section, text FROM policy_chunks ORDER BY chunk_id")
    ids = [int(r[0]) for r in rows]
    docs, sections, texts = [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    rng = random.Random(11)
    sample = sorted(rng.sample(range(len(rows)), min(N, len(rows))))
    with ThreadPoolExecutor(8) as pool:
        questions = list(pool.map(lambda i: make_question(texts[i]), sample))
    print(f"{len(rows)} chunks, {len(sample)} generated questions, e.g.:")
    for q in questions[:4]:
        print("  -", q)

    report = {}
    for label, model, dims in CONFIGS:
        C, Q = embed(texts, model, dims), embed(questions, model, dims)
        sims = Q @ C.T
        for hybrid in (False, True):
            ranks, pages_ok = [], []
            for qi, ci in enumerate(sample):
                s = sims[qi]
                if hybrid:  # the production re-rank: vector candidates, then keyword overlap
                    cand = np.argsort(-s)[:rag.CANDIDATES]
                    score = {int(j): rag.W_SEMANTIC * s[j] + rag.W_LEXICAL * rag.lexical_score(questions[qi], texts[j]) for j in cand}
                    order = sorted(score, key=lambda j: -score[j])
                else:
                    order = list(np.argsort(-s))
                rank = order.index(ci) + 1 if ci in order else 10**6
                ranks.append(rank)
                top3 = order[:3]
                pages_ok.append(any(docs[j] == docs[ci] and rag.page_of(sections[j]) == rag.page_of(sections[ci]) for j in top3))
            name = f"{label} + {'hybrid' if hybrid else 'vector only'}"
            report[name] = metrics(ranks, pages_ok)
            m = report[name]
            print(f"{name:42s} hit@1 {m['hit@1']:.2f}  hit@3 {m['hit@3']:.2f}  hit@5 {m['hit@5']:.2f}  hit@10 {m['hit@10']:.2f}  mrr {m['mrr']:.3f}  same-page@3 {m['same_page@3']:.2f}")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps({"n": len(sample), "results": report}, indent=1))
