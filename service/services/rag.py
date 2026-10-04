import random
import re
from functools import lru_cache

from openai import OpenAI

from .. import db
from ..config import get_settings

DIM = 384

VECTOR_SQL = """SELECT target_id, reference_id, distance FROM TD_VectorDistance(
 ON (SELECT * FROM q_emb WHERE qid BETWEEN {lo} AND {hi}) AS TargetTable ON policy_emb AS ReferenceTable DIMENSION
 USING TargetIDColumn('qid') TargetFeatureColumns('[1:{dim}]') RefIDColumn('chunk_id')
 RefFeatureColumns('[1:{dim}]') DistanceMeasure('COSINE') TopK({n})) AS dt"""

# Hybrid ranking: the in-database vector search proposes CANDIDATES chunks per question, then a keyword-overlap
# score re-ranks them. Measured on this PDF through the real pipeline (quote one sentence, is its chunk in the top 3?,
# ~270 queries): vector only ~0.87; hybrid with 25 candidates 0.952; hybrid with 60 candidates 0.993. A chunk the
# vector search ranks 30th never gets its keyword boost unless the pool is wide enough.
CANDIDATES = 60
W_SEMANTIC, W_LEXICAL = 0.7, 0.3
STOPWORDS = frozenset(
    "the and for are but not you your all any can had her was one our out has have had how its may who why what when "
    "where which that this with from they will would there their been were into than then them these those such "
    "should must shall also each other some could about after before over under between".split())

_PAGE = re.compile(r"^pp?\.(\d+)")


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS}


def lexical_score(query: str, text: str) -> float:
    """Share of the query's keywords that appear in the chunk (0..1)."""
    wanted = keywords(query)
    return len(wanted & keywords(text)) / len(wanted) if wanted else 0.0


def page_of(section: str) -> int | None:
    """First PDF page of a chunk label such as 'p.12' or 'pp.54-55'; None for non-PDF labels like '3.1'."""
    m = _PAGE.match(section or "")
    return int(m.group(1)) if m else None


def embed(question: str) -> list[float]:
    s = get_settings()
    r = OpenAI(api_key=s.openai_api_key, timeout=30).embeddings.create(
        model=s.embed_model, input=[question], dimensions=DIM)
    return r.data[0].embedding


@lru_cache(maxsize=1)
def _ensure_query_table() -> bool:
    cols = ", ".join(f"e{i} FLOAT" for i in range(DIM))
    try:
        db.execute(f"CREATE TABLE q_emb (qid INTEGER, {cols}) PRIMARY INDEX (qid)")
    except Exception:
        pass  # already exists
    return True


def embed_many(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    r = OpenAI(api_key=s.openai_api_key, timeout=60).embeddings.create(
        model=s.embed_model, input=texts, dimensions=DIM)
    return [d.embedding for d in r.data]


@lru_cache(maxsize=1)
def _chunks() -> dict[int, tuple[str, str, str]]:
    return {int(i): (doc, section, text)
            for i, doc, section, text in db.query("SELECT chunk_id, doc, section, text FROM policy_chunks")}


def clear_chunk_cache() -> None:
    _chunks.cache_clear()


def retrieve_many(questions: list[str], k: int = 4) -> list[list[dict]]:
    """Top-k policy chunks per question by in-database cosine distance (TD_VectorDistance), in one query.
    Every call uses its own block of query ids in q_emb, so overlapping requests do not interfere."""
    if not questions:
        return []
    _ensure_query_table()
    lo = random.randint(1, 2**31 - 1 - len(questions))
    hi = lo + len(questions) - 1
    vectors = embed_many(questions)
    with db.locked():  # insert the query vectors, search, clean up: one unit of Teradata work
        try:
            for qid, vec in zip(range(lo, hi + 1), vectors):
                db.execute(f"INSERT INTO q_emb VALUES ({qid}, {', '.join(f'{float(x):.8f}' for x in vec)})")
            rows = db.query(VECTOR_SQL.format(lo=lo, hi=hi, dim=DIM, n=max(CANDIDATES, k)))
        finally:
            db.execute(f"DELETE FROM q_emb WHERE qid BETWEEN {lo} AND {hi}")
    chunks = _chunks()
    out: list[list[dict]] = [[] for _ in questions]
    for target, cid, dist in sorted(rows, key=lambda r: (r[0], r[2])):
        doc, section, text = chunks[int(cid)]
        q = questions[int(target) - lo]
        score = W_SEMANTIC * (1 - float(dist)) + W_LEXICAL * lexical_score(q, text)
        out[int(target) - lo].append({"doc": doc, "chunk_id": int(cid), "section": section, "page": page_of(section),
                                      "text": text, "distance": float(dist), "score": score})
    return [sorted(hits, key=lambda h: -h["score"])[:k] for hits in out]


def retrieve(question: str, k: int = 4) -> list[dict]:
    return retrieve_many([question], k)[0]
