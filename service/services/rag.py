from functools import lru_cache

from openai import OpenAI
from teradataml import execute_sql

from .. import db
from ..config import get_settings

DIM = 384

VECTOR_SQL = """SELECT reference_id, distance FROM TD_VectorDistance(
 ON q_emb AS TargetTable ON policy_emb AS ReferenceTable DIMENSION
 USING TargetIDColumn('qid') TargetFeatureColumns('[1:{dim}]') RefIDColumn('chunk_id')
 RefFeatureColumns('[1:{dim}]') DistanceMeasure('COSINE') TopK({k})) AS dt ORDER BY distance"""


def embed(question: str) -> list[float]:
    s = get_settings()
    r = OpenAI(api_key=s.openai_api_key, timeout=30).embeddings.create(
        model=s.embed_model, input=[question], dimensions=DIM)
    return r.data[0].embedding


@lru_cache(maxsize=1)
def _ensure_query_table() -> bool:
    cols = ", ".join(f"e{i} FLOAT" for i in range(DIM))
    try:
        execute_sql(f"CREATE TABLE q_emb (qid INTEGER, {cols}) PRIMARY INDEX (qid)")
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


def retrieve(question: str, k: int = 4) -> list[dict]:
    """Top-k policy chunks by in-database cosine distance (TD_VectorDistance).
    q_emb is a shared one-row table, so this is single-user only: fine for a demo."""
    _ensure_query_table()
    vec = ", ".join(f"{float(x):.8f}" for x in embed(question))
    execute_sql("DELETE FROM q_emb")
    execute_sql(f"INSERT INTO q_emb VALUES (1, {vec})")
    chunks = _chunks()
    return [{"doc": chunks[int(cid)][0], "chunk_id": int(cid), "section": chunks[int(cid)][1],
             "text": chunks[int(cid)][2], "distance": float(d)}
            for cid, d in db.query(VECTOR_SQL.format(dim=DIM, k=k))]
