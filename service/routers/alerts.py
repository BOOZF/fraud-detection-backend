from fastapi import APIRouter, HTTPException

from .. import data, db, reasons

router = APIRouter(prefix="/api")

MAX_ALERTS = 5000



def _load(min_prob: float, limit: int) -> list[dict]:
    sql = (f"SELECT TOP {limit} t.txn_id, t.amount_myr, t.channel, t.merchant_cat, s.Prob_1 AS prob, t.txn_ts "
           f"FROM txn t JOIN txn_scores s ON t.txn_id = s.txn_id "
           f"WHERE s.Prob_1 >= {float(min_prob)} ORDER BY s.Prob_1 DESC")
    df = db.query_df(sql)
    df["txn_id"] = df["txn_id"].astype(int)
    df["amount_myr"] = df["amount_myr"].astype(float)
    df["prob"] = df["prob"].astype(float)
    df["txn_ts"] = df["txn_ts"].astype(str)
    return df.to_dict("records")


@router.get("/alerts")
def alerts(min_prob: float = 0.0, limit: int = 5000):
    limit = max(1, min(limit, MAX_ALERTS))
    return db.cached(("alerts", float(min_prob), limit), lambda: _load(min_prob, limit))


@router.get("/alerts/{txn_id}")
def alert_detail(txn_id: int):
    found = data.get_txn(txn_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"transaction {txn_id} not found")
    txn, prob, sql = found
    return {"txn": txn, "prob": prob, "reasons": reasons.for_txn(txn), "sql": sql}
