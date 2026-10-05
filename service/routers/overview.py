"""Everything the dashboard's trend chart, period-over-period deltas and merchant list need, computed in Teradata."""
from fastapi import APIRouter

from .. import db

router = APIRouter(prefix="/api")

DAILY_SQL = (
    "SELECT SUBSTR(CAST(t.txn_ts AS VARCHAR(30)), 1, 10) AS d, t.channel, COUNT(*) AS txn, SUM(t.is_fraud) AS fraud, "
    f"SUM(CASE WHEN s.Prob_1 >= {db.ALERT_THRESHOLD} THEN 1 ELSE 0 END) AS alerts, SUM(t.amount_myr) AS amount "
    "FROM txn t JOIN txn_scores s ON t.txn_id = s.txn_id GROUP BY 1, 2 ORDER BY 1, 2"
)
MERCHANT_SQL = (
    "SELECT t.merchant_cat, COUNT(*) AS txn, SUM(t.is_fraud) AS fraud, "
    f"SUM(CASE WHEN s.Prob_1 >= {db.ALERT_THRESHOLD} THEN 1 ELSE 0 END) AS alerts, SUM(t.amount_myr) AS amount "
    "FROM txn t JOIN txn_scores s ON t.txn_id = s.txn_id GROUP BY 1 ORDER BY 3 DESC"
)


def _compute() -> dict:
    daily = db.query_df(DAILY_SQL)
    merchants = db.query_df(MERCHANT_SQL)
    return {
        "daily": [{"date": str(r.d), "channel": r.channel, "txn": int(r.txn), "fraud": int(r.fraud),
                   "alerts": int(r.alerts), "amount": round(float(r.amount), 2)} for r in daily.itertuples(index=False)],
        "merchants": [{"merchant_cat": r.merchant_cat, "txn": int(r.txn), "fraud": int(r.fraud),
                       "alerts": int(r.alerts), "amount": round(float(r.amount), 2)} for r in merchants.itertuples(index=False)],
        "sql": [DAILY_SQL, MERCHANT_SQL],
    }


@router.get("/overview")
def overview():
    return db.cached("overview", _compute)
