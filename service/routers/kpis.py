from fastapi import APIRouter

from .. import db

router = APIRouter(prefix="/api")

CHANNEL_SQL = "SELECT channel, COUNT(*) AS txn, SUM(is_fraud) AS fraud FROM txn GROUP BY channel ORDER BY 3 DESC"
ALERTS_SQL = f"SELECT COUNT(*) AS n FROM txn_scores WHERE Prob_1 >= {db.ALERT_THRESHOLD}"


def _compute() -> dict:
    channels = db.query_df(CHANNEL_SQL)
    total, fraud = int(channels["txn"].sum()), int(channels["fraud"].sum())
    open_alerts = int(db.query_df(ALERTS_SQL)["n"].iloc[0])
    return {
        "total_txn": total,
        "fraud_txn": fraud,
        "fraud_rate": fraud / total,
        "alerts_open": open_alerts,
        "fraud_by_channel": [
            {"channel": r.channel, "txn": int(r.txn), "fraud": int(r.fraud)}
            for r in channels.itertuples(index=False)
        ],
        "sql": [CHANNEL_SQL, ALERTS_SQL],
    }


@router.get("/kpis")
def kpis():
    return db.cached("kpis", _compute)
