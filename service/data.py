from . import db

TXN_COLS = ["txn_id", "customer_id", "txn_ts", "amount_myr", "channel", "merchant_cat", "is_foreign",
            "device_new", "hour_of_day", "km_from_home", "txn_count_1h", "amt_ratio_30d",
            "account_age_days", "is_fraud"]
_FLOATS = ("amount_myr", "km_from_home", "amt_ratio_30d")
_STRINGS = ("txn_ts", "channel", "merchant_cat")


def get_txn(txn_id: int) -> tuple[dict, float, str] | None:
    sql = (f"SELECT {', '.join('t.' + c for c in TXN_COLS)}, s.Prob_1 "
           f"FROM txn t JOIN txn_scores s ON t.txn_id = s.txn_id WHERE t.txn_id = {int(txn_id)}")
    rows = db.query(sql)
    if not rows:
        return None
    *vals, prob = rows[0]
    txn = dict(zip(TXN_COLS, vals))
    for k in TXN_COLS:
        txn[k] = float(txn[k]) if k in _FLOATS else str(txn[k]) if k in _STRINGS else int(txn[k])
    return txn, float(prob), sql


def get_customer(txn: dict) -> tuple[dict, str]:
    """Behavioural summary of the customer behind a transaction (no personal identifiers beyond the id)."""
    sql = (
        "SELECT COUNT(*) AS n, AVG(t.amount_myr) AS avg_amt, SUM(t.amount_myr) AS total_amt, "
        "MIN(t.txn_ts) AS first_ts, MAX(t.txn_ts) AS last_ts, "
        f"SUM(CASE WHEN s.Prob_1 >= {db.ALERT_THRESHOLD} THEN 1 ELSE 0 END) AS flagged "
        "FROM txn t JOIN txn_scores s ON t.txn_id = s.txn_id "
        f"WHERE t.customer_id = {int(txn['customer_id'])}"
    )
    r = db.query_df(sql).iloc[0]
    return {
        "customer_id": txn["customer_id"],
        "account_age_days": txn["account_age_days"],
        "txn_count": int(r["n"]),
        "flagged_count": int(r["flagged"]),
        "avg_amount_myr": float(r["avg_amt"]),
        "total_amount_myr": float(r["total_amt"]),
        "first_txn_ts": str(r["first_ts"]),
        "last_txn_ts": str(r["last_ts"]),
    }, sql
