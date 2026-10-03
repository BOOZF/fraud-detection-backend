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
