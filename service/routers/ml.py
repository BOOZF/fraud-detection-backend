import time

from fastapi import APIRouter
from teradataml import DataFrame, XGBoostPredict

from .. import db

router = APIRouter(prefix="/api")

METRICS_SQL = ("SELECT auc, gini, train_rows, test_rows, train_seconds, algorithm, features "
               "FROM model_metrics")


def _model_card() -> dict:
    r = db.query_df(METRICS_SQL).iloc[0]
    return {"auc": float(r["auc"]), "gini": float(r["gini"]), "train_rows": int(r["train_rows"]),
            "test_rows": int(r["test_rows"]), "train_seconds": float(r["train_seconds"]),
            "algorithm": r["algorithm"], "features": r["features"].split(","), "sql": METRICS_SQL}


@router.get("/model")
def model_card():
    return db.cached("model", _model_card)


def _rescore() -> dict:
    t0 = time.perf_counter()
    pred = XGBoostPredict(newdata=DataFrame("txn_features"), object=DataFrame("fraud_xgb_model"),
                          id_column="txn_id", model_type="Classification", output_prob=True,
                          output_responses=["0", "1"])
    pred.result.to_sql("txn_scores", if_exists="replace", primary_index="txn_id")
    seconds = round(time.perf_counter() - t0, 2)
    db.clear_cache()  # scores changed: drop cached KPIs and alert lists
    rows = db.query("SELECT COUNT(*) FROM txn_scores")[0][0]
    sql = "-- XGBoostPredict over txn_features, persisted to txn_scores\n" + pred.result.show_query()
    return {"rows": int(rows), "seconds": seconds, "sql": sql}


@router.post("/score")
def rescore():
    return db.run(_rescore)  # holds the Teradata lock; reconnects and retries once if the connection died
