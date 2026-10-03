import time

import pandas as pd

from teradataml import (ROC, ClassificationEvaluator, copy_to_sql, DataFrame, TrainTestSplit, XGBoost,
                        XGBoostPredict)

from common import connect

FEATS = ["amount_myr", "is_foreign", "device_new", "hour_of_day", "km_from_home", "txn_count_1h",
         "amt_ratio_30d", "account_age_days", "ch_ecom", "ch_duitnow", "ch_fpx", "ch_atm",
         "mc_highrisk", "mc_resale", "night_txn"]

connect()
tdf = DataFrame("txn_features")
split = TrainTestSplit(data=tdf, id_column="txn_id", train_size=0.8, test_size=0.2, seed=42)
train = split.result[split.result.TD_IsTrainRow == 1].drop("TD_IsTrainRow", axis=1)
test = split.result[split.result.TD_IsTrainRow == 0].drop("TD_IsTrainRow", axis=1)

t0 = time.time()
model = XGBoost(data=train, input_columns=FEATS, response_column="is_fraud",
                model_type="Classification", iter_num=50, max_depth=6, lambda1=1, shrinkage_factor=0.3,
                min_node_size=5, base_score=0.0141, num_boosted_trees=8, seed=42)
train_secs = round(time.time() - t0, 1)
print("train secs", train_secs)
model.result.to_sql("fraud_xgb_model", if_exists="replace")

pred = XGBoostPredict(newdata=test, object=model.result, id_column="txn_id",
                      model_type="Classification", output_prob=True,
                      output_responses=["0", "1"], accumulate="is_fraud")
ev = ClassificationEvaluator(data=pred.result, observation_column="is_fraud",
                             prediction_column="Prediction", labels=["0", "1"])
print(ev.output_data)
roc = ROC(data=pred.result, probability_column="Prob_1", observation_column="is_fraud",
          positive_class="1")
print(roc.result)

auc_row = roc.result.to_pandas().iloc[0]
metrics = pd.DataFrame([{
    "auc": float(auc_row["AUC"]), "gini": float(auc_row["GINI"]),
    "train_rows": int(train.shape[0]), "test_rows": int(test.shape[0]),
    "train_seconds": train_secs, "algorithm": "Teradata in-database XGBoost (TD_XGBoost)",
    "features": ",".join(FEATS)}])
copy_to_sql(metrics, "model_metrics", if_exists="replace")

allpred = XGBoostPredict(newdata=DataFrame("txn_features"), object=DataFrame("fraud_xgb_model"),
                         id_column="txn_id", model_type="Classification", output_prob=True,
                         output_responses=["0", "1"])
allpred.result.to_sql("txn_scores", if_exists="replace", primary_index="txn_id")
print(DataFrame("txn_scores").head(3))
