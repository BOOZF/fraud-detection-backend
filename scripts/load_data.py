import pandas as pd
from teradataml import execute_sql, fastload

from common import ROOT, connect

VIEW = """REPLACE VIEW txn_features AS
SELECT txn_id, is_fraud, amount_myr, is_foreign, device_new, hour_of_day, km_from_home,
       txn_count_1h, amt_ratio_30d, account_age_days,
       CASE WHEN channel='CARD_ECOM' THEN 1 ELSE 0 END AS ch_ecom,
       CASE WHEN channel='DUITNOW'   THEN 1 ELSE 0 END AS ch_duitnow,
       CASE WHEN channel='FPX'       THEN 1 ELSE 0 END AS ch_fpx,
       CASE WHEN channel='ATM'       THEN 1 ELSE 0 END AS ch_atm,
       CASE WHEN merchant_cat IN ('CRYPTO','GAMING') THEN 1 ELSE 0 END AS mc_highrisk,
       CASE WHEN merchant_cat IN ('ELECTRONICS','LUXURY') THEN 1 ELSE 0 END AS mc_resale,
       CASE WHEN hour_of_day BETWEEN 1 AND 5 THEN 1 ELSE 0 END AS night_txn
FROM txn"""

if __name__ == "__main__":
    connect()
    df = pd.read_parquet(ROOT / "txn.parquet")
    fastload(df=df, table_name="txn", if_exists="replace", primary_index="txn_id")
    execute_sql(VIEW)
    print(execute_sql("SELECT COUNT(*), SUM(is_fraud) FROM txn_features").fetchall())
