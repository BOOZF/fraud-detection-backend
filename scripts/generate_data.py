"""Synthetic Malaysia XX Bank transactions (see Teradata_Demo_Implementation_Plan.md, section 5)."""
import numpy as np
import pandas as pd

from common import ROOT


def generate(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ch = rng.choice(["CARD_POS", "CARD_ECOM", "DUITNOW", "FPX", "ATM"], n, p=[.35, .25, .2, .12, .08])
    mc = rng.choice(["GROCERY", "FNB", "ELECTRONICS", "TRAVEL", "LUXURY", "CRYPTO", "GAMING"], n,
                    p=[.3, .28, .12, .1, .08, .06, .06])
    df = pd.DataFrame({
        "txn_id": np.arange(1, n + 1), "customer_id": rng.integers(1, 20_000, n),
        "txn_ts": pd.Timestamp("2026-07-01") + pd.to_timedelta(rng.integers(0, 90 * 86400, n), unit="s"),
        "amount_myr": np.round(rng.lognormal(4.2, 1.0, n), 2), "channel": ch, "merchant_cat": mc,
        "is_foreign": (rng.random(n) < .07).astype(int), "device_new": (rng.random(n) < .05).astype(int),
        "hour_of_day": rng.integers(0, 24, n), "km_from_home": np.round(rng.exponential(8, n), 1),
        "txn_count_1h": rng.poisson(1.2, n), "account_age_days": rng.integers(5, 4000, n)})
    df["amt_ratio_30d"] = np.round(df.amount_myr / rng.lognormal(4.2, .5, n), 2)
    # Strongly separable signal so the in-DB model reaches a demo-grade AUC (~1.5% positives)
    z = (-11.6 + 3.6 * df.device_new + 3.2 * df.is_foreign + 2.8 * df.merchant_cat.isin(["CRYPTO", "GAMING"])
         + 1.8 * (df.channel == "CARD_ECOM") + 1.6 * np.log1p(df.amt_ratio_30d) + 0.8 * df.txn_count_1h
         + 2.4 * df.hour_of_day.between(1, 5) + 2.0 * (df.account_age_days < 60) + 0.05 * df.km_from_home)
    df["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


if __name__ == "__main__":
    d = generate()
    print("rows", len(d), "fraud rate", round(d.is_fraud.mean(), 4))
    d.to_parquet(ROOT / "txn.parquet")
