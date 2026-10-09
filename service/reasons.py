"""Rule-based reason codes that mirror the model features (the model gives the probability,
the rules give analysts human-readable reasons)."""

HIGH_RISK_MERCHANTS = {"CRYPTO": "crypto", "GAMING": "gaming"}


def for_txn(t: dict) -> list[str]:
    reasons = []
    if t["device_new"]:
        reasons.append("New device")
    if t["is_foreign"]:
        reasons.append("Foreign transaction")
    if t["amt_ratio_30d"] >= 3:
        reasons.append(f"Amount {t['amt_ratio_30d']:.1f}x the 30-day average")
    if t["merchant_cat"] in HIGH_RISK_MERCHANTS:
        reasons.append(f"High-risk merchant ({HIGH_RISK_MERCHANTS[t['merchant_cat']]})")
    if 1 <= t["hour_of_day"] <= 5:
        reasons.append(f"Transaction at {t['hour_of_day']}am")
    if t["txn_count_1h"] >= 4:
        reasons.append(f"Velocity: {t['txn_count_1h']} transactions in the last hour")
    if t["account_age_days"] < 60:
        reasons.append(f"New account ({t['account_age_days']} days old)")
    if t["channel"] == "CARD_ECOM":
        reasons.append("Card-not-present (e-commerce)")
    if t["km_from_home"] >= 50:
        reasons.append(f"{t['km_from_home']:.0f} km from home")
    return reasons or ["No single rule fired; score driven by the combination of features"]
