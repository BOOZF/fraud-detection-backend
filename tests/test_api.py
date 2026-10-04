def test_health_reports_teradata_version(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["td_version"][0].isdigit()


def test_kpis_summarise_loaded_transactions(client):
    r = client.get("/api/kpis")
    assert r.status_code == 200
    k = r.json()
    assert k["total_txn"] == 200_000  # rows loaded by scripts/load_data.py
    assert k["fraud_txn"] == 2823
    assert abs(k["fraud_rate"] - 2823 / 200_000) < 1e-9
    channels = {c["channel"]: c for c in k["fraud_by_channel"]}
    assert set(channels) == {"CARD_POS", "CARD_ECOM", "DUITNOW", "FPX", "ATM"}
    assert sum(c["fraud"] for c in channels.values()) == 2823


def test_alerts_are_filtered_and_sorted_by_probability(client):
    r = client.get("/api/alerts", params={"min_prob": 0.5, "limit": 5})
    assert r.status_code == 200
    alerts = r.json()
    assert 0 < len(alerts) <= 5
    probs = [a["prob"] for a in alerts]
    assert probs == sorted(probs, reverse=True)
    assert all(p >= 0.5 for p in probs)
    assert set(alerts[0]) == {"txn_id", "amount_myr", "channel", "merchant_cat", "prob", "txn_ts"}


def test_alert_detail_has_reason_codes_for_new_device(client):
    alerts = client.get("/api/alerts", params={"min_prob": 0.5, "limit": 50}).json()
    details = [client.get(f"/api/alerts/{a['txn_id']}").json() for a in alerts[:10]]
    new_device = [d for d in details if d["txn"]["device_new"] == 1]
    assert new_device, "expected at least one high-probability alert on a new device"
    for d in new_device:
        assert "New device" in d["reasons"]
    for d in details:
        assert 0.5 <= d["prob"] <= 1.0
        assert d["reasons"] and all(isinstance(x, str) for x in d["reasons"])
        assert "txn_scores" in d["sql"]


def test_alert_detail_unknown_id_is_404(client):
    assert client.get("/api/alerts/999999999").status_code == 404


def test_model_card_reports_in_db_training_metrics(client):
    m = client.get("/api/model").json()
    assert 0.7 < m["auc"] <= 1.0
    assert m["train_rows"] + m["test_rows"] == 200_000
    assert m["train_seconds"] > 0
    assert "amount_myr" in m["features"]


def test_rescore_runs_xgboost_predict_in_database(client):
    r = client.post("/api/score")
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == 200_000
    assert body["seconds"] > 0
    assert "XGBoostPredict" in body["sql"]


def test_alerts_default_has_no_probability_threshold_and_is_sorted_descending(client):
    r = client.get("/api/alerts", params={"limit": 1000})
    assert r.status_code == 200
    probs = [a["prob"] for a in r.json()]
    assert len(probs) == 1000  # more than the 255 transactions scoring >= 0.8, and above the old 500 cap
    assert probs == sorted(probs, reverse=True)
    assert min(probs) < 0.8  # the old default threshold is gone


def test_alert_detail_has_a_customer_summary_separate_from_the_transaction(client):
    top = client.get("/api/alerts", params={"limit": 1}).json()[0]
    d = client.get(f"/api/alerts/{top['txn_id']}").json()
    c = d["customer"]
    assert c["customer_id"] == d["txn"]["customer_id"]
    assert c["txn_count"] >= 1
    assert c["flagged_count"] >= 1  # the transaction itself scores >= 0.8
    assert c["avg_amount_myr"] > 0 and c["total_amount_myr"] >= c["avg_amount_myr"]
    assert c["account_age_days"] == d["txn"]["account_age_days"]
    assert c["first_txn_ts"] <= c["last_txn_ts"]
    assert "customer" in d["sql_customer"] or "customer_id" in d["sql_customer"]
