"""/api/overview feeds the dashboard's trend chart, period deltas and merchant list. It must add up to /api/kpis."""
import re


def test_daily_rows_add_up_to_the_kpis(client):
    kpis = client.get("/api/kpis").json()
    daily = client.get("/api/overview").json()["daily"]
    assert sum(r["txn"] for r in daily) == kpis["total_txn"]
    assert sum(r["fraud"] for r in daily) == kpis["fraud_txn"]
    assert sum(r["alerts"] for r in daily) == kpis["alerts_open"]
    assert {r["channel"] for r in daily} == {c["channel"] for c in kpis["fraud_by_channel"]}


def test_rows_are_dated_and_ordered(client):
    daily = client.get("/api/overview").json()["daily"]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date"]) for r in daily)
    assert [r["date"] for r in daily] == sorted(r["date"] for r in daily)
    assert all(isinstance(r["amount"], float) and r["fraud"] <= r["txn"] for r in daily)


def test_merchants_add_up_and_come_with_the_sql(client):
    body = client.get("/api/overview").json()
    total = client.get("/api/kpis").json()["total_txn"]
    assert sum(m["txn"] for m in body["merchants"]) == total
    assert {"merchant_cat", "txn", "fraud", "alerts", "amount"} <= set(body["merchants"][0])
    assert body["merchants"] == sorted(body["merchants"], key=lambda m: -m["fraud"])
    assert body["sql"] and all("SELECT" in s for s in body["sql"])
