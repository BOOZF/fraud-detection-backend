import pandas as pd

from service import db


def _spy(monkeypatch):
    calls = []
    real = db.query_df

    def spy(sql: str) -> pd.DataFrame:
        calls.append(sql)
        return real(sql)

    monkeypatch.setattr(db, "query_df", spy)
    return calls


def test_read_endpoints_hit_the_database_once_until_a_rescore(client, monkeypatch):
    db.clear_cache()
    calls = _spy(monkeypatch)

    first = client.get("/api/alerts", params={"limit": 20}).json()
    second = client.get("/api/alerts", params={"limit": 20}).json()
    assert first == second
    assert len(calls) == 1, "second identical request must be served from the in-memory cache"

    client.get("/api/kpis"), client.get("/api/kpis")
    client.get("/api/model"), client.get("/api/model")
    kpi_and_model_queries = len(calls) - 1
    assert kpi_and_model_queries <= 3, "kpis (2 queries) and model (1 query) are each fetched once"

    def alert_queries() -> int:
        return sum("JOIN txn_scores" in sql for sql in calls)

    before = alert_queries()
    assert client.post("/api/score").status_code == 200  # re-scoring changes txn_scores
    client.get("/api/alerts", params={"limit": 20})
    assert alert_queries() == before + 1, "cache must be invalidated by a re-score"


def test_different_alert_queries_are_cached_separately(client):
    db.clear_cache()
    top5 = client.get("/api/alerts", params={"limit": 5}).json()
    top10 = client.get("/api/alerts", params={"limit": 10}).json()
    assert len(top5) == 5 and len(top10) == 10
    assert top10[:5] == top5


def test_prewarm_loads_the_cache_so_the_first_page_load_is_instant(client, monkeypatch):
    from service.app import _prewarm  # called from the app lifespan at startup

    db.clear_cache()
    calls = _spy(monkeypatch)
    _prewarm()
    warmed = len(calls)
    assert warmed >= 3, "prewarm should load KPIs, the model card and the alert list"
    client.get("/api/kpis"), client.get("/api/model"), client.get("/api/alerts")
    assert len(calls) == warmed, "first requests after startup must be served from the cache"
