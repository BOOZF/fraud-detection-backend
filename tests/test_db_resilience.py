import pytest

from service import db

DEAD = "[Teradata SQL Driver] [Error 299] 1 is not a valid connection pool handle"


@pytest.fixture
def reconnects(monkeypatch, client):
    """Stand-in for db.reconnect that only counts calls (a real reconnect takes ~20 s)."""
    calls = []
    monkeypatch.setattr(db, "reconnect", lambda: calls.append(1))
    return calls


def _fail_first_time(monkeypatch, message):
    real = db.execute_sql
    state = {"calls": 0}

    def flaky(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError(message)
        return real(*args, **kwargs)

    monkeypatch.setattr(db, "execute_sql", flaky)
    return state


@pytest.mark.parametrize("message", [
    DEAD,
    "'NoneType' object has no attribute 'cursor'",  # connection missing: an earlier reconnect failed part-way
    "[Teradata][teradataml](TDML_2054) Current Teradata Vantage Connection context is empty or not set.",
])
def test_a_dead_connection_is_replaced_and_the_query_retried(monkeypatch, reconnects, message):
    state = _fail_first_time(monkeypatch, message)
    df = db.query_df("SELECT 7 AS x")
    assert int(df["x"].iloc[0]) == 7
    assert len(reconnects) == 1 and state["calls"] == 2


def test_statements_without_rows_are_also_retried(monkeypatch, reconnects):
    state = _fail_first_time(monkeypatch, DEAD)
    db.execute("SELECT 1 AS x")  # any statement is fine for the retry behaviour
    assert len(reconnects) == 1 and state["calls"] == 2


def test_other_errors_are_not_retried_or_hidden(monkeypatch, reconnects):
    _fail_first_time(monkeypatch, "Syntax error, expected something like a name")
    with pytest.raises(RuntimeError, match="Syntax error"):
        db.query_df("SELEKT 1")
    assert reconnects == []


def test_a_connection_that_stays_dead_raises_after_one_retry(monkeypatch, reconnects):
    def always_dead(*args, **kwargs):
        raise RuntimeError(DEAD)

    monkeypatch.setattr(db, "execute_sql", always_dead)
    with pytest.raises(RuntimeError, match="connection pool handle"):
        db.query_df("SELECT 1 AS x")
    assert len(reconnects) == 1


def test_run_retries_a_whole_unit_of_work_once(monkeypatch, reconnects):
    attempts = []

    def unit():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError(DEAD)
        return "done"

    assert db.run(unit) == "done"
    assert len(attempts) == 2 and len(reconnects) == 1


def test_the_service_keeps_working_after_a_real_reconnect(client):
    db.reconnect()  # the real thing: drops the connection and logs on again
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/kpis").status_code == 200
