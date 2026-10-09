import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
from teradataml import create_context, execute_sql, remove_context

from .config import get_settings

log = logging.getLogger(__name__)

ALERT_THRESHOLD = 0.8
CACHE_TTL_SECONDS = 600

_cache: dict[Any, tuple[float, Any]] = {}
_lock = threading.Lock()

# teradataml shares ONE connection per process and it is not safe for simultaneous use: overlapping requests
# fail with "not a valid connection pool handle". Every Teradata call therefore runs under this reentrant lock.
# Keep slow network calls (OpenAI embeddings / chat) OUTSIDE it so they never block other requests.
_conn_lock = threading.RLock()


def locked():
    """Context manager for a multi-statement unit of Teradata work (reentrant)."""
    return _conn_lock


# Symptoms of a lost or invalidated shared connection (the driver closed its pool, or a reconnect left none).
DEAD_CONNECTION_MARKERS = (
    "not a valid connection pool handle",
    "'NoneType' object has no attribute 'cursor'",
    "TDML_2054",  # teradataml: connection context is empty or not set
)


def _is_dead_connection(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in DEAD_CONNECTION_MARKERS)


def reconnect() -> None:
    """Replace the shared Teradata connection (e.g. after the driver invalidated it)."""
    cfg = get_settings()
    with _conn_lock:
        try:
            remove_context()
        except Exception:
            pass  # the old connection is already dead
        create_context(host=cfg.td_host, username=cfg.td_user, password=cfg.td_password)


def run(unit: Callable[[], Any]) -> Any:
    """Run a unit of Teradata work under the lock. If the shared connection turns out to be dead, reconnect and
    retry the whole unit once. Units must therefore be safe to repeat."""
    with _conn_lock:
        try:
            return unit()
        except Exception as e:
            if not _is_dead_connection(e):
                raise
            log.warning("Teradata connection was invalid (%s); reconnecting and retrying once", str(e)[:160])
            reconnect()
            return unit()


def execute(sql: str, parameters: list[tuple] | None = None) -> None:
    """Run a statement that returns no rows (DDL / INSERT / DELETE)."""
    def unit():
        if parameters is None:
            execute_sql(sql)
        else:
            execute_sql(sql, parameters)

    run(unit)


def query_df(sql: str) -> pd.DataFrame:
    """Run a SELECT against Teradata and return a pandas DataFrame.

    Rows come from teradataml's own execute_sql (the one connection teradataml manages). pandas.read_sql was
    tried and rejected: it leaves transactions open / opens extra pools on that shared connection."""
    def unit():
        cursor = execute_sql(sql)
        return [d[0] for d in cursor.description], cursor.fetchall()

    columns, rows = run(unit)
    return pd.DataFrame([tuple(r) for r in rows], columns=columns)


def query(sql: str) -> list[tuple]:
    return list(query_df(sql).itertuples(index=False, name=None))


def timed_query(sql: str) -> tuple[list[tuple], float]:
    t0 = time.perf_counter()
    rows = query(sql)
    return rows, round(time.perf_counter() - t0, 3)


def cached(key: Any, compute: Callable[[], Any], ttl: float = CACHE_TTL_SECONDS) -> Any:
    """In-memory cache for read-mostly results (scores only change on a re-score)."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = compute()
    with _lock:
        _cache[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def drop_cached(key: Any) -> None:
    with _lock:
        _cache.pop(key, None)
