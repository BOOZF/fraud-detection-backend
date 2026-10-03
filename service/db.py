import threading
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
from teradataml import get_context

ALERT_THRESHOLD = 0.8
CACHE_TTL_SECONDS = 600

_cache: dict[Any, tuple[float, Any]] = {}
_lock = threading.Lock()


def query_df(sql: str) -> pd.DataFrame:
    """Run a SELECT against Teradata and return a pandas DataFrame."""
    return pd.read_sql(sql, get_context())


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
