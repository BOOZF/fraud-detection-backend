import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from teradataml import create_context, execute_sql, remove_context

from .config import get_settings
from .routers import alerts, copilot, documents, kpis, ml

log = logging.getLogger(__name__)


def _prewarm() -> None:
    """Load the read-heavy endpoints into the in-memory cache so the first page load is instant."""
    for load in (kpis.kpis, ml.model_card, alerts.alerts):
        try:
            load()
        except Exception:
            log.exception("cache prewarm failed for %s", load.__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    create_context(host=s.td_host, username=s.td_user, password=s.td_password)
    _prewarm()
    try:
        yield
    finally:
        remove_context()


app = FastAPI(title="Malaysia XX Bank Fraud Detection Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(kpis.router)
app.include_router(alerts.router)
app.include_router(ml.router)
app.include_router(copilot.router)
app.include_router(documents.router)


@app.get("/api/health")
def health():
    row = execute_sql("SELECT InfoData FROM DBC.DBCInfoV WHERE InfoKey = 'VERSION'").fetchone()
    return {"ok": True, "td_version": row[0]}
