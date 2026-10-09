import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from teradataml import create_context, remove_context

from . import db
from .config import get_settings
from .routers import alerts, chat, copilot, documents, kpis, ml, overview

log = logging.getLogger(__name__)


def _prewarm() -> None:
    """Load the read-heavy endpoints into the in-memory cache so the first page load is instant."""
    for load in (kpis.kpis, ml.model_card, alerts.alerts, overview.overview):
        try:
            load()
        except Exception:
            log.exception("cache prewarm failed for %s", load.__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    with db.locked():
        create_context(host=s.td_host, username=s.td_user, password=s.td_password)
    _prewarm()
    try:
        yield
    finally:
        with db.locked():
            try:
                remove_context()
            except Exception:
                log.warning("Teradata connection was already closed at shutdown")


app = FastAPI(title="Malaysia XX Bank Fraud Detection Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(kpis.router)
app.include_router(overview.router)
app.include_router(alerts.router)
app.include_router(ml.router)
app.include_router(copilot.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    row = db.query_df("SELECT InfoData FROM DBC.DBCInfoV WHERE InfoKey = 'VERSION'").iloc[0]
    return {"ok": True, "td_version": row["InfoData"]}
