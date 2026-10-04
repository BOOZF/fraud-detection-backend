from fastapi import APIRouter, HTTPException

from .. import db
from ..services import brief

router = APIRouter(prefix="/api")


@router.post("/alerts/{txn_id}/brief")
def alert_brief(txn_id: int):
    """The copilot's structured answers to the standard questions for one alert, with citations (cached)."""
    def compute():
        try:
            return brief.build(txn_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Copilot unavailable: {e}")

    result = db.cached(("brief", txn_id), compute)
    if result is None:
        db.drop_cached(("brief", txn_id))
        raise HTTPException(status_code=404, detail=f"transaction {txn_id} not found")
    return result
