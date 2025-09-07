from fastapi import APIRouter, Depends
from utils.auth import require_pub_or_basic as _require_pub_or_basic
from services.sb_consumer import (
    reset_counters as _sb_reset,
    get_counters_snapshot as _sb_snapshot,
)


router = APIRouter()


@router.post("/metrics/counters/reset")
def counters_reset(_auth: None = Depends(_require_pub_or_basic)) -> dict:
    """Reset SB incoming and DB upsert success counters to zero."""
    try:
        _sb_reset()
    except Exception:
        pass
    return {"ok": True}


@router.get("/metrics/counters")
def counters_get(_auth: None = Depends(_require_pub_or_basic)) -> dict:
    try:
        return _sb_snapshot()
    except Exception:
        # Fallback to empty counters if module unavailable
        return {
            "sb_incoming": 0,
            "db_upsert_success": 0,
            "sb_success": 0,
            "sb_errors": 0,
            "sb_abandoned": 0,
            "sb_empty": 0,
            "errors": {
                "insufficient_history": 0,
                "insufficient_data": 0,
                "calc_failed": 0,
                "other": 0,
            },
            "insufficient_data_raw": [],
            "recent": [],
        }


