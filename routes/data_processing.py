from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from utils.auth import require_pub_or_basic as _require_auth
from services.application.queue_orchestration_service import QueueOrchestrationService, UniverseFilter


router = APIRouter(prefix="/api/data-processing", tags=["data-processing"])


@router.post("/start")
def start_data_processing(
    request_data: Dict[str, Any] = Body(...),
    _auth: None = Depends(_require_auth),
) -> Dict[str, Any]:
    try:
        svc = QueueOrchestrationService()
        symbols: List[str] = request_data.get("symbols") or []
        if symbols:
            return svc.enqueue_symbol_batch(symbols)
        f = UniverseFilter(
            five_stars=bool(request_data.get("five_stars", False)),
            country=request_data.get("country"),
            instrument_types=(request_data.get("instrument_types") or None),
            exclude_exchanges=(request_data.get("exclude_exchanges") or None),
            limit=int(request_data.get("limit", 0)) or None,
        )
        return svc.start_universe_processing(f)
    except Exception as e:
        raise HTTPException(500, f"Start failed: {e}")


@router.get("/status")
def get_processing_status(_auth: None = Depends(_require_auth)) -> Dict[str, Any]:
    # Minimal status endpoint; can be extended with SB queue stats if needed
    from services.sb_consumer import get_counters_snapshot

    try:
        counters = get_counters_snapshot()
        return {"success": True, "counters": counters}
    except Exception as e:
        raise HTTPException(500, f"Status failed: {e}")


