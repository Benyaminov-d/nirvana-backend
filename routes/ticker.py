from fastapi import APIRouter, HTTPException, Depends, Query  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
import os
import random
import json as _json
import time as _time
from utils.auth import (
    basic_auth_if_configured as _basic_auth_if_configured,
    require_pub_or_basic as _require_pub_or_basic,
)
from services.domain.cvar_unified_service import CvarUnifiedService
from core.db import get_db_session
from core.models import CvarSnapshot, PriceSeries
from functools import partial as _partial
import threading
import concurrent.futures
import logging
from utils.service_bus import (
    sb_connection_string as _sb_conn,
    sb_queue_name as _sb_queue,
)
from sqlalchemy import func, and_  # type: ignore


_logger = logging.getLogger("routes.ticker")
_ticker_lock = threading.Lock()
_submitted_symbols: set[str] = set()
_ticker_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("NVAR_WARM_WORKERS", "4")))
)
# Throttle guard to avoid mass prewarm on frequent page loads
_last_prewarm_ts: float = 0.0


def _build_symbol_pool(n: int) -> list[str]:
    # Build pool from DB when available; otherwise fallback to a minimal set
    sess = get_db_session()
    syms: list[str] = []
    if sess is not None:
        try:
            rows = (
                sess.query(CvarSnapshot.symbol)
                .distinct()
                .all()  # type: ignore
            )
            syms = [s for (s,) in rows]
        except Exception:
            syms = []
        finally:
            try:
                sess.close()
            except Exception:
                pass
    if not syms:
        syms = ["SP500TR", "BTC", "ETH"]
    random.shuffle(syms)
    return syms[:n] if n > 0 else syms


router = APIRouter()


@router.get("/ticker", response_class=HTMLResponse)
def ticker_page(
    _auth: None = Depends(_basic_auth_if_configured),
) -> HTMLResponse:
    from pathlib import Path
    FRONT = Path(__file__).parents[2] / "frontend"
    page = FRONT / "templates" / "ticker.html"
    if page.exists():
        return HTMLResponse(page.read_text(), 200)
    raise HTTPException(404, "ticker.html not found")


@router.get("/ticker/prewarm")
def ticker_prewarm(
    n: int = 30,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    _ = _basic_auth_if_configured
    _basic_auth_if_configured  # type: ignore
    # Cooldown: avoid repeated mass prewarm when n <= 0 (UI bootstrap)
    try:
        cooldown_s = max(0, int(os.getenv("NVAR_PREWARM_COOLDOWN_S", "600")))
    except Exception:
        cooldown_s = 600
    global _last_prewarm_ts
    now_ts = _time.time()
    if n <= 0 and cooldown_s > 0 and (now_ts - _last_prewarm_ts) < cooldown_s:
        return {
            "scheduled": [],
            "total_submitted": len(_submitted_symbols),
            "throttled": True,
        }

    # Build pool
    pool = _build_symbol_pool(n if n > 0 else 0)
    if n > 0:
        pool = pool[:n]
    # Deduplicate and record submitted; cap how many we consider per call
    try:
        cap = max(1, int(os.getenv("NVAR_PREWARM_MAX", "50")))
    except Exception:
        cap = 50
    scheduled: list[str] = []
    for sym in pool:
        with _ticker_lock:
            if sym not in _submitted_symbols:
                _submitted_symbols.add(sym)
                scheduled.append(sym)
        if len(scheduled) >= cap:
            break

    if not scheduled:
        return {"scheduled": [], "total_submitted": len(_submitted_symbols)}

    # Filter by DB freshness/completeness
    # (95 or 99 must have any values on latest date)
    try:
        stale_days = int(os.getenv("NVAR_DB_MAX_AGE_DAYS", "7"))
    except Exception:
        stale_days = 7
    to_enqueue: list[str] = []
    reasons_by_symbol: dict[str, str] = {}
    sess = get_db_session()
    if sess is not None:
        try:
            rows = (
                sess.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.symbol.in_(scheduled)
                )  # type: ignore[arg-type]
                .order_by(
                    CvarSnapshot.symbol.asc(),
                    CvarSnapshot.as_of_date.desc(),
                )
                .all()
            )
            by_symbol: dict[str, list[CvarSnapshot]] = {}
            for r in rows:
                by_symbol.setdefault(r.symbol, []).append(r)
            from datetime import date as _date
            today = _date.today()

            def _row_has_any(rr: CvarSnapshot) -> bool:
                try:
                    v1 = getattr(rr, "cvar_nig", None)
                    v2 = getattr(rr, "cvar_ghst", None)
                    v3 = getattr(rr, "cvar_evar", None)
                    _vals = [v1, v2, v3]
                    floats = [float(v) for v in _vals if v is not None]
                    floats = [v for v in floats if v == v]
                    return bool(floats)
                except Exception:
                    return False

            for sym in scheduled:
                sym_rows = by_symbol.get(sym, [])
                if not sym_rows:
                    to_enqueue.append(sym)
                    reasons_by_symbol[sym] = "no_rows"
                    continue
                latest_as_of = sym_rows[0].as_of_date
                same = [r for r in sym_rows if r.as_of_date == latest_as_of]
                has_any_95 = any(
                    (r.alpha_label == 95) and _row_has_any(r) for r in same
                )
                has_any_99 = any(
                    (r.alpha_label == 99) and _row_has_any(r) for r in same
                )
                is_incomplete = not (has_any_95 or has_any_99)
                try:
                    is_stale = (today - latest_as_of).days > stale_days
                except Exception:
                    is_stale = True
                if is_incomplete or is_stale:
                    to_enqueue.append(sym)
                    reasons_by_symbol[sym] = (
                        "incomplete" if is_incomplete else "stale"
                    )
                else:
                    reasons_by_symbol[sym] = "ok"
        finally:
            try:
                sess.close()
            except Exception:
                pass
    else:
        # No DB: enqueue all scheduled
        to_enqueue = list(scheduled)
        for sym in to_enqueue:
            reasons_by_symbol[sym] = "no_db"

    # Prefer enqueuing batches to Service Bus if configured;
    # fallback to local warm
    conn = _sb_conn()
    q = _sb_queue()
    if conn and q and to_enqueue:
        try:
            # Lazy import to avoid hard dependency and linter errors
            from azure.servicebus import (  # type: ignore
                ServiceBusClient,
                ServiceBusMessage,
            )
            batch_size = max(50, int(os.getenv("SB_BATCH", "100")))
            chunk_size = max(50, int(os.getenv("SB_SYMBOLS_PER_MSG", "100")))
            with ServiceBusClient.from_connection_string(conn) as client:
                sender = client.get_queue_sender(queue_name=q)
                with sender:
                    pending: list[ServiceBusMessage] = []
                    for i in range(0, len(to_enqueue), chunk_size):
                        chunk = to_enqueue[i: i + chunk_size]
                        for sym in chunk:
                            body = {
                                "symbol": sym,
                                "alphas": [0.99, 0.95, 0.50],
                                # For prewarm we avoid force by default;
                                # allow override via env
                                "force": bool(
                                    int(os.getenv("NVAR_PREWARM_FORCE", "0"))
                                ),
                            }
                            cid = f"prewarm-{sym}-{int(_time.time())}"
                            pending.append(
                                ServiceBusMessage(
                                    _json.dumps(body),
                                    correlation_id=cid,
                                )
                            )
                            if len(pending) >= batch_size:
                                sender.send_messages(pending)
                                pending.clear()
                    if pending:
                        sender.send_messages(pending)
            _last_prewarm_ts = now_ts
            return {
                "scheduled": to_enqueue,
                "total_submitted": len(_submitted_symbols),
                "mode": "sb",
            }
        except Exception:
            pass

    # Fallback: local warm per symbol
    for sym in to_enqueue:
        _ticker_executor.submit(
            _partial(
                CvarUnifiedService().get_cvar_data,
                sym,
                force_recalculate=False,
            )
        )
    _last_prewarm_ts = now_ts
    return {
        "scheduled": to_enqueue,
        "total_submitted": len(_submitted_symbols),
        "mode": "local",
    }


@router.get("/ticker/feed")
def ticker_feed(
    n: int = 30,
    exclude: str = "",
    five_stars: bool = False,
    country: str | None = None,
    # TEMPORARY: Disabled auth for frontend debugging
    # _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    # TEMPORARY: Return hardcoded response while debugging DB issues
    symbols_data = [
        {"symbol": "AAPL", "as_of": "2025-09-06", "cvar50": 0.15, "cvar95": 0.25, "cvar99": 0.35},
        {"symbol": "MSFT", "as_of": "2025-09-06", "cvar50": 0.12, "cvar95": 0.22, "cvar99": 0.32},
        {"symbol": "GOOGL", "as_of": "2025-09-06", "cvar50": 0.18, "cvar95": 0.28, "cvar99": 0.38},
        {"symbol": "TSLA", "as_of": "2025-09-06", "cvar50": 0.25, "cvar95": 0.35, "cvar99": 0.45},
        {"symbol": "NVDA", "as_of": "2025-09-06", "cvar50": 0.20, "cvar95": 0.30, "cvar99": 0.40},
    ]
    
    return {
        "items": symbols_data[:n] if n > 0 else symbols_data,
        "nightly_last_run": None,
        "fallback_used": True,
        "requested_country": country,
        "countries_used": ["US"],
        "instrument_types_used": ["Stock"],
        "final_data_source": "hardcoded_temp",
        "debug_info": {
            "out_length": len(symbols_data[:n] if n > 0 else symbols_data),
            "requested_country_debug": country,
            "condition_check": {
                "not_out": False,
                "has_requested_country": bool(country),
                "not_us": country != "US" if country else False
            },
            "has_cvar_data": True
        }
    }

@router.get("/ticker/five_stars_batch")
def five_stars_batch(
    alpha: int = Query(99, description="Alpha label: 50, 95 or 99"),
    country: str | None = Query(
        None,
        description="Restrict to country; default US",
    ),
    limit: int = Query(0, description="0 = all; >0 limits number of items"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "Database not configured")
    try:
        latest_per_symbol = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .filter(CvarSnapshot.alpha_label == alpha)
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        q = (
            sess.query(CvarSnapshot, PriceSeries)
            .join(
                latest_per_symbol,
                and_(
                    CvarSnapshot.symbol == latest_per_symbol.c.symbol,
                    CvarSnapshot.as_of_date == latest_per_symbol.c.mx,
                ),
            )
            .outerjoin(PriceSeries, PriceSeries.symbol == CvarSnapshot.symbol)
            .filter(CvarSnapshot.alpha_label == alpha)
            .filter(PriceSeries.five_stars == 1)  # type: ignore
        )
        if country:
            q = q.filter(
                PriceSeries.country == country  # type: ignore
            )
        else:
            # Default to US
            q = q.filter(
                PriceSeries.country.in_(
                    ["US", "USA", "United States"]
                )  # type: ignore
            )
        rows = q.all()

        def _flt(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        items = []
        for snap, ps in rows:
            vals = [
                _flt(snap.cvar_nig),
                _flt(snap.cvar_ghst),
                _flt(snap.cvar_evar),
            ]
            vals = [v for v in vals if v is not None and v == v and v >= 0]
            worst = max(vals) if vals else None
            items.append(
                {
                    "symbol": snap.symbol,
                    "name": getattr(ps, "name", None),
                    "as_of": (
                        snap.as_of_date.isoformat()
                        if snap.as_of_date
                        else None
                    ),
                    "alpha": int(alpha),
                    "value": worst,
                }
            )
        # Optional limit
        if isinstance(limit, int) and limit > 0:
            items = items[:limit]
        return {"items": items}
    except Exception:
        raise HTTPException(500, "failed to query five-stars batch")
    finally:
        try:
            sess.close()
        except Exception:
            pass
