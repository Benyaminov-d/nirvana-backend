from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import re as _re
import requests
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore
from starlette.requests import Request  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from utils.auth import require_pub_or_basic as _require_pub_or_basic
from utils.common import resolve_eodhd_endpoint_symbol
from core.db import get_db_session
from core.models import PriceSeries, CvarSnapshot
from sqlalchemy.sql import func  # type: ignore
from sqlalchemy import or_  # type: ignore

from services.domain.cvar_unified_service import CvarUnifiedService
from services.compass_recommendations_service import (
    CompassRecommendationsService,
)
from services.assistant import (
    AssistantAction,
    AssistantIntent,
    call_llm_intent,
    fallback_intent,
    detect_locale,
    get_thread_dialog,
)


router = APIRouter(prefix="/demo", tags=["demo"])

_LOG = logging.getLogger("demo")
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s demo %(levelname)s: %(message)s")
    )
    _LOG.addHandler(_h)
_LOG.setLevel(logging.INFO)
_LOG.propagate = False


def _ip(request: Request) -> str:
    try:
        return (request.client.host if request.client else "?") or "?"
    except Exception:
        return "?"


def _sid(request: Request) -> str:
    """
    Session identifier used for light UI context only.
    Priority: header -> cookie -> IP.
    """
    try:
        sid = (
            request.headers.get("x-session-id")
            or request.cookies.get("nir_sid")
            or _ip(request)
        )
        return sid or "default"
    except Exception:
        return _ip(request)


# Simple in-memory hourly limiter with increased limits for demo
_RL: Dict[str, List[float]] = {}


# Clear rate limit cache (for immediate effect after limit increases)
def _clear_rate_limits():
    global _RL
    _RL.clear()


# Minimal ephemeral context per session (UI hints like last_candidates/mode)
_CTX: Dict[str, Dict[str, Any]] = {}


def _rate_limit_ok(sid: str, limit_per_hour: int = 60) -> bool:
    now = time.time()
    cutoff = now - 3600.0
    arr = _RL.get(sid) or []
    arr = [t for t in arr if t >= cutoff]
    if len(arr) >= max(1, int(limit_per_hour)):
        _RL[sid] = arr
        return False
    arr.append(now)
    _RL[sid] = arr
    return True


def _type_normalize(raw: Optional[str]) -> str:
    s = (raw or "").strip().lower()
    if "etf" in s:
        return "etf"
    if "fund" in s:
        return "mutual_fund"
    if s in ("equity", "stock", "common stock", "cs"):
        return "equity"
    return "equity" if s else "equity"


def _rank_match(q: str, symbol: str, name: str) -> Tuple[int, int, int, str]:
    """Rank by semantic closeness to query.

    Priority (lower is better):
    0: exact symbol match or ticker in parentheses equals query
    1: exact name match
    2: name/symbol startswith query
    3: substring anywhere
    9: fallback
    """
    ql = (q or "").strip().lower()
    sy = (symbol or "").strip().lower()
    nm = (name or symbol or "").strip().lower()

    paren_ticker = ""
    try:
        if ")" in nm and "(" in nm:
            start = nm.rfind("(")
            end = nm.rfind(")")
            if 0 <= start < end:
                paren_ticker = nm[start + 1: end].strip()
    except Exception:
        paren_ticker = ""

    rank = 9
    if ql and (ql == sy or ql == paren_ticker.lower()):
        rank = 0
    elif ql and ql == nm:
        rank = 1
    elif ql and (sy.startswith(ql) or nm.startswith(ql)):
        rank = 2
    elif ql and (ql in sy or ql in nm):
        rank = 3

    length_penalty = len(nm or sy)

    try:
        tokens = [
            t
            for t in ql.replace("(", " ").replace(")", " ").split()
            if t
        ]
        match_tokens = 0
        for t in tokens:
            if t and (t in nm or t in sy):
                match_tokens += 1
    except Exception:
        match_tokens = 0

    return (rank, length_penalty, -match_tokens, sy)


class RecommendationRequest(BaseModel):
    loss_tolerance_pct: float = Field(
        ..., description="Negative percent, e.g., -25"
    )
    seed_symbol: Optional[str] = Field(
        None, description="Hint symbol user started with"
    )
    country: Optional[str] = Field(
        None, description="Country code to filter products (e.g., US, UK, CA)"
    )


class AssistantRequest(BaseModel):
    message: str = Field(
        ..., description="User message for Satya"
    )
    thread_id: Optional[str] = Field(
        None,
        description=(
            "OpenAI Assistants API thread id (persist dialog on OpenAI)"
        ),
    )
    country: Optional[str] = Field(
        None,
        description="Country code to filter products (e.g., US, UK, CA)"
    )


class AssistantResponse(BaseModel):
    assistant_message: str
    candidates: Optional[list[dict]] = None
    right_pane: Optional[dict] = None
    # New fields to let the client render the conversation and keep the thread:
    thread_id: Optional[str] = None
    dialog: Optional[list[dict]] = None  # [{role, text, created_at}]
    summary_symbol: Optional[str] = None


class MarketQuote(BaseModel):
    symbol: str
    name: str
    current_price: float
    change: float
    change_percent: float
    after_hours_price: Optional[float] = None
    after_hours_change: Optional[float] = None
    after_hours_change_percent: Optional[float] = None
    open_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    last_updated: Optional[str] = None


class HistoricalDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: Optional[float] = None
    volume: Optional[int] = None
    price: float  # For charts - uses adjusted_close when available, else close


class HistoricalDataResponse(BaseModel):
    symbol: str
    period: str
    data: List[HistoricalDataPoint]


@router.get("/assistant/thread/{thread_id}")
def assistant_thread(thread_id: str, request: Request) -> dict:
    """Return dialog list for an existing OpenAI Assistants thread id.
    Safe to call even if Assistants is not configured (returns empty list)."""
    sid = _sid(request)
    if not _rate_limit_ok(sid):
        raise HTTPException(429, "rate limit exceeded")
    th = (thread_id or "").strip()
    if not th:
        return {"thread_id": None, "dialog": []}
    dialog = get_thread_dialog(th, limit=30)
    return {"thread_id": th, "dialog": dialog}


@router.get("/search")
def demo_search(
    request: Request,
    q: str = Query(..., description="Ticker or product name"),
    limit: int = Query(10, ge=1, le=50),
    country: str | None = Query(None, description="Filter by country"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    t0 = time.time()
    sid = _sid(request)
    if not _rate_limit_ok(sid):
        raise HTTPException(429, "rate limit exceeded")
    query = (q or "").strip()
    if not query:
        return {"items": []}
    sess = get_db_session()
    if sess is None:
        return {"items": []}
    try:
        needle = f"%{query.lower()}%"
        base = sess.query(PriceSeries)
        filt = or_(
            func.lower(PriceSeries.symbol).like(needle),  # type: ignore
            func.lower(PriceSeries.name).like(needle),  # type: ignore
        )
        
        # Filter by valid=1 products only
        base = base.filter(PriceSeries.valid == 1)
        
        # Filter by country if specified
        if country:
            base = base.filter(PriceSeries.country == country)
        
        recs = base.filter(filt).limit(max(50, limit)).all()
        recs_sorted = sorted(
            recs,
            key=lambda r: _rank_match(
                query, r.symbol or "", r.name or r.symbol or ""
            ),
        )[: int(max(1, limit))]
        items = [
            {
                "symbol": r.symbol,
                "name": (r.name or r.symbol),
                "type": _type_normalize(r.instrument_type),
                "country": r.country or "US",
            }
            for r in recs_sorted
        ]
        dt = int((time.time() - t0) * 1000)
        try:
            _LOG.info(
                "%s",
                {
                    "event": "search_submit",
                    "q": query,
                    "count": len(items),
                    "latency_ms": dt,
                    "sid": sid,
                },
            )
        except Exception:
            pass
        return {"items": items}
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/instrument/{symbol}/summary")
def instrument_summary(
    request: Request,
    symbol: str,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    t0 = time.time()
    sid = _sid(request)
    if not _rate_limit_ok(sid):
        raise HTTPException(429, "rate limit exceeded")
    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        ps = (
            sess.query(PriceSeries)
            .filter(PriceSeries.symbol == sym)  # type: ignore
            .order_by((PriceSeries.country == "US").desc())  # type: ignore
            .first()
        )
        if ps is None:
            raise HTTPException(404, "symbol not found")
        name = ps.name or sym
        type_norm = _type_normalize(ps.instrument_type)
        country = ps.country or "US"

        # Enforce strictly local CVaR computation for this endpoint
        # Try to read latest snapshots from DB first; if none, compute local
        def _worst_from_row(row: Any) -> float | None:
            try:
                xs = [row.cvar_nig, row.cvar_ghst, row.cvar_evar]
                xs = [float(v) for v in xs if v is not None]
                xs = [v for v in xs if v == v]
                return max(xs) if xs else None
            except Exception:
                return None

        def _latest(alpha: int) -> Any | None:
            try:
                return (
                    sess.query(CvarSnapshot)
                    .filter(CvarSnapshot.symbol == sym)  # type: ignore
                    .filter(CvarSnapshot.alpha_label == alpha)
                    .order_by(CvarSnapshot.as_of_date.desc())
                    .first()
                )
            except Exception:
                return None

        row50 = _latest(50)
        row95 = _latest(95)
        row99 = _latest(99)
        w50_db = _worst_from_row(row50) if row50 else None
        w95_db = _worst_from_row(row95) if row95 else None
        w99_db = _worst_from_row(row99) if row99 else None

        # Initialize data variable for later use
        data = {}
        
        if all(v is None for v in (w50_db, w95_db, w99_db)):
            svc = CvarUnifiedService(mode="local")
            data = svc.get_cvar_data(
                sym, force_recalculate=False, prefer_local=True
            )
            w50 = svc.get_worst_annual(data, "cvar50")
            w95 = svc.get_worst_annual(data, "cvar95")
            w99 = svc.get_worst_annual(data, "cvar99")
        else:
            w50 = w50_db
            w95 = w95_db
            w99 = w99_db
            # Mark as successful since we have DB data
            data = {"success": True}

        # Insufficient history detection
        def _blk_has_any(payload: dict, key: str) -> bool:
            try:
                blk = payload.get(key) or {}
                ann = blk.get("annual") or {}
                vals = [ann.get("nig"), ann.get("ghst"), ann.get("evar")]
                xs = [float(v) for v in vals if v is not None]
                xs = [v for v in xs if v == v]
                return bool(xs)
            except Exception:
                return False

        insufficient_history = False
        try:
            insufficient_history = bool(
                getattr(ps, "insufficient_history", 0) == 1
            )
        except Exception:
            insufficient_history = False
        # Check if we have CVaR data in database
        has_cvar_data = False
        if not insufficient_history:
            try:
                if bool(data.get("success", True)):
                    has_cvar_data = (
                        _blk_has_any(data, "cvar50")
                        or _blk_has_any(data, "cvar95")
                        or _blk_has_any(data, "cvar99")
                    )
                    # If no CVaR data but symbol is valid, try local calculation
                    # Only mark insufficient_history if symbol is actually invalid
                    if not has_cvar_data:
                        # Check if symbol is valid - if valid=1, we should calculate locally
                        symbol_valid = getattr(ps, "valid", None)
                        if symbol_valid != 1:
                            insufficient_history = True
            except Exception:
                insufficient_history = False

        def _to_pct_neg(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            try:
                return round(-100.0 * float(x), 1)
            except Exception:
                return None

        try:
            passes_standard = bool(getattr(ps, "five_stars", 0) == 1)
        except Exception:
            passes_standard = False

        # Add helpful message for insufficient history
        insufficient_message = None
        if insufficient_history:
            insufficient_message = "Data not available - insufficient price history for reliable risk calculation"

        out = {
            "symbol": sym,
            "name": name,
            "type": type_norm,
            "country": country,
            "insufficient_history": bool(insufficient_history),
            "nirvana_standard_pass": passes_standard,
            "loss_levels": {
                "down_year": {
                    "label": "~1-in-5",
                    "cvar_pct": _to_pct_neg(w50),
                },
                "one_in_20": {
                    "label": "1-in-20",
                    "cvar95_pct": _to_pct_neg(w95),
                },
                "one_in_100": {
                    "label": "1-in-100",
                    "cvar99_pct": _to_pct_neg(w99),
                },
                "message": insufficient_message,
            },
        }
        dt = int((time.time() - t0) * 1000)
        try:
            _LOG.info(
                "%s",
                {
                    "event": "instrument_view",
                    "symbol": sym,
                    "latency_ms": dt,
                    "sid": sid,
                },
            )
        except Exception:
            pass
        return out
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _compute_summary_payload(symbol: str) -> dict:
    request = type(
        "_R",
        (),
        {"client": type("_C", (), {"host": "127.0.0.1"})()},
        # dummy Request-like
    )()
    return instrument_summary(request, symbol)  # type: ignore[arg-type]


@router.get("/instrument/{symbol}/quote", response_model=MarketQuote)
def get_market_quote(
    symbol: str,
    request: Request,
    _auth: None = Depends(_require_pub_or_basic),
) -> MarketQuote:
    """Get detailed market quote with real-time data from EODHD."""
    sid = _sid(request)
    if not _rate_limit_ok(sid, limit_per_hour=200):  # Rate limit for quote
        raise HTTPException(429, "rate limit exceeded")

    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(501, "EODHD_API_KEY not configured")

    # Get instrument info from database
    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")

    try:
        ps = (
            sess.query(PriceSeries)
            .filter(PriceSeries.symbol == sym)
            .order_by((PriceSeries.country == "US").desc())
            .first()
        )

        if ps is None:
            raise HTTPException(404, "symbol not found")

        name = ps.name or sym
        endpoint_symbol = resolve_eodhd_endpoint_symbol(sym)

        # Get real-time quote
        url = f"https://eodhistoricaldata.com/api/real-time/{endpoint_symbol}"
        params = {"api_token": api_key, "fmt": "json"}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            quote_data = resp.json()
        except Exception as exc:
            raise HTTPException(502, f"Failed to fetch quote: {exc}")

        # Get fundamental data
        fund_url = f"https://eodhistoricaldata.com/api/fundamentals/" \
                   f"{endpoint_symbol}"
        fund_params = {"api_token": api_key, "fmt": "json"}
        fundamentals = {}

        try:
            fund_resp = requests.get(fund_url, params=fund_params, timeout=30)
            fund_resp.raise_for_status()
            fundamentals = fund_resp.json()
        except Exception:
            pass  # Fundamentals are optional

        # Parse quote data
        current_price = float(quote_data.get("close", 0))
        previous_close = float(quote_data.get("previousClose", current_price))
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) \
            if previous_close > 0 else 0

        # Parse fundamental data
        highlights = fundamentals.get("Highlights", {})

        return MarketQuote(
            symbol=sym,
            name=name,
            current_price=current_price,
            change=change,
            change_percent=change_percent,
            open_price=(float(quote_data.get("open"))
                        if quote_data.get("open") else None),
            high=(float(quote_data.get("high"))
                  if quote_data.get("high") else None),
            low=(float(quote_data.get("low"))
                 if quote_data.get("low") else None),
            volume=(int(quote_data.get("volume"))
                    if quote_data.get("volume") else None),
            market_cap=highlights.get("MarketCapitalization"),
            pe_ratio=highlights.get("PERatio"),
            eps=highlights.get("EarningsShare"),
            year_high=highlights.get("52WeekHigh"),
            year_low=highlights.get("52WeekLow"),
            last_updated=(str(quote_data.get("timestamp"))
                          if quote_data.get("timestamp") else None)
        )
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/instrument/{symbol}/history",
            response_model=HistoricalDataResponse)
def get_historical_data(
    symbol: str,
    request: Request,
    period: str = Query("1Y", description="Time period: " +
                        "1D, 5D, 1M, 6M, YTD, 1Y, 5Y, Max"),
    _auth: None = Depends(_require_pub_or_basic),
) -> HistoricalDataResponse:
    """Get historical price data for charts."""
    sid = _sid(request)
    if not _rate_limit_ok(sid, limit_per_hour=200):  # Rate limit for history
        raise HTTPException(429, "rate limit exceeded")

    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(501, "EODHD_API_KEY not configured")

    endpoint_symbol = resolve_eodhd_endpoint_symbol(sym)

    # Calculate date range based on period
    today = datetime.now().date()

    period_mapping = {
        "1D": 1,
        "5D": 5,
        "1M": 30,
        "6M": 180,
        "YTD": (today - datetime(today.year, 1, 1).date()).days,
        "1Y": 365,
        "5Y": 365 * 5,
        "Max": None  # No limit
    }

    days = period_mapping.get(period, 365)

    url = f"https://eodhistoricaldata.com/api/eod/{endpoint_symbol}"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "order": "d"
    }

    if days:
        from_date = today - timedelta(days=days)
        params["from"] = from_date.isoformat()

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        raw_data = resp.json()
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch historical data: {exc}")

    if not isinstance(raw_data, list):
        raise HTTPException(502, "Invalid data format from EODHD")

    data_points = []
    for item in raw_data:
        try:
            adj_close = float(item.get("adjusted_close")) \
                if item.get("adjusted_close") else None
            volume = int(item.get("volume")) \
                if item.get("volume") else None
            close_price = float(item["close"])
            
            # Use adjusted_close for price if available and positive, otherwise use close
            price = adj_close if (adj_close is not None and adj_close > 0) else close_price
            
            data_points.append(HistoricalDataPoint(
                date=item["date"],
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=close_price,
                adjusted_close=adj_close,
                volume=volume,
                price=price
            ))
        except (KeyError, ValueError, TypeError):
            continue  # Skip invalid data points

    return HistoricalDataResponse(
        symbol=sym,
        period=period,
        data=data_points
    )


@router.post("/recommendations")
def recommendations(
    request: Request,
    body: RecommendationRequest,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """Get recommendations using Arman's correct algorithm."""
    t0 = time.time()
    sid = _sid(request)
    if not _rate_limit_ok(sid):
        raise HTTPException(429, "rate limit exceeded")

    try:
        lt_pct = float(body.loss_tolerance_pct)
    except Exception:
        raise HTTPException(400, "invalid loss_tolerance_pct")

    # Use Arman's correct algorithm via service
    # Configuration now comes from environment variables with sensible defaults
    service = CompassRecommendationsService()

    try:
        result = service.get_recommendations(
            loss_tolerance_pct=lt_pct,
            country=body.country or "US",  # Use country from request body
            seed_symbol=body.seed_symbol,
        )

        # Log performance and results
        dt = int((time.time() - t0) * 1000)
        try:
            _LOG.info(
                "%s",
                {
                    "event": "recommendations_view",
                    "algorithm": "arman_winsorized",
                    "pct": float(lt_pct),
                    "count": len(result.get("results", [])),
                    "latency_ms": dt,
                    "sid": sid,
                    "metadata": result.get("metadata", {}),
                },
            )
        except Exception:
            pass

        return result

    except Exception as e:
        _LOG.error("Recommendations service error: %s", e)
        raise HTTPException(500, f"Recommendation service error: {str(e)}")


def _compute_matches_payload(
    loss_tolerance_pct: float, seed_symbol: Optional[str] = None, country: str = "US"
) -> dict:
    """Compute matches using Arman's algorithm directly."""
    # Configuration now comes from environment variables with sensible defaults
    service = CompassRecommendationsService()

    return service.get_recommendations(
        loss_tolerance_pct=loss_tolerance_pct,
        country=country,
        seed_symbol=seed_symbol,
    )


def _search_candidates_us(query: str, limit: int = 5) -> list[dict]:
    """Legacy function for backward compatibility."""
    return _search_candidates(query, limit, country="US")


def _search_candidates(query: str, limit: int = 5, country: str = "US") -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    sess = get_db_session()
    if sess is None:
        return []
    try:
        from sqlalchemy.sql import func as _f  # type: ignore
        needle = f"%{q.lower()}%"
        base = sess.query(PriceSeries)
        
        # Filter by valid=1 products only
        base = base.filter(PriceSeries.valid == 1)
        
        # Filter by country if specified
        if country:
            base = base.filter(PriceSeries.country == country)
        
        recs = (
            base.filter(
                _f.lower(PriceSeries.symbol).like(needle)  # type: ignore
                | _f.lower(PriceSeries.name).like(needle)  # type: ignore
            )
            .limit(200)
            .all()
        )

        def _accept(rec: PriceSeries) -> bool:
            sy = (getattr(rec, "symbol", "") or "").lower()
            nm = (getattr(rec, "name", "") or "").lower()
            toks = [t for t in q.lower().split() if t]
            if not toks:
                return False
            par = ""
            if ")" in nm and "(" in nm:
                try:
                    s = nm.rfind("(")
                    e = nm.rfind(")")
                    if 0 <= s < e:
                        par = nm[s + 1: e].strip()
                except Exception:
                    par = ""
            base_nm = nm.split("(")[0].strip() if nm else ""
            for t in toks:
                if t in sy or t in nm or t == par or t == base_nm:
                    return True
            return False

        recs = [r for r in recs if _accept(r)]

        def _rank(rec: PriceSeries) -> tuple:
            sy = (getattr(rec, "symbol", "") or "").lower()
            nm = (getattr(rec, "name", "") or "").lower()
            ql = q.lower()

            paren = ""
            if ")" in nm and "(" in nm:
                try:
                    s = nm.rfind("(")
                    e = nm.rfind(")")
                    if 0 <= s < e:
                        paren = nm[s + 1: e].strip()
                except Exception:
                    paren = ""

            rank = 9
            if ql == sy or (paren and ql == paren):
                rank = 0
            elif ql == nm:
                rank = 1
            else:
                base_name = nm.split("(")[0].strip() if nm else ""
                if base_name and ql == base_name:
                    rank = 1
                elif sy.startswith(ql) or nm.startswith(ql):
                    rank = 2
                elif (ql in sy) or (ql in nm):
                    rank = 3

            toks = [t for t in ql.split() if t]
            matches = sum(1 for t in toks if t and (t in nm or t in sy))
            return (rank, len(nm or sy), -matches, sy)

        recs_sorted = sorted(recs, key=_rank)[: int(max(1, limit))]
        return [
            {
                "symbol": r.symbol,
                "name": (r.name or r.symbol),
                "type": _type_normalize(r.instrument_type),
                "country": r.country or "US",
            }
            for r in recs_sorted
        ]
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.post("/assistant", response_model=AssistantResponse)
def assistant_router(
    body: AssistantRequest, request: Request
) -> AssistantResponse:
    sid = _sid(request)
    if not _rate_limit_ok(sid, limit_per_hour=300):  # Rate limit for assistant
        raise HTTPException(429, "rate limit exceeded")

    user_msg = (body.message or "").strip()
    if not user_msg:
        return AssistantResponse(
            assistant_message=(
                "Please write a product name/ticker or your loss "
                "tolerance (e.g., 10%)."
            ),
            thread_id=body.thread_id,
            dialog=[],
        )

    # Defaults for thread/dialog and intent object
    thread_id = body.thread_id
    dialog: list[dict] = []
    intent_obj: Optional[AssistantIntent] = None

    # Coerce plain numeric inputs into MATCHES to avoid repeated ask_tol
    # Examples: "20", "20 pct", "20%" → -20
    try:
        m1 = _re.search(r"(-?\d+\.?\d*)\s*%", user_msg)
        m2 = _re.search(r"(\d+\.?\d*)\s*(percent|pct)", user_msg, _re.I)
        tol_val: float | None = None
        if m1:
            tol_val = float(m1.group(1))
        elif m2:
            tol_val = float(m2.group(1))
        else:
            if all(
                ch in "0123456789.- " for ch in user_msg.strip()
            ):
                tol_val = float(user_msg.strip())
        if tol_val is not None:
            tol_val = -abs(float(tol_val))
            intent_obj = AssistantIntent(
                action=AssistantAction.MATCHES,
                loss_tolerance_pct=tol_val,
            ).coerce()
    except Exception:
        pass

    # Heuristic: phrases referencing "my loss tolerance"
    # → reuse last tol if known
    try:
        if intent_obj is None:
            s_lc2 = (user_msg or "").strip().lower()
            if any(
                p in s_lc2
                for p in (
                    "within my loss tolerance",
                    "my loss tolerance",
                    "same tolerance",
                )
            ):
                last_tol = (_CTX.get(sid, {}) or {}).get("last_tol")
                if last_tol is not None:
                    intent_obj = AssistantIntent(
                        action=AssistantAction.MATCHES,
                        loss_tolerance_pct=float(last_tol),
                    ).coerce()
                else:
                    intent_obj = AssistantIntent(
                        action=AssistantAction.ASK_TOL
                    ).coerce()
    except Exception:
        pass

    # locale + LLM intent; Assistants API will persist dialog on OpenAI side
    locale = detect_locale(user_msg)
    if intent_obj is None:
        intent_obj, thread_id, dialog = call_llm_intent(
            user_msg, locale, session_id=sid, thread_id=body.thread_id
        )
    if intent_obj is None:
        intent_obj = fallback_intent(user_msg)

    # Rely on AI to choose weather/comment;
    # no heuristic domain forcing here

    ai_text = (intent_obj.assistant_text or "").strip()

    def _msg(k_en: str, k_ru: str) -> str:
        return k_ru if locale == "ru" else k_en

    def handle_clarify() -> AssistantResponse:
        s_lc = (user_msg or "").strip().lower()
        items_auto = _search_candidates_us(user_msg, limit=5)
        if items_auto:
            wants_details = any(
                w in s_lc
                for w in (
                    "detail", "details", "now for", "about",
                    "show me", "product"
                )
            )
            if wants_details:
                sym_pick = items_auto[0]["symbol"]
                try:
                    summ = _compute_summary_payload(sym_pick)
                    rp = {"pane": "instrument_summary", **summ}
                except Exception:
                    rp = {"pane": "none"}
                return AssistantResponse(
                    assistant_message=(
                        ai_text
                        or _msg("Showing details...", "Showing details...")
                    ),
                    candidates=items_auto,
                    right_pane=rp,
                    thread_id=thread_id,
                    dialog=dialog,
                )
        return AssistantResponse(
            assistant_message=(
                ai_text
                or _msg(
                    "Here is a list of products you may be interested in:",
                    "Here is a list of products you may be interested in:",
                )
            ),
            candidates=items_auto,
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

        try:
            prev_items = (_CTX.get(sid, {}) or {}).get("last_candidates") or []
        except Exception:
            prev_items = []
        finance_hint = any(
            k in s_lc
            for k in (
                "financial product", "finance", "ticker",
                "details", "show", "now", "yes"
            )
        )
        if prev_items and finance_hint:
            sym_pick = prev_items[0].get("symbol")
            if sym_pick:
                try:
                    summ = _compute_summary_payload(str(sym_pick))
                    rp = {"pane": "instrument_summary", **summ}
                except Exception:
                    rp = {"pane": "none"}
                return AssistantResponse(
                    assistant_message=(
                        ai_text
                        or _msg("Showing details...", "Showing details...")
                    ),
                    candidates=prev_items,
                    right_pane=rp,
                    thread_id=thread_id,
                    dialog=dialog,
                )
        return AssistantResponse(
            assistant_message=(
                ai_text
                or _msg(
                    "Which product do you mean (name or ticker)? "
                    "Or share a keyword, or your loss tolerance (e.g., 10%).",
                    "Which product do you mean (name or ticker)? "
                    "Or share a keyword, or your loss tolerance (e.g., 10%).",
                )
            ),
            candidates=[],
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_ask_tol() -> AssistantResponse:
        return AssistantResponse(
            assistant_message=(
                ai_text
                or _msg(
                    "What loss tolerance should I use? (e.g., 10%)",
                    "What loss tolerance should I use? (e.g., 10%)",
                )
            ),
            candidates=[],
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_matches() -> AssistantResponse:
        lt = intent_obj.loss_tolerance_pct
        if lt is None or not (lt == lt):
            return handle_ask_tol()
        data = _compute_matches_payload(float(lt), country=body.country or "US")
        try:
            _CTX.setdefault(sid, {})["mode"] = "finance"
            _CTX[sid]["last_tol"] = float(lt)
        except Exception:
            pass
        return AssistantResponse(
            assistant_message=(
                ai_text
                or _msg(
                    "Here are search results ranked by Compass Score.",
                    "Here are search results ranked by Compass Score.",
                )
            ),
            right_pane={"pane": "matches", **data},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_summary() -> AssistantResponse:
        q_sym = (intent_obj.symbol or "").strip()
        try:
            mode_prev = _CTX.get(sid, {}).get("mode")
        except Exception:
            mode_prev = None
        # For summary/details, search globally without country filter
        items = _search_candidates(q_sym or user_msg, limit=5, country=None)
        try:
            if items and mode_prev == "finance":
                _CTX.setdefault(sid, {})["last_candidates"] = items
        except Exception:
            pass
        try:
            prev_items2 = _CTX.get(sid, {}).get("last_candidates") or []
        except Exception:
            prev_items2 = []
        sym_pick = (
            q_sym
            or (items[0]["symbol"] if items else "")
            or (prev_items2[0].get("symbol") if prev_items2 else "")
        ).strip()
        # Chat-first summary: AI text only;
        # UI will render CVaR and fetch values
        if sym_pick:
            msg = (
                ai_text or f"Showing details for {sym_pick}."
            ).strip()
            return AssistantResponse(
                assistant_message=msg,
                candidates=[],
                right_pane={"pane": "none"},
                thread_id=thread_id,
                dialog=dialog,
                summary_symbol=sym_pick,
            )

        # Otherwise act like a candidate search list in chat
        msg_list = ai_text or (
            _msg(
                "Here is a list of products you may be interested in:",
                "Here is a list of products you may be interested in:",
            )
            if items
            else _msg(
                "I didn't find products matching your query.",
                "I didn't find products matching your query.",
            )
        )
        return AssistantResponse(
            assistant_message=msg_list,
            candidates=items,
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_candidates() -> AssistantResponse:
        q = (intent_obj.query or "").strip()
        items = _search_candidates(q or user_msg, limit=5, country=body.country or "US")
        try:
            if items:
                _CTX.setdefault(sid, {})["mode"] = "finance"
                _CTX[sid]["last_candidates"] = items
        except Exception:
            pass
        msg = ai_text or (
            _msg(
                "Here is a list of products you may be interested in:",
                "Here is a list of products you may be interested in:",
            )
            if items
            else _msg(
                "I didn't find products matching your query.",
                "I didn't find products matching your query.",
            )
        )
        return AssistantResponse(
            assistant_message=msg,
            candidates=items,
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_comment() -> AssistantResponse:
        # Pure small-talk output from AI; avoid echoing user text
        text = ai_text or (
            "Happy to help. If you need, I can pull product details "
            "or suggest matches by your loss tolerance."
        )
        return AssistantResponse(
            assistant_message=text,
            candidates=[],
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_weather() -> AssistantResponse:
        # Use AI-provided query (location).
        # No regex on user text; no echo
        loc = (intent_obj.query or "").strip()
        if not loc:
            return AssistantResponse(
                assistant_message=(
                    ai_text
                    or "Please provide a location "
                    "(e.g., 'weather in London')."
                ),
                candidates=[],
                right_pane={"pane": "none"},
                thread_id=thread_id,
                dialog=dialog,
            )
        out_txt = ai_text or ""
        try:
            from services.external.weather import (
                get_current_by_text,
            )  # type: ignore
            wx = get_current_by_text(loc)
            if wx and wx.get("temperature_c") is not None and not out_txt:
                city = wx.get("city") or loc
                desc = wx.get("description") or "weather"
                temp = wx.get("temperature_c")
                wind = wx.get("windspeed_ms")
                out_txt = (
                    f"{city}: {desc}, {temp:.0f}°C, wind {wind:.0f} m/s."
                )
        except Exception:
            pass
        if not out_txt:
            out_txt = "I couldn't retrieve weather right now."
        return AssistantResponse(
            assistant_message=out_txt,
            candidates=[],
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    def handle_help() -> AssistantResponse:
        return AssistantResponse(
            assistant_message=(
                ai_text
                or _msg(
                    "I can show product details or pick matches. "
                    "Try 'details for Apple' or "
                    "'I don't want to lose more than 10%'.",
                    "I can show product details or pick matches. "
                    "Try 'details for Apple' or "
                    "'I don't want to lose more than 10%'.",
                )
            ),
            right_pane={"pane": "none"},
            thread_id=thread_id,
            dialog=dialog,
        )

    handlers = {
        AssistantAction.CLARIFY: handle_clarify,
        AssistantAction.ASK_TOL: handle_ask_tol,
        AssistantAction.MATCHES: handle_matches,
        AssistantAction.SUMMARY: handle_summary,
        AssistantAction.CANDIDATES: handle_candidates,
        AssistantAction.COMMENT: handle_comment,
        AssistantAction.WEATHER: handle_weather,
        AssistantAction.HELP: handle_help,
    }

    return handlers.get(intent_obj.action, handle_help)()
