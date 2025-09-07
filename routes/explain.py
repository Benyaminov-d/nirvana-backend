from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, Query

from utils.auth import require_pub_or_basic as _require_pub_or_basic
from services.explain import generate_explanation
from core.db import get_db_session
from core.models import PriceSeries


router = APIRouter()


@router.get("/explain")
def explain(
    symbol: str = Query(..., description="Symbol to explain, e.g., BTC"),
    c50: float | None = Query(None, description="Worst-of annual CVaR 50 as fraction (0.25=25%)"),
    c95: float | None = Query(None, description="Worst-of annual CVaR 95 as fraction"),
    c99: float | None = Query(None, description="Worst-of annual CVaR 99 as fraction"),
    as_of: str | None = Query(None, description="As-of date (YYYY-MM-DD) for context"),
    total_return: float | None = Query(None, description="Total return over history (percent) if available"),
    years: float | None = Query(None, description="Years of data observed if available"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(400, "symbol required")
    # Fetch name for nicer header; DB is optional
    sess = get_db_session()
    name = None
    meta: dict[str, str | None] = {}
    if sess is not None:
        try:
            row = (
                sess.query(
                    PriceSeries.name,
                    PriceSeries.country,
                    PriceSeries.exchange,
                    PriceSeries.currency,
                    PriceSeries.instrument_type,
                    PriceSeries.isin,
                )  # type: ignore
                .filter(PriceSeries.symbol == sym)
                .one_or_none()
            )
            if row:
                name = row[0]
                meta = {
                    "country": row[1],
                    "exchange": row[2],
                    "currency": row[3],
                    "type": row[4],
                    "isin": row[5],
                    "as_of": as_of,
                    "total_return_pct": total_return,
                    "years": years,
                }
        except Exception:
            name = None
        finally:
            try:
                sess.close()
            except Exception:
                pass
    # Optional CVaR summary values (fractions)
    cvar_vals = None
    try:
        if any(v is not None for v in (c50, c95, c99)):
            cvar_vals = {"c50": c50, "c95": c95, "c99": c99}
    except Exception:
        cvar_vals = None
    data = generate_explanation(sym, name, cvar=cvar_vals, meta=meta)
    return {"symbol": sym, "name": name, **data}


