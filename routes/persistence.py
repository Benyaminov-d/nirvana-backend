from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import Response
import io
from typing import Iterable
from utils.auth import (
    basic_auth_if_configured as _basic_auth_if_configured,
    require_pub_or_basic as _require_pub_or_basic,
)
from core.db import get_db_session
from core.models import CvarSnapshot, PriceSeries


router = APIRouter()


@router.get("/history")
def history(
    symbol: str,
    alpha: int = Query(95, description="50, 95 or 99"),
    limit: int = 200,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    _ = _basic_auth_if_configured
    _basic_auth_if_configured  # type: ignore
    if alpha not in (50, 95, 99):
        alpha = 95
    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        q = (
            sess.query(CvarSnapshot)
            .filter(
                CvarSnapshot.symbol == symbol,
                CvarSnapshot.alpha_label == alpha,
            )
            .order_by(CvarSnapshot.as_of_date.desc())
        )
        rows = q.limit(max(1, min(limit, 1000))).all()
        out: list[dict] = []
        for r in reversed(rows):  # chronological
            vals_raw = [r.cvar_nig, r.cvar_ghst, r.cvar_evar]
            vals = [v for v in vals_raw if v is not None]
            worst = max(vals) if vals else None
            item = {
                "as_of": r.as_of_date.isoformat(),
                "years": r.years,
                "alpha": r.alpha_label,
                "nig": r.cvar_nig,
                "ghst": r.cvar_ghst,
                "evar": r.cvar_evar,
                "worst": worst,
            }
            out.append(item)
        return {"symbol": symbol, "alpha": alpha, "items": out}
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/export_cvars")
def export_cvars(_auth: None = Depends(_require_pub_or_basic)) -> Response:
    """Deprecated: kept for backward compatibility. Use /export/cvars."""
    from routes.export import export_cvars_text  # lazy import to avoid cycle
    return export_cvars_text()  # type: ignore

@router.get("/export_cvars_csv")
def export_cvars_csv(
    _auth: None = Depends(_require_pub_or_basic), levels: str = Query(""), level: str = Query("")
) -> Response:
    """Deprecated: kept for backward compatibility. Use /export/cvars.csv."""
    from routes.export import export_cvars_csv as _csv_impl  # lazy import

    return _csv_impl(levels=levels, level=level)  # type: ignore


@router.get("/purge/symbol")
def purge_symbol(
    symbol: str = Query(..., description="Symbol to purge from DB"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """Delete all rows for a symbol from PriceSeries and CvarSnapshot.

    Auth: requires public or basic auth (same as other protected endpoints).
    """
    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    deleted_cvar = 0
    deleted_ps = 0
    try:
        try:
            deleted_cvar = (
                sess.query(CvarSnapshot)
                .filter(CvarSnapshot.symbol == symbol)
                .delete(synchronize_session=False)
            )
        except Exception:
            sess.rollback()
            raise
        try:
            deleted_ps = (
                sess.query(PriceSeries)
                .filter(PriceSeries.symbol == symbol)
                .delete(synchronize_session=False)
            )
        except Exception:
            sess.rollback()
            raise
        sess.commit()
        return {"symbol": symbol, "deleted_cvar": int(deleted_cvar), "deleted_price_series": int(deleted_ps)}
    except HTTPException:
        raise
    except Exception as exc:
        try:
            sess.rollback()
        except Exception:
            pass
        raise HTTPException(500, f"purge failed: {exc}")
    finally:
        try:
            sess.close()
        except Exception:
            pass