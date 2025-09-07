from fastapi import APIRouter, HTTPException, Query, Depends  # type: ignore
from fastapi.responses import HTMLResponse, StreamingResponse  # type: ignore
from pathlib import Path
from services.domain.cvar_unified_service import CvarUnifiedService
from config import get_config as _get_config
from utils.auth import (
    basic_auth_if_configured as _basic_auth_if_configured,
    require_pub_or_basic as _require_pub_or_basic,
    mint_public_token as _mint_public_token,
)
from utils.common import parse_csv_list as _parse_csv_list
from utils.common import db_base_symbol as _db_base_symbol
from utils.common import _eodhd_suffix_for as _suffix_for
from core.db import get_db_session
from core.models import PriceSeries
from core.db_models.snapshot import CvarSnapshot
from core.persistence import upsert_snapshot_row
from utils import service_bus as _sb
import asyncio
import json as _json
import time as _time
import os
import logging as _log


router = APIRouter()


FRONT = Path(__file__).parents[2] / "frontend"


@router.post("/cvar/recalc_all")
@router.get("/cvar/recalc_all")
def cvar_recalc_all_legacy(
    limit: int = Query(0, description="0=all, >0 to limit symbols"),
    products: str = Query(
        "",
        description="Optional comma-separated symbols to restrict",
    ),
    force: bool = True,
    local: bool = Query(
        False,
        description="If true, compute locally now instead of enqueuing",
    ),
    five_stars: bool = Query(
        False,
        description=(
            "If true and products not provided, process only five-star symbols"
        ),
    ),
    ready_only: bool = Query(
        True,
        description=(
            "When true (default), only symbols with sufficient "
            "history (insufficient_history=0)"
        ),
    ),
    include_unknown: bool = Query(
        False,
        description=(
            "When true, include symbols with unknown status "
            "(insufficient_history IS NULL)"
        ),
    ),
    country: str | None = Query(
        None,
        description="Optional country filter when building symbol list",
    ),
    types: str | None = Query(
        None,
        description=(
            "Optional comma-separated instrument types "
            "(e.g., 'Mutual Fund,ETF')"
        ),
    ),
    _auth: None = Depends(_require_pub_or_basic),
    file_for_single: str | None = Query(
        None,
        description=(
            "Optional local CSV path to use when products has exactly "
            "one symbol"
        ),
    ),
    verbose: int = Query(
        0,
        description="Set to 1 to return per-symbol details (local mode)",
    ),
    service: str | None = Query(
        None,
        description="Override service mode: local | remote | auto",
    ),
    nav_interp: int | None = Query(
        None,
        description=(
            "When set (1/0), temporarily enable/disable NAV interpolation "
            "for this request"
        ),
    ),
    exclude_exchange: str | None = Query(
        None,
        description=(
            "Optional comma-separated list of exchanges to exclude "
            "(e.g., 'PINK,OTC')"
        ),
    ),
    legacy: bool = Query(
        False,
        description="DEPRECATED: Use /cvar/recalc-orchestrated for new architecture"
    )
) -> dict:
    """
    LEGACY ENDPOINT - Use /api/orchestration/batch-recalc for new architecture.
    
    This endpoint is maintained for backward compatibility but will be deprecated.
    The new orchestrated batch recalculation provides better performance,
    error handling, and monitoring capabilities.
    """
    
    # REDIRECT TO NEW ARCHITECTURE - for backward compatibility during migration
    from services.application.cvar_orchestration_service import CvarOrchestrationService
    from utils.common import parse_csv_list
    
    # Initialize orchestration service
    orchestration_service = CvarOrchestrationService()
    
    try:
        # Parse symbols if provided
        symbol_list = None
        if products:
            symbol_list = parse_csv_list(products)
        
        # Prepare filters for symbol selection
        filters = None if symbol_list else {
            "five_stars": five_stars,
            "ready_only": ready_only,
            "include_unknown": include_unknown,
            "country": country,
            "instrument_types": [t.strip() for t in types.split(",") if t.strip()] if types else None,
            "exclude_exchanges": [e.strip() for e in exclude_exchange.split(",") if e.strip()] if exclude_exchange else None,
            "limit": limit if limit > 0 else None
        }
        
        # Execute using new orchestration service
        result = orchestration_service.batch_recalculate_cvar(
            symbols=symbol_list,
            filters=filters,
            max_workers=4 if local else 6,  # Adjust workers based on local/remote
            verbose=bool(verbose)
        )
        
        # Format response to match legacy structure
        legacy_response = {
            "symbols": result.get("symbols_processed", 0),
            "submitted": result.get("successful_calculations", 0),
            "failed": result.get("failed_calculations", 0),
            "architecture": "orchestrated_refactored",
            "legacy_compatibility": True,
            "migration_message": "This request was processed using new architecture for better performance",
            "new_endpoint_recommendation": "/api/orchestration/batch-recalc"
        }
        
        # Add detailed results if verbose
        if verbose and result.get("statistics") and result["statistics"].get("details"):
            legacy_response["details"] = result["statistics"]["details"]
        
        return legacy_response
        
    except Exception as e:
        logger.error(f"Legacy recalc_all redirection failed: {e}")
        raise HTTPException(500, f"Recalculation failed: {str(e)}")


@router.post("/cvar/recalc-orchestrated") 
@router.get("/cvar/recalc-orchestrated")
def cvar_recalc_orchestrated(
    limit: int = Query(0, description="0=all, >0 to limit symbols"),
    products: str = Query(
        "",
        description="Optional comma-separated symbols to restrict",
    ),
    force: bool = Query(True, description="Force recalculation even if cached data exists"),
    five_stars: bool = Query(
        False,
        description="If true and products not provided, process only five-star symbols",
    ),
    ready_only: bool = Query(
        True,
        description="When true (default), only symbols with sufficient history",
    ),
    include_unknown: bool = Query(
        False,
        description="When true, include symbols with unknown status",
    ),
    country: str | None = Query(
        None,
        description="Optional country filter when building symbol list",
    ),
    types: str | None = Query(
        None,
        description="Optional comma-separated instrument types (e.g., 'Mutual Fund,ETF')",
    ),
    exclude_exchange: str | None = Query(
        None,
        description="Optional comma-separated list of exchanges to exclude",
    ),
    max_workers: int = Query(
        6, 
        description="Number of parallel workers (1-8)",
    ),
    verbose: bool = Query(
        False,
        description="Return detailed per-symbol results",
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """
    Modern orchestrated CVaR batch recalculation with clean architecture.
    
    This endpoint replaces /cvar/recalc_all with improved:
    - Clean separation of concerns using orchestration services
    - Better parallel processing and resource management  
    - Comprehensive monitoring and error handling
    - No direct database access in route handler
    - Type-safe repository operations
    """
    
    from services.application.cvar_orchestration_service import CvarOrchestrationService
    from utils.common import parse_csv_list
    
    # Initialize orchestration service
    orchestration_service = CvarOrchestrationService()
    
    try:
        # Parse and validate inputs
        symbol_list = None
        if products:
            symbol_list = parse_csv_list(products)
            logger.info(f"Processing specific symbols: {len(symbol_list)} provided")
        
        # Prepare filters for symbol selection  
        filters = None if symbol_list else {
            "five_stars": five_stars,
            "ready_only": ready_only,
            "include_unknown": include_unknown,
            "country": country,
            "instrument_types": [t.strip() for t in types.split(",") if t.strip()] if types else None,
            "exclude_exchanges": [e.strip() for e in exclude_exchange.split(",") if e.strip()] if exclude_exchange else None,
            "limit": limit if limit > 0 else None
        }
        
        # Validate max_workers parameter
        max_workers = max(1, min(8, max_workers))
        
        # Execute orchestrated batch recalculation
        result = orchestration_service.batch_recalculate_cvar(
            symbols=symbol_list,
            filters=filters,
            max_workers=max_workers,
            verbose=verbose
        )
        
        # Add endpoint metadata
        result["endpoint_info"] = {
            "endpoint": "/cvar/recalc-orchestrated",
            "architecture": "clean_orchestrated",
            "features": [
                "Parallel processing with configurable workers",
                "Repository-based data access",
                "Domain service business logic",
                "Application service orchestration",
                "Comprehensive error handling"
            ],
            "parameters_applied": {
                "max_workers": max_workers,
                "force_recalculation": force,
                "filters": filters or "specific_symbols"
            }
        }
        
        logger.info(
            f"Orchestrated recalculation completed: {result.get('successful_calculations', 0)}"
            f"/{result.get('symbols_processed', 0)} symbols processed"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Orchestrated CVaR recalculation failed: {e}")
        raise HTTPException(500, f"Orchestrated recalculation failed: {str(e)}")


# Legacy recalc_all function implementation removed during refactoring
# The functionality has been replaced by:
# - cvar_recalc_all_legacy() - backward compatible endpoint that redirects to new architecture  
# - cvar_recalc_orchestrated() - clean implementation using orchestration services
# This removes ~420 lines of legacy SQL-heavy code

@router.post("/cvar/recalc_canada_mf_etf")
@router.get("/cvar/recalc_canada_mf_etf")
def cvar_recalc_canada_mf_etf(
    local: bool = Query(
        False,
        description=(
            "If true, compute locally/now (or via function when service="
            "remote)"
        ),
    ),
    service: str | None = Query(
        None,
        description="Override service mode: local | remote | auto",
    ),
    include_unknown: bool = Query(
        False,
        description=(
            "Include unknown insufficient_history when building symbol set"
        ),
    ),
    limit: int = Query(0, description="0=all, >0 to limit symbols"),
    verbose: int = Query(0, description="Return per-symbol details (local)"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    # Delegate to recalc_all with fixed country/types
    return cvar_recalc_all(  # type: ignore[call-arg]
        limit=limit,
        products="",
        force=True,
        local=local,
        five_stars=False,
        ready_only=True,
        include_unknown=include_unknown,
        country="Canada",
        types="Mutual Fund,ETF",
        _auth=None,  # already authorized above
        file_for_single=None,
        verbose=verbose,
        service=service,
    )


@router.post("/cvar/recalc_symbol_remote")
def cvar_recalc_symbol_remote(
    symbol: str = Query(..., description="Symbol to recalc via function"),
    recalculate: bool = Query(
        True, description="Force recalc in function"
    ),
    todate: str | None = Query(
        None, description="As-of date override (optional)"
    ),
    service: str | None = Query(
        "remote",
        description="Service mode override (default remote)",
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """Compute CVaR for a single symbol via Azure Function (or chosen service)
    and persist snapshots for 50/95/99.
    """
    svc = CvarUnifiedService(mode=service or _get_config().cvar_service_mode)
    data = svc.get_cvar_data(
        symbol,
        force_recalculate=bool(recalculate),
        to_date=todate,
        prefer_local=True,
    )
    if not isinstance(data, dict) or not data.get("success"):
        # If loader gate tripped locally, mark insufficient_history in DB
        try:
            code = data.get("code") if isinstance(data, dict) else None
            if code == "insufficient_history":
                sess = get_db_session()
                if sess is not None:
                    try:
                        _base = _db_base_symbol(symbol)
                        rec = (
                            sess.query(PriceSeries)
                            .filter(
                                PriceSeries.symbol == _base  # type: ignore
                            )
                            .one_or_none()
                        )
                        if rec is not None:
                            rec.insufficient_history = 1
                            sess.commit()
                    finally:
                        try:
                            sess.close()
                        except Exception:
                            pass
        except Exception:
            pass
        reason = data.get("error") if isinstance(data, dict) else "unknown"
        raise HTTPException(502, f"function_failed: {reason}")
    as_of_s = str(data.get("as_of_date"))
    try:
        from datetime import datetime as _dt
        as_of_date = _dt.fromisoformat(as_of_s).date()
    except Exception:
        raise HTTPException(422, f"bad_as_of_date: {as_of_s}")

    def _flt(x: object | None) -> float | None:
        try:
            if x is None:
                return None
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return None

    # Persist 50/95/99 blocks and report actual persisted labels
    from datetime import datetime as _dt
    persisted: list[int] = []
    _u_logger = _log.getLogger("cvar_upsert")
    # Extract optional years/returns/start_date from remote payload if present
    try:
        years_val = None
        yv = data.get("years")
        if yv is None:
            _summ_obj = data.get("summary")
            summ = _summ_obj if isinstance(_summ_obj, dict) else {}
            yv = summ.get("years") if isinstance(summ, dict) else None
        if yv is None:
            # local calculator exposes years under data_summary
            _ds = data.get("data_summary")
            if isinstance(_ds, dict):
                yv = _ds.get("years")
        years_val = _flt(yv)  # type: ignore[assignment]
    except Exception:
        years_val = None
    try:
        return_as_of = _flt(data.get("return_as_of"))
    except Exception:
        return_as_of = None
    try:
        return_annual = _flt(data.get("annual_return"))
    except Exception:
        return_annual = None
    # When using local compute, payload may not include returns; derive if
    # missing
    if return_as_of is None and return_annual is None:
        try:
            from services.prices import (
                load_prices as _load_prices,  # type: ignore
            )
            p = _load_prices(symbol, to_date=None)
            if isinstance(p, dict) and p.get("success"):
                r2: list[float] = []
                try:
                    _ro = p.get("returns")
                    if isinstance(_ro, (list, tuple)):
                        r2 = [float(x) for x in _ro]
                    else:
                        import numpy as _np  # type: ignore
                        arr = _np.asarray(_ro, dtype=float)
                        if hasattr(arr, "tolist"):
                            r2 = [float(x) for x in arr.tolist()]
                except Exception:
                    r2 = []
                if r2:
                    try:
                        return_as_of = float(r2[-1])
                    except Exception:
                        return_as_of = None
                    try:
                        import math as _m
                        window = 252
                        tail = r2[-window:] if len(r2) > window else r2
                        acc = 1.0
                        for rr in tail:
                            try:
                                acc *= (1.0 + float(rr))
                            except Exception:
                                continue
                        return_annual = float(acc - 1.0)
                        if not (
                            _m.isfinite(return_annual)
                            if return_annual is not None
                            else False
                        ):
                            return_annual = None
                    except Exception:
                        return_annual = None
                if years_val is None:
                    try:
                        y2 = (p.get("summary") or {}).get("years")
                        years_val = (
                            float(y2) if y2 is not None else None
                        )
                    except Exception:
                        years_val = years_val
        except Exception:
            pass
    for label in (50, 95, 99):
        key = f"cvar{label}"
        blk = data.get(key) or {}
        ann = blk.get("annual") if isinstance(blk, dict) else {}
        # Log values before DB write
        try:
            _u_logger.info(
                (
                    "upsert try: symbol=%s as_of=%s label=%d "
                    "nig=%s ghst=%s evar=%s alpha=%s"
                ),
                symbol,
                as_of_s,
                int(label),
                ann.get("nig") if isinstance(ann, dict) else None,
                ann.get("ghst") if isinstance(ann, dict) else None,
                ann.get("evar") if isinstance(ann, dict) else None,
                blk.get("alpha") if isinstance(blk, dict) else None,
            )
        except Exception:
            pass
        ok = upsert_snapshot_row(
            symbol=symbol,
            as_of_date=as_of_date,
            alpha_label=int(label),
            alpha_conf=(
                _flt(blk.get("alpha")) if isinstance(blk, dict) else None
            ),
            years=years_val,
            cvar_nig=_flt(getattr(ann, "get", lambda *_: None)("nig")),
            cvar_ghst=_flt(getattr(ann, "get", lambda *_: None)("ghst")),
            cvar_evar=_flt(getattr(ann, "get", lambda *_: None)("evar")),
            source="remote",
            return_as_of=return_as_of,
            return_annual=return_annual,
        )
        if ok:
            persisted.append(int(label))
        else:
            try:
                _u_logger.warning(
                    "upsert failed: symbol=%s as_of=%s label=%d",
                    symbol,
                    as_of_s,
                    int(label),
                )
            except Exception:
                pass
    # On any success, clear insufficient_history flag for base symbol
    if persisted:
        try:
            _base = _db_base_symbol(symbol)
            sess = get_db_session()
            if sess is not None:
                try:
                    rec = sess.query(PriceSeries).filter(  # type: ignore
                        PriceSeries.symbol == _base
                    ).one_or_none()
                    if rec is not None and getattr(rec, "insufficient_history", None) != 0:
                        rec.insufficient_history = 0
                        sess.commit()
                finally:
                    try:
                        sess.close()
                    except Exception:
                        pass
        except Exception:
            pass
    if not persisted:
        raise HTTPException(500, "persist_failed: no labels saved")
    return {
        "symbol": symbol,
        "as_of": as_of_s,
        "persisted_labels": persisted,
        "service": svc.mode,
    }


@router.get("/cvar", response_class=HTMLResponse)
def cvar_page(
    _auth: None = Depends(_basic_auth_if_configured),
) -> HTMLResponse:
    # Prefer new location; fallback to legacy old_front/templates
    page = FRONT / "templates" / "cvar.html"
    if not page.exists():
        legacy = FRONT / "old_front" / "templates" / "cvar.html"
        if legacy.exists():
            page = legacy
    if page.exists():
        return HTMLResponse(page.read_text(), 200)
    raise HTTPException(404, "cvar.html not found")


@router.get("/cvar/calculator", response_class=HTMLResponse)
def cvar_calculator_page(
    _auth: None = Depends(_basic_auth_if_configured),
) -> HTMLResponse:
    # Prefer new location; fallback to legacy old_front/templates
    page = FRONT / "templates" / "cvar_manual.html"
    if not page.exists():
        legacy = FRONT / "old_front" / "templates" / "cvar_manual.html"
        if legacy.exists():
            page = legacy
    if page.exists():
        content = page.read_text()
        tok = _mint_public_token()
        if tok:
            resp = HTMLResponse(content, 200)
            import os as _os
            cookie_domain = _os.getenv("NIR_COOKIE_DOMAIN") or None
            _flag = _os.getenv("NIR_COOKIE_SECURE", "0")
            cookie_secure = _flag.lower() in ("1", "true", "yes")
            resp.set_cookie(
                key="nir_pub",
                value=tok,
                httponly=True,
                samesite="lax",
                max_age=600,
                path="/",
                domain=cookie_domain,
                secure=cookie_secure,
            )
            return resp
        return HTMLResponse(content, 200)
    raise HTTPException(404, "cvar_manual.html not found")


@router.get("/lambert/benchmarks")
def lambert_benchmarks(
    symbol: str = "SP500TR",
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    _ = _basic_auth_if_configured  # enforce auth
    _basic_auth_if_configured  # type: ignore
    svc = CvarUnifiedService(mode=getattr(_get_config(), 'cvar_service_mode', 'local'))
    try:
        data = svc.get_lambert_benchmarks(symbol)
    except Exception as exc:
        raise HTTPException(404, str(exc))
    return {"symbol": symbol, **data}


@router.get("/cvar/curve")
def cvar_curve(
    symbol: str = "BTC",
    alpha: int = Query(95, description="50, 95 or 99"),
    recalculate: bool = False,
    todate: str | None = Query(
        None,
        description=("As-of date: ddmmyyyy or yyyymmdd or ISO"),
    ),
    service: str | None = Query(
        None,
        description="Override service mode: local | remote | auto (ignored in new architecture)",
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """
    Return full annual CVaR block (Lambert style).
    
    REFACTORED: Now uses new domain service architecture instead of direct DB access.
    """
    from services.domain.cvar_unified_service import CvarUnifiedService
    from repositories import PriceSeriesRepository
    
    # Initialize services using new architecture
    cvar_service = CvarUnifiedService()
    price_repo = PriceSeriesRepository()
    
    # Validate alpha parameter
    if alpha not in (50, 95, 99):
        alpha = 95
    
    try:
        # Get CVaR data using unified domain service
        data = cvar_service.get_cvar_data(
            symbol=symbol,
            force_recalculate=recalculate,
            to_date=todate,  # Historical calculation support
            prefer_local=True  # Service parameter ignored for now
        )
        
        # Handle calculation failures with proper error handling
        if not data.get("success"):
            error_code = data.get("code", "calc_failed")
            
            # Update insufficient_history flag using repository (no direct SQL)
            if error_code == "insufficient_history":
                try:
                    from utils.common import db_base_symbol as _db_base_symbol
                    base_symbol = _db_base_symbol(symbol)
                    price_repo.update_insufficient_history(base_symbol, 1)
                except Exception as e:
                    logger.warning(f"Failed to update insufficient_history for {symbol}: {e}")
            
            # Return structured error response
            detail = data if isinstance(data, dict) else {"detail": str(data)}
            status = 422 if error_code == "insufficient_history" else 500
            raise HTTPException(status, detail)
        
        # Extract the requested alpha block
        key = "cvar50" if alpha == 50 else ("cvar99" if alpha == 99 else "cvar95")
        block = data.get(key, {})
        
        if not block:
            raise HTTPException(404, f"No CVaR data available for alpha {alpha}")
        
        # Get instrument name using repository (no direct SQL)
        inst_name = price_repo.get_symbol_name(symbol)
        
        # Return clean response in Lambert style
        return {
            "alpha": alpha,
            "annual": block.get("annual", {}),  # nig / ghst / evar (252-day)
            "snapshot": block.get("snapshot", {}),  # identical numbers - kept for UI
            "alpha_used": block.get("alpha"),
            "as_of": data.get("as_of_date"),
            "symbol": symbol,
            "name": inst_name,
            "cached": data.get("cached", False),
            "architecture": "refactored_domain_service"  # Indicates new architecture
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CVaR curve failed for {symbol}: {e}")
        raise HTTPException(500, f"Failed to get CVaR curve: {str(e)}")


@router.get("/cvar/curve-all")
def cvar_curve_all(
    symbol: str = "BTC",
    recalculate: bool = False,
    todate: str | None = Query(
        None,
        description=("As-of date: ddmmyyyy or yyyymmdd or ISO"),
    ),
    service: str | None = Query(
        None,
        description="Override service mode: local | remote | auto",
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """Return both 95% and 99% CVaR blocks in one response."""
    svc = CvarUnifiedService(mode=service or _get_config().cvar_service_mode)
    data = svc.get_cvar_data(
        symbol,
        force_recalculate=recalculate,
        to_date=todate,
        prefer_local=True,
    )
    if not data.get("success"):
        # Mark insufficient_history in DB when loader gate trips
        try:
            code = data.get("code") if isinstance(data, dict) else None
            if code == "insufficient_history":
                sess = get_db_session()
                if sess is not None:
                    try:
                        _base = _db_base_symbol(symbol)
                        rec = (
                            sess.query(PriceSeries)
                            .filter(
                                PriceSeries.symbol == _base  # type: ignore
                            )
                            .one_or_none()
                        )
                        if rec is not None:
                            rec.insufficient_history = 1
                            sess.commit()
                    finally:
                        try:
                            sess.close()
                        except Exception:
                            pass
        except Exception:
            pass
        detail = data if isinstance(data, dict) else {"detail": str(data)}
        status = 422 if data.get("code") == "insufficient_history" else 500
        raise HTTPException(status, detail)
    return {
        "symbol": symbol,
        "as_of": data["as_of_date"],
        "cached": data.get("cached", False),
        "cvar50": {
            "annual": data["cvar50"]["annual"],
            "snapshot": data["cvar50"]["snapshot"],
            "alpha_used": data["cvar50"].get("alpha"),
            "as_of": data["as_of_date"],
            "cached": data.get("cached", False),
            "lambert": data["cvar50"].get("lambert"),
        },
        "cvar95": {
            "annual": data["cvar95"]["annual"],
            "snapshot": data["cvar95"]["snapshot"],
            "alpha_used": data["cvar95"].get("alpha"),
            "as_of": data["as_of_date"],
            "cached": data.get("cached", False),
            "lambert": data["cvar95"].get("lambert"),
        },
        "cvar99": {
            "annual": data["cvar99"]["annual"],
            "snapshot": data["cvar99"]["snapshot"],
            "alpha_used": data["cvar99"].get("alpha"),
            "as_of": data["as_of_date"],
            "cached": data.get("cached", False),
            "lambert": data["cvar99"].get("lambert"),
        },
    }


@router.get("/cvar/snapshot")
def cvar_snapshot(
    symbol: str = "BTC",
    alpha: int = Query(95, description="50, 95 or 99"),
    recalculate: bool = False,
    service: str | None = Query(None, description="Override service mode"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    # REFACTORED: Using new domain service architecture
    from services.domain.cvar_unified_service import CvarUnifiedService
    
    # Initialize unified domain service 
    cvar_service = CvarUnifiedService()
    
    # Get CVaR data using new service
    data = cvar_service.get_cvar_data(
        symbol=symbol,
        force_recalculate=recalculate,
        prefer_local=True  # service parameter ignored for now
    )
    
    # Validate alpha parameter
    if alpha not in (50, 95, 99):
        alpha = 95
    
    # Check if data is valid
    if not data.get("success"):
        detail = data if isinstance(data, dict) else {"detail": str(data)}
        status = 422 if data.get("code") == "insufficient_history" else 500
        raise HTTPException(status, detail)
    
    # Extract the requested alpha block
    key = "cvar50" if alpha == 50 else ("cvar99" if alpha == 99 else "cvar95")
    block = data.get(key, {})
    
    # Check if block has valid values, if not force recalculation
    if isinstance(block, dict):
        ann = block.get("annual", {})
        if isinstance(ann, dict):
            def _to_float(x: object) -> float:
                try:
                    return float(x)  # type: ignore[arg-type]
                except Exception:
                    return float("nan")
            
            nig_v = _to_float(ann.get("nig"))
            ghst_v = _to_float(ann.get("ghst"))
            evar_v = _to_float(ann.get("evar"))
            has_vals = any(v == v for v in (nig_v, ghst_v, evar_v))
            
            if not has_vals:
                # Force recalculation if no valid values
                data = cvar_service.get_cvar_data(
                    symbol=symbol,
                    force_recalculate=True,
                    prefer_local=True
                )
                if not data.get("success"):
                    detail = data if isinstance(data, dict) else {"detail": str(data)}
                    status = 422 if data.get("code") == "insufficient_history" else 500
                    raise HTTPException(status, detail)
                
                block = data.get(key, {})
    return {
        "alpha": alpha,
        "snapshot": block["snapshot"],
        "alpha_used": block.get("alpha"),
        "as_of": data["as_of_date"],
        "symbol": symbol,
        "cached": data.get("cached", False),
    }


@router.get("/cvar/clear-cache")
def clear_cache(
    _auth: None = Depends(_require_pub_or_basic),
    symbol: str | None = Query(
        None,
        description="Optional symbol to clear only its cache",
    ),
) -> dict[str, bool]:
    # REFACTORED: Using new domain service instead of singleton calculator
    from services.domain.cvar_unified_service import CvarUnifiedService
    
    cvar_service = CvarUnifiedService()
    cvar_service.clear_cache(symbol)
    
    # Legacy cache clearing removed - now handled by unified service
    
    return {"success": True}


@router.get("/cvar/events")
async def cvar_events(
    symbol: str = Query(..., description="Symbol to wait for"),
    timeout: int = Query(120, description="Timeout in seconds"),
    _auth: None = Depends(_require_pub_or_basic),
) -> StreamingResponse:
    """SSE stream that waits until CVaR results for a symbol are available.
    Emits ping/update/timeout events.
    """
    _ = _basic_auth_if_configured
    _basic_auth_if_configured  # type: ignore

    calc = CvarUnifiedService()

    def _has_any(block: dict | None) -> bool:
        try:
            ann = block.get("annual") if isinstance(block, dict) else None
            vals: list[float] = []
            if isinstance(ann, dict):
                for k in ("nig", "ghst", "evar"):
                    try:
                        v = ann.get(k)  # type: ignore[index]
                        vals.append(float(v))  # type: ignore[arg-type]
                    except Exception:
                        continue
            vals = [v for v in vals if v == v]
            return bool(vals)
        except Exception:
            return False

    async def _gen():
        start = _time.perf_counter()
        # advise client to retry quickly on disconnects
        yield "retry: 2000\n\n"
        # initial ping
        yield "event: ping\n" "data: {}\n\n"
        while True:
            try:
                cached = calc.get_cached(symbol)
            except Exception:
                cached = None
            if isinstance(cached, dict):
                c50 = cached.get("cvar50")
                c95 = cached.get("cvar95")
                c99 = cached.get("cvar99")
                if _has_any(c95) or _has_any(c99) or _has_any(c50):
                    out = {
                        "symbol": symbol,
                        "as_of": cached.get("as_of_date"),
                        "cached": True,
                        "cvar50": {
                            "annual": (c50 or {}).get("annual"),
                            "snapshot": (c50 or {}).get("snapshot"),
                            "alpha_used": (c50 or {}).get("alpha"),
                            "as_of": cached.get("as_of_date"),
                            "cached": True,
                            "lambert": (c50 or {}).get("lambert"),
                        },
                        "cvar95": {
                            "annual": (c95 or {}).get("annual"),
                            "snapshot": (c95 or {}).get("snapshot"),
                            "alpha_used": (c95 or {}).get("alpha"),
                            "as_of": cached.get("as_of_date"),
                            "cached": True,
                            "lambert": (c95 or {}).get("lambert"),
                        },
                        "cvar99": {
                            "annual": (c99 or {}).get("annual"),
                            "snapshot": (c99 or {}).get("snapshot"),
                            "alpha_used": (c99 or {}).get("alpha"),
                            "as_of": cached.get("as_of_date"),
                            "cached": True,
                            "lambert": (c99 or {}).get("lambert"),
                        },
                    }
                    yield "event: update\n" f"data: {_json.dumps(out)}\n\n"
                    break
            # timeout check
            if (_time.perf_counter() - start) > max(1, int(timeout)):
                yield "event: timeout\n" "data: {}\n\n"
                break
            # keep-alive ping
            yield "event: ping\n" "data: {}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.get("/cvar/temp-evar-analysis")
def temp_evar_analysis(
    country: str | None = Query(
        None,
        description=(
            "Optional comma-separated country filter (e.g., 'US,Canada')"
        ),
    ),
    types: str | None = Query(
        None,
        description=(
            "Optional comma-separated instrument types "
            "(e.g., 'ETF,Mutual Fund')"
        ),
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """
    Analyze when EVaR is the largest among NIG, GHST, and EVaR.

    REFACTORED: Uses repository pattern instead of direct database access.
    Calculates the percentage of cases where EVaR > NIG AND EVaR > GHST.
    """
    from repositories import CvarRepository
    
    # Initialize repository using new architecture
    cvar_repo = CvarRepository()
    
    try:
        # Parse types parameter
        types_list = None
        if types:
            types_list = [t.strip() for t in types.split(",") if t.strip()]
        
        # Use repository method instead of direct SQL queries
        analysis_result = cvar_repo.get_evar_analysis(
            country=country,
            types=types_list
        )
        
        # Return the analysis result (repository handles all SQL complexity)
        return analysis_result
        
    except Exception as exc:
        logger.error(f"EVaR analysis failed: {exc}")
        raise HTTPException(500, f"Analysis failed: {str(exc)}")
