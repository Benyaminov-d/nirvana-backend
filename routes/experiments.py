from __future__ import annotations
# flake8: noqa
# pyright: reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportOptionalSubscript=false

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from sqlalchemy import and_, func  # type: ignore
from typing import Any, Dict

from utils.auth import (
    basic_auth_if_configured as _basic_auth_if_configured,
)
from core.db import get_db_session
from core.models import CvarSnapshot, PriceSeries
from services.compass_anchors import calibrate_special_sets
from utils.common import canonical_instrument_type as _canon_type
from core.persistence import upsert_snapshot_row
import logging as _log
import numpy as _np
from config import get_config as _get_config

_CONFIG = _get_config()


# Optional: timeseries helpers from shared lib (compute annual return from log)
try:  # pragma: no cover - optional import
    import importlib as _il  # type: ignore
    _ts_mod = _il.import_module("nirvana_risk.timeseries")
    _ts_ann1y = getattr(_ts_mod, "compute_annual_return_1y_from_log", None)
except Exception:  # noqa: BLE001
    _ts_ann1y = None  # type: ignore


router = APIRouter()


@router.get("/experiments/validate-mf-etf")
def exp_validate_mf_etf(
    limit: int = 100,
    dry_run: int = 1,
    symbol: str | None = None,
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Fetch fresh time series from EODHD for Mutual Funds & ETFs
    and run strict validator.

    - limit: cap number of symbols to process (random or top updated)
    - dry_run: 1 to avoid DB writes, 0 to persist anomaly reports and flags
    """
    import os
    import requests  # type: ignore
    from utils.common import resolve_eodhd_endpoint_symbol

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        sym = (symbol or "").strip().upper()
        if sym:
            q = sess.query(PriceSeries).filter(PriceSeries.symbol == sym)
        else:
            q = sess.query(PriceSeries).filter(
                PriceSeries.instrument_type.in_(["Mutual Fund", "ETF"])  # type: ignore
            )
        rows = q.limit(max(1, int(limit))).all()
        if sym and not rows:
            class _Tmp:
                def __init__(self, s: str):
                    self.symbol = s
                    self.instrument_type = None
            rows = [_Tmp(sym)]
        api_key = os.getenv("EODHD_API_KEY")
        if not api_key:
            raise HTTPException(501, "EODHD_API_KEY not set")

        import importlib
        validate_and_clean = None
        try:
            _mod = importlib.import_module("nirvana_risk_core.src_core")
            validate_and_clean = getattr(_mod, "validate_and_clean", None)
        except Exception:
            validate_and_clean = None  # type: ignore

        processed: list[dict[str, object]] = []
        save = int(dry_run) == 0

        # helper: anomaly detector (lib-backed when available)
        def _detect_prices(dates: list[str], prices: list[float]) -> dict | None:
            try:
                if len(prices) < 2:
                    return {"reason": "too_few_prices", "n": len(prices)}
                import importlib
                det_jump = None
                det_reg = None
                try:
                    _mod = importlib.import_module("nirvana_risk.timeseries.validation")
                    det_jump = getattr(_mod, "detect_extreme_adjacent_jump", None)
                    det_reg = getattr(_mod, "detect_regime_change_near_zero_to_large", None)
                except Exception:
                    det_jump = None
                    det_reg = None
                max_ratio = 1.0
                min_ratio = 1.0
                for i in range(1, len(prices)):
                    p0 = float(prices[i - 1])
                    p1 = float(prices[i])
                    if p0 > 0 and p1 > 0:
                        r = p1 / p0
                        if r > max_ratio:
                            max_ratio = r
                        if r < min_ratio:
                            min_ratio = r
                RATIO_MAX = float(os.getenv("NVAR_ANOM_RATIO_MAX", "100"))
                BIG_SPAN = float(os.getenv("NVAR_ANOM_BIG_SPAN", "1000"))
                near_zero_cut = float(os.getenv("NVAR_ANOM_NEAR_ZERO", "0.001"))
                flags: list[str] = []
                # Local checks only to avoid indentation/syntax issues
                if max_ratio > RATIO_MAX or (min_ratio > 0 and min_ratio < (1.0 / RATIO_MAX)):
                    flags.append("extreme_adjacent_jump")
                try:
                    pmin = float(min(prices))
                    pmax = float(max(prices))
                except Exception:
                    pmin, pmax = 0.0, 0.0
                if (pmin > 0 and pmax / max(pmin, 1e-12) > BIG_SPAN) and (pmin < near_zero_cut < pmax):
                    flags.append("regime_change_near_zero_to_large")
                if not flags:
                    return None
                rep = {
                    "flags": flags,
                    "max_ratio": max_ratio,
                    "min_ratio": min_ratio,
                    "pmin": pmin,
                    "pmax": pmax,
                    "n_prices": len(prices),
                }
                try:
                    if dates and len(dates) == len(prices):
                        rep["first_date"] = dates[0]
                        rep["last_date"] = dates[-1]
                except Exception:
                    pass
                return rep
            except Exception:
                return {"reason": "anomaly_detection_failed"}

        # Decide if we'd mark insufficient_history under current policy
        def _would_mark_insufficient(anom: object) -> tuple[bool, str]:
            if not isinstance(anom, dict):
                return (False, "no_anomalies")
            rep = anom.get("report") if isinstance(anom.get("report"), list) else []
            summ = anom.get("summary") if isinstance(anom.get("summary"), dict) else {}
            try:
                ld = summ.get("local_detector") if isinstance(summ, dict) else None
                if isinstance(ld, dict):
                    fl = ld.get("flags")
                    if isinstance(fl, list) and len(fl) > 0:
                        return (True, "local_detector_flags")
            except Exception:
                pass
            severe_types = {
                "price_discontinuity",
                "split_mismatch",
                "zero_or_negative_price",
                "negative_price",
                "duplicate_date",
                "backward_date",
            }
            if isinstance(rep, list):
                for it in rep:
                    try:
                        tp = it.get("type") if isinstance(it, dict) else None
                        if isinstance(tp, str) and tp in severe_types:
                            return (True, f"severe_report:{tp}")
                    except Exception:
                        continue
            return (False, "no_fatal_anomaly")

        # fetch/validate with parallelism to avoid long-running request
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from datetime import datetime as _dt, timedelta as _td
            years = int(os.getenv("EXP_VALIDATE_YEARS", "25"))
            workers = max(2, min(16, int(_CONFIG.exp_validate_workers)))
            since = (_dt.utcnow().date() - _td(days=int(years * 365.25))).isoformat()

            def _job(ps: PriceSeries) -> dict[str, object]:
                sym = ps.symbol
                endpoint_symbol = resolve_eodhd_endpoint_symbol(sym)
                url = f"https://eodhistoricaldata.com/api/eod/{endpoint_symbol}"
                try:
                    r = requests.get(
                        url,
                        params={"api_token": api_key, "fmt": "json", "order": "d", "from": since},
                        timeout=(5, 15),
                    )
                    r.raise_for_status()
                    raw = r.json()
                except Exception as exc:
                    return {"symbol": sym, "status": "fetch_error", "error": str(exc)}
                dates: list[str] = []
                prices: list[float] = []
                for row in raw:
                    try:
                        d = str(row.get("date"))
                        adj = row.get("adjusted_close")
                        close_v = row.get("close")
                        pv = adj if (adj is not None and float(adj) > 0) else close_v
                        fv = float(pv) if pv is not None else None
                        if fv is not None and fv > 0:
                            dates.append(d)
                            prices.append(fv)
                    except Exception:
                        continue
                anomalies = None
                if validate_and_clean is not None and len(prices) >= 2:
                    try:
                        res_v = validate_and_clean(
                            dates,
                            prices,
                            symbol=sym,
                            policy=os.getenv("NVAR_VALIDATION_POLICY", "autofix"),
                            asset_class=_canon_type(ps.instrument_type) or "",
                        )
                        if isinstance(res_v, dict):
                            anomalies = {
                                "report": res_v.get("report") if isinstance(res_v.get("report"), list) else [],
                                "summary": res_v.get("summary") if isinstance(res_v.get("summary"), dict) else {},
                                "policy": os.getenv("NVAR_VALIDATION_POLICY", "autofix"),
                                "asset_class": _canon_type(ps.instrument_type) or "",
                            }
                    except Exception:
                        anomalies = None
                det = _detect_prices(dates, prices)
                status = "ok"
                detail: dict[str, object] = {"symbol": sym, "n_prices": len(prices)}
                if anomalies or det:
                    status = "anomaly"
                    merged = {"report": [], "summary": {}, "policy": os.getenv("NVAR_VALIDATION_POLICY", "autofix"), "asset_class": _canon_type(ps.instrument_type) or ""}
                    if anomalies:
                        merged["report"] = anomalies.get("report", [])  # type: ignore
                        merged["summary"] = anomalies.get("summary", {})  # type: ignore
                    if det:
                        try:
                            rep = merged.get("report")  # type: ignore
                            if isinstance(rep, list):
                                rep.append(det)
                        except Exception:
                            pass
                        try:
                            sm = merged.get("summary")  # type: ignore
                            if isinstance(sm, dict):
                                sm.update({"local_detector": det})
                        except Exception:
                            pass
                    detail["anomalies"] = merged
                    # dry-run decision preview
                    would_mark, reason = _would_mark_insufficient(merged)
                    detail["insufficient_history_decision"] = {
                        "would_mark": bool(would_mark),
                        "reason": reason,
                    }
                    if save:
                        try:
                            from core.persistence import save_cvar_result
                            from datetime import date as _date
                            payload = {"as_of_date": (_date.today().isoformat()), "anomalies_report": merged}
                            save_cvar_result(sym, payload)
                        except Exception:
                            pass
                if status == "ok":
                    detail["insufficient_history_decision"] = {
                        "would_mark": False,
                        "reason": "no_anomalies",
                    }
                return {"symbol": sym, "status": status, **detail}

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_job, ps) for ps in rows]
                for f in as_completed(futs):
                    try:
                        processed.append(f.result())
                    except Exception as exc:
                        processed.append({"symbol": "?", "status": "error", "error": str(exc)})
        except Exception as exc:
            return JSONResponse({"error": f"execution_failed: {exc}"}, status_code=500)

        return JSONResponse({"items": processed, "count": len(processed), "dry_run": int(not save)})
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/experiments/reprocess")
def exp_reprocess_all(
    limit: int = 0,
    dry_run: int = 1,
    symbol: str | None = None,
    recheck_all: int = 0,
    recalibrate: int = 0,
    years_funds: int | None = None,
    years_others: int | None = None,
    types: str | None = Query(
        None,
        description=(
            "Optional comma-separated instrument types (e.g., 'Mutual Fund,ETF')"
        ),
    ),
    workers: int | None = Query(
        None,
        description=(
            "Number of parallel workers (set 1 to disable parallelism)"
        ),
    ),
    symbols: str | None = Query(
        None,
        description=(
            "Optional comma-separated list of symbols to restrict the run"
        ),
    ),
    verbose: int = Query(
        0,
        description=(
            "When 1, emit detailed decision logs and include summary in output"
        ),
    ),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """Revalidate, recompute annual_return and CVaR(50/95/99), update DB; recalibrate anchors.

    - Processes all products with insufficient_history NULL or 0. If symbol provided, only that symbol.
    - In dry_run=0: return 200 immediately and run in background with logs.
    """
    import threading as _th
    import os as _os
    from datetime import datetime as _dt
    from core.db import get_db_session as _get_sess
    from core.models import PriceSeries as _PS

    logger = _log.getLogger("experiments")
    if not logger.handlers:
        _h = _log.StreamHandler()
        _h.setFormatter(_log.Formatter("%(asctime)s experiments %(levelname)s: %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(_log.INFO)

    sym = (symbol or "").strip().upper()

    def _detect_prices(dates: list[str], prices: list[float]) -> dict | None:
        try:
            if len(prices) < 2:
                return {"reason": "too_few_prices", "n": len(prices)}
            import importlib
            det_jump = None
            det_reg = None
            try:
                _mod = importlib.import_module("nirvana_risk.timeseries.validation")
                det_jump = getattr(_mod, "detect_extreme_adjacent_jump", None)
                det_reg = getattr(_mod, "detect_regime_change_near_zero_to_large", None)
            except Exception:
                det_jump = None
                det_reg = None
            max_ratio = 1.0
            min_ratio = 1.0
            for i in range(1, len(prices)):
                p0 = float(prices[i - 1])
                p1 = float(prices[i])
                if p0 > 0 and p1 > 0:
                    r = p1 / p0
                    if r > max_ratio:
                        max_ratio = r
                    if r < min_ratio:
                        min_ratio = r
            RATIO_MAX = float(_os.getenv("NVAR_ANOM_RATIO_MAX", "100"))
            BIG_SPAN = float(_os.getenv("NVAR_ANOM_BIG_SPAN", "1000"))
            near_zero_cut = float(_os.getenv("NVAR_ANOM_NEAR_ZERO", "0.001"))
            flags: list[str] = []
            if callable(det_jump):
                try:
                    j = det_jump(type("PS", (), {"prices": __import__("numpy").array(prices)})(), ratio_cap=RATIO_MAX)  # type: ignore[misc]
                    if bool(j):
                        flags.append("extreme_adjacent_jump")
                except Exception:
                    pass
            if max_ratio > RATIO_MAX or (min_ratio > 0 and min_ratio < (1.0 / RATIO_MAX)):
                flags.append("extreme_adjacent_jump")
            try:
                pmin = float(min(prices))
                pmax = float(max(prices))
            except Exception:
                pmin, pmax = 0.0, 0.0
            if callable(det_reg):
                try:
                    r = det_reg(
                        type("PS", (), {"prices": __import__("numpy").array(prices)})(),
                        eps=near_zero_cut,
                        ratio_cap=BIG_SPAN,
                    )  # type: ignore[misc]
                    if bool(r):
                        flags.append("regime_change_near_zero_to_large")
                except Exception:
                    pass
            if (pmin > 0 and pmax / max(pmin, 1e-12) > BIG_SPAN) and (pmin < near_zero_cut < pmax):
                flags.append("regime_change_near_zero_to_large")
            if not flags:
                return None
            rep = {
                "flags": flags,
                "max_ratio": max_ratio,
                "min_ratio": min_ratio,
                "pmin": pmin,
                "pmax": pmax,
                "n_prices": len(prices),
            }
            if dates and len(dates) == len(prices):
                rep["first_date"] = dates[0]
                rep["last_date"] = dates[-1]
            return rep
        except Exception:
            return {"reason": "anomaly_detection_failed"}

    def _do_work() -> dict:
        # Helpers for safe typing/casts in summaries/counters
        def _to_int(val: object, default: int = 0) -> int:
            try:
                if isinstance(val, bool):
                    return int(val)
                if isinstance(val, (int, float)):
                    return int(val)
                if isinstance(val, str) and val.strip() != "":
                    return int(float(val))
            except Exception:
                pass
            return default
        sess = _get_sess()
        if sess is None:
            logger.error("DB not configured")
            return {"error": "db_not_configured"}
        try:
            # Explicit symbol filters take precedence
            if symbols:
                kept = [s.strip().upper() for s in str(symbols).split(",") if s.strip()]
                kept = list(dict.fromkeys(kept))
                q = sess.query(_PS).filter(_PS.symbol.in_(kept))  # type: ignore
            elif sym:
                # When a specific symbol is requested, process it regardless of insufficient_history flag
                q = sess.query(_PS).filter(_PS.symbol == sym)
            else:
                # Otherwise, by default process only those not marked insufficient;
                # if recheck_all=1, process the entire universe to allow resets.
                if int(recheck_all) != 0:
                    q = sess.query(_PS)
                else:
                    q = sess.query(_PS).filter((_PS.insufficient_history.is_(None)) | (_PS.insufficient_history == 0))  # type: ignore
            # Optional type filter (e.g., Mutual Fund, ETF)
            if types:
                try:
                    allowed = [
                        t.strip() for t in str(types).split(",") if t.strip()
                    ]
                except Exception:
                    allowed = []
                if allowed:
                    q = q.filter(_PS.instrument_type.in_(allowed))  # type: ignore
            rows = q.all() if int(limit) <= 0 else q.limit(max(1, int(limit))).all()
            total = len(rows)
            logger.info("reprocess: symbols=%d dry_run=%d", total, int(dry_run))
            ok = 0
            bad = 0
            updated = 0
            details: list[dict[str, object]] = []
            # Aggregated dry-run summary
            marked = 0
            reasons: dict[str, int] = {}
            load_errors = 0
            skipped = 0
            reason_details: dict[str, dict[str, object]] = {}
            dropped_points_recent_total = 0
            dropped_points_recent_with_flag = 0
            # Resolve horizons once
            try:
                yrs_fun = int(years_funds) if years_funds is not None else int(_os.getenv("EXP_REPROCESS_YEARS_FUNDS", "0"))
            except Exception:
                yrs_fun = 0
            try:
                yrs_oth = int(years_others) if years_others is not None else int(_os.getenv("EXP_REPROCESS_YEARS_OTHERS", "0"))
            except Exception:
                yrs_oth = 0
            
            def _anomaly_present(an: object) -> bool:
                if not isinstance(an, dict):
                    return False
                rep = an.get("anomalies_report") if "anomalies_report" in an else an
                # rep may already be the inner dict
                if not isinstance(rep, dict):
                    return False
                report_list = rep.get("report")
                summary_obj = rep.get("summary")

                # Rule: Fatal only if recent window is bad OR any year breaches threshold OR hard anomalies present.
                # 1) Recent window rule: dropped_points_last_252 > limit -> fatal
                try:
                    if isinstance(summary_obj, dict):
                        dl = summary_obj.get("dropped_points_last_252")
                        if dl is not None and int(dl) > int(_os.getenv("NVAR_MAX_DROP_LAST252", "10")):
                            return True
                except Exception:
                    pass

                # 1b) Per-year rule: any year with dropped_points_by_year > limit -> fatal
                try:
                    if isinstance(summary_obj, dict):
                        by_year = summary_obj.get("dropped_points_by_year")
                        limit = int(_os.getenv("NVAR_MAX_DROP_LAST252", "10"))
                        if isinstance(by_year, dict):
                            for _, cnt in by_year.items():
                                try:
                                    if int(cnt) > limit:
                                        return True
                                except Exception:
                                    continue
                except Exception:
                    pass

                # 2) Hard anomalies which are always fatal
                fatal_types = {
                    "zero_or_negative_price",
                    "negative_price",
                    "backward_date",
                }

                # (intentionally ignore generic severity="fatal"; use our hard rules only)

                # 3) Specific fatal anomaly types (hard set only)
                if isinstance(report_list, list):
                    for it in report_list:
                        if not isinstance(it, dict):
                            continue
                        tp = it.get("type")
                        if isinstance(tp, str) and tp in fatal_types:
                            return True

                # 4) Local detector flags and long-horizon oddities are NOT fatal by themselves
                # We intentionally ignore generic flags in summary/local_detector here.
                _ = summary_obj  # keep for possible future use
                return False

            # Build job list and run in parallel (deduplicate by symbol)
            items: list[tuple[int, str, str]] = []
            _seen: set[str] = set()
            idx_counter = 0
            for ps in rows:
                s = str(ps.symbol)
                if s in _seen:
                    continue
                _seen.add(s)
                idx_counter += 1
                items.append((idx_counter, s, (_canon_type(ps.instrument_type) or "")))
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _env_workers = int(_CONFIG.exp_reprocess_workers)
            _req_workers = int(workers) if workers is not None else _env_workers
            _req_workers = 1 if _req_workers <= 1 else _req_workers
            pool_workers = max(2, min(16, _req_workers))
            stop_event = _th.Event()

            def _job(idx0: int, symb: str, itype: str) -> dict[str, object]:
                # verbose logger helper
                def _v(msg: str, *args: object) -> None:
                    try:
                        if int(verbose) != 0:
                            logger.info(msg, *args)
                    except Exception:
                        pass
                if stop_event.is_set():
                    return {"ok": 0, "bad": 0, "updated": 0, "symbol": symb, "skipped": 1}
                _sess = _get_sess()
                if _sess is None:
                    return {"ok": 0, "bad": 1, "updated": 0, "symbol": symb}
                try:
                    logger.info("[%d/%d] %s: load prices", idx0, total, symb)
                    # Full horizon for funds/ETFs (we compute CVaR); limited window for others
                    is_fund = itype in ("Mutual Fund", "ETF")
                    # Reprocess focuses on 1y return; relax strict source/years constraints for loader
                    try:
                        _os.environ["NVAR_ALLOW_CLOSE_FALLBACK"] = "1"
                        _os.environ["NVAR_MIN_YEARS"] = "1"
                    except Exception:
                        pass
                    # CVaR is not recomputed in this endpoint, so no need to fetch full history.
                    # To speed up, restrict horizon to recent years for all types.
                    from services.prices import load_prices as _load_prices
                    if is_fund:
                        res = _load_prices(
                            symb,
                            to_date=None,
                            from_years=(int(yrs_fun) if int(yrs_fun) > 0 else None),
                        )
                    else:
                        res = _load_prices(
                            symb,
                            to_date=None,
                            from_years=(int(yrs_oth) if int(yrs_oth) > 0 else None),
                        )
                    # basic metrics right after load
                    try:
                        _r = res.get("returns")
                        if _r is None:
                            _r = res.get("returns_log")
                        _arr = _np.asarray(_r if _r is not None else [], float)
                        _v("loaded: %s n_returns=%d years=%.2f as_of=%s", symb, int(_arr.size), float(_arr.size) / 252.0, str(res.get("as_of_date")))
                    except Exception:
                        pass
                except Exception as exc:
                    logger.warning("%s: load failed: %s", symb, exc)
                    try:
                        if "exceeded your daily api requests limit" in str(exc).lower():
                            stop_event.set()
                    except Exception:
                        pass
                    try:
                        _sess.close()
                    except Exception:
                        pass
                    return {"ok": 0, "bad": 1, "updated": 0, "symbol": symb, "summary": {"error": f"load_failed: {exc}"}, "insufficient_history_decision": {"would_mark": False, "reason": "load_failed"}}
                if not isinstance(res, dict):
                    logger.warning("%s: load not successful (bad shape)", symb)
                    try:
                        _sess.close()
                    except Exception:
                        pass
                    return {"ok": 0, "bad": 1, "updated": 0, "symbol": symb, "summary": {"error": "bad_loader_shape"}, "insufficient_history_decision": {"would_mark": False, "reason": "load_error"}}
                if not bool(res.get("success", True)):
                    err_s = str(res.get("error") or "")
                    if err_s:
                        logger.warning("%s: loader error: %s", symb, err_s)
                        _v("loader_error_detail: %s -> %s", symb, err_s)
                    # Special case: too few returns – still proceed to compute 1y return if possible
                    if err_s.strip().lower().startswith("too few returns"):
                        # continue to compute annual return from whatever returns present
                        pass
                    else:
                        try:
                            if "exceeded your daily api requests limit" in err_s.lower():
                                stop_event.set()
                        except Exception:
                            pass
                        try:
                            _sess.close()
                        except Exception:
                            pass
                        _reason = "quota_exceeded" if "exceeded your daily api requests limit" in err_s.lower() else "load_error"
                        return {"ok": 0, "bad": 1, "updated": 0, "symbol": symb, "skipped": 1, "summary": {"error": err_s or "loader_error"}, "insufficient_history_decision": {"would_mark": False, "reason": _reason}}
                anom = res.get("anomalies_report")
                # Persist anomaly report best-effort (so anomaly table is not empty)
                if int(dry_run) == 0 and isinstance(anom, dict):
                    try:
                        from core.persistence import save_cvar_result as _save
                        _save(symb, {"as_of_date": res.get("as_of_date"), "anomalies_report": anom})
                    except Exception:
                        pass
                try:
                    _r = res.get("returns")
                    if _r is None:
                        _r = res.get("returns_log")
                    rets_log = _np.asarray(_r if _r is not None else [], float)
                    simple = _np.expm1(rets_log)
                except Exception:
                    simple = _np.asarray([], float)
                    rets_log = _np.asarray([], float)
                mark_insufficient_due_to_few = False
                annual_return: float | None = None
                if simple.size < 10:
                    # Do not abort: proceed with a safe fallback (clear annual_return)
                    # to overwrite any stale/corrupted values in DB. Also mark
                    # insufficient_history=1 so such instruments are excluded downstream.
                    logger.warning("%s: too few returns (%d) -> fallback with return_annual=None and mark insufficient_history", symb, int(simple.size))
                    annual_return = None  # type: ignore
                    mark_insufficient_due_to_few = True
                # Dropped-points enforcement is disabled during reprocess since CVaR is not recomputed here.
                # Keep this block behind an opt-in flag if needed in future.
                # if bool(int(_os.getenv("EXP_ENFORCE_DROP_LAST252", "0"))):
                #     try:
                #         summ_obj = (anom or {}).get("summary") if isinstance(anom, dict) else None
                #         dropped_last = None
                #         if isinstance(summ_obj, dict):
                #             dropped_last = summ_obj.get("dropped_points_last_252")
                #         if dropped_last is not None and int(dropped_last) > int(_os.getenv("NVAR_MAX_DROP_LAST252", "10")):
                #             logger.warning("%s: too many dropped points in last 252 (%s) -> mark insufficient_history", symb, int(dropped_last))
                #             if int(dry_run) == 0:
                #                 try:
                #                     _ps = _sess.query(_PS).filter(_PS.symbol == symb).one_or_none()
                #                     if _ps is not None:
                #                         _ps.insufficient_history = 1
                #                         _sess.merge(_ps)
                #                         _sess.commit()
                #                 except Exception:
                #                     _sess.rollback()
                #             try:
                #                 _sess.close()
                #             except Exception:
                #                 pass
                #             return {"ok": 0, "bad": 1, "updated": 0}
                #     except Exception:
                #         pass
                if _anomaly_present(anom):
                    logger.warning("%s: anomalies present -> mark insufficient_history", symb)
                    _v("decision: %s -> insufficient_history (fatal_anomaly)", symb)
                    if int(dry_run) == 0:
                        try:
                            _ps = _sess.query(_PS).filter(_PS.symbol == symb).one_or_none()
                            if _ps is not None:
                                _ps.insufficient_history = 1
                                # also track dropped points diagnostics on the series
                                try:
                                    summ_obj = (anom or {}).get("summary") if isinstance(anom, dict) else None
                                    dp = None
                                    if isinstance(summ_obj, dict):
                                        dp = summ_obj.get("dropped_points_last_252")
                                    if dp is not None:
                                        _ps.dropped_points_recent = int(dp)
                                        _ps.has_dropped_points_recent = 1 if int(dp) > 0 else 0
                                except Exception:
                                    pass
                                _sess.merge(_ps)
                                _sess.commit()
                        except Exception:
                            _sess.rollback()
                    try:
                        _sess.close()
                    except Exception:
                        pass
                    # Prefer anomalies summary if available; fallback to loader summary
                    fallback_summary = res.get("summary") if isinstance(res, dict) else None
                    # enrich decision with dropped-points
                    try:
                        summ_obj = anom.get("summary") if isinstance(anom, dict) else None
                        dl = summ_obj.get("dropped_points_last_252") if isinstance(summ_obj, dict) else None
                        dl_i = int(dl) if dl is not None else None
                        has_dl = (dl_i is not None and dl_i > 0)
                    except Exception:
                        dl_i = None
                        has_dl = False
                    # Attach diagnostics from loader summary if available
                    diag_last = None
                    diag_cov = None
                    diag_liq = None
                    try:
                        if isinstance(fallback_summary, dict):
                            diag_last = fallback_summary.get("last_252")
                            diag_cov = fallback_summary.get("coverage_by_year")
                            diag_liq = fallback_summary.get("liquidity_decision")
                    except Exception:
                        pass
                    return {
                        "ok": 0,
                        "bad": 1,
                        "updated": 0,
                        "symbol": symb,
                        "summary": (
                            anom.get("summary") if isinstance(anom, dict) else fallback_summary
                        ),
                        "insufficient_history_decision": {
                            "would_mark": True,
                            "reason": "fatal_anomaly",
                            "dropped_points_recent": dl_i,
                            "has_dropped_points_recent": bool(has_dl),
                        },
                        "last_252": diag_last,
                        "coverage_by_year": diag_cov,
                        "per_year": diag_cov,
                        "liquidity_decision": diag_liq,
                        "as_of_date": res.get("as_of_date"),
                        "as_of": res.get("as_of_date"),
                        "decision": {
                            "would_mark": True,
                            "reason": "fatal_anomaly",
                            "dropped_points_recent": dl_i,
                            "has_dropped_points_recent": bool(has_dl),
                        },
                    }
                # Log last-252 window size for diagnostics
                tail = simple[-252:] if simple.size > 252 else simple
                _v("metrics_last252: %s n=%d", symb, int(tail.size))

                # Low-liquidity decision: prefer loader's liquidity_decision
                s_obj = res.get("summary") if isinstance(res, dict) else None
                liq = s_obj.get("liquidity_decision") if isinstance(s_obj, dict) else None
                try:
                    # Temporarily disable low-liquidity decision handling here (handled upstream)
                    pass
                except Exception:
                    pass
                # Annual return for last 252 days (may be None if tail empty)
                try:
                    if _ts_ann1y is not None:
                        tail_log = (
                            rets_log[-252:] if rets_log.size > 252 else rets_log
                        )
                        annual_return = _ts_ann1y(tail_log)  # type: ignore[misc]
                    else:
                        _acc_prod = float(_np.prod(1.0 + tail))
                        annual_return = float(_acc_prod - 1.0)
                except Exception:
                    annual_return = None
                _v("annual_return: %s value=%s", symb, str(annual_return))
                try:
                    # return_as_of is last daily simple return; may be None
                    r_last = simple[-1] if simple.size > 0 else None
                    return_as_of = (
                        float(r_last) if isinstance(r_last, (int, float)) else None
                    )
                except Exception:
                    return_as_of = None
                # CVaR recomputation disabled for reprocess: update returns only
                blocks: dict[str, object] = {}
                if int(dry_run) == 0:
                    from core.persistence import _coerce_date as _dtc
                    from core.models import CvarSnapshot as _Snap
                    as_of_str = str(res.get("as_of_date"))
                    as_of_date = _dtc(as_of_str)
                    try:
                        existing_latest = (
                            _sess.query(func.max(_Snap.as_of_date))
                            .filter(_Snap.symbol == symb)
                            .scalar()
                        )
                    except Exception:
                        existing_latest = None
                    if existing_latest is not None and as_of_date is not None and existing_latest > as_of_date:
                        logger.info("%s: upsert using existing latest as_of=%s instead of computed %s", symb, existing_latest, as_of_date)
                        as_of_date = existing_latest
                    if isinstance(blocks, dict) and blocks:
                        for label in (50, 95, 99):
                            blk = blocks.get(f"cvar{label}") if isinstance(blocks, dict) else None
                            ann: Dict[str, Any]
                            if isinstance(blk, dict):
                                _annual_raw = blk.get("annual")
                                ann = _annual_raw if isinstance(_annual_raw, dict) else {}
                            else:
                                ann = {}
                            alpha_conf_val = None
                            if isinstance(blk, dict):
                                try:
                                    _araw = blk.get("alpha")
                                    if isinstance(_araw, (int, float, str)):
                                        alpha_conf_val = float(_araw)
                                except Exception:
                                    alpha_conf_val = None
                            try:
                                upsert_snapshot_row(
                                    symbol=symb,
                                    as_of_date=as_of_date or _dt.utcnow().date(),
                                    alpha_label=int(label),
                                    alpha_conf=alpha_conf_val,
                                    years=1,
                                    cvar_nig=ann.get("nig"),
                                    cvar_ghst=ann.get("ghst"),
                                    cvar_evar=ann.get("evar"),
                                    source="reprocess_local",
                                    start_date=res.get("start_date"),
                                    return_as_of=return_as_of,
                                    return_annual=annual_return,
                                    instrument_id=None,
                                    session=_sess,
                                )
                            except Exception:
                                _sess.rollback()
                                try:
                                    _sess.close()
                                except Exception:
                                    pass
                                return {"ok": 0, "bad": 1, "updated": 0}
                        try:
                            _ps = _sess.query(_PS).filter(_PS.symbol == symb).one_or_none()
                            if _ps is not None:
                                _ps.insufficient_history = 0
                                # persist dropped points diagnostics even for valid rows
                                try:
                                    summ_obj = (anom or {}).get("summary") if isinstance(anom, dict) else None
                                    dp = None
                                    if isinstance(summ_obj, dict):
                                        dp = summ_obj.get("dropped_points_last_252")
                                    if dp is not None:
                                        _ps.dropped_points_recent = int(dp)
                                        _ps.has_dropped_points_recent = 1 if int(dp) > 0 else 0
                                    else:
                                        _ps.dropped_points_recent = None
                                        _ps.has_dropped_points_recent = 0
                                except Exception:
                                    pass
                                _sess.merge(_ps)
                            _sess.commit()
                        except Exception:
                            _sess.rollback()
                            try:
                                _sess.close()
                            except Exception:
                                pass
                            return {"ok": 0, "bad": 1, "updated": 0}
                        try:
                            _sess.close()
                        except Exception:
                            pass
                        return {"ok": 1, "bad": 0, "updated": 1}
                    else:
                        # Non-funds: update only returns (overwrite latest snapshot when exact as_of missing)
                        try:
                            from sqlalchemy import text as _sqltext  # type: ignore
                            for label in (50, 95, 99):
                                res1 = _sess.execute(
                                    _sqltext(
                                        """
                                        UPDATE cvar_snapshot
                                        SET return_as_of = :ra,
                                            return_annual = :rann,
                                            updated_at = now()
                                        WHERE symbol = :sym
                                          AND as_of_date = :asof
                                          AND alpha_label = :lbl
                                        """
                                    ),
                                    {
                                        "ra": return_as_of,
                                        "rann": annual_return,
                                        "sym": symb,
                                        "asof": (as_of_date or _dt.utcnow().date()),
                                        "lbl": int(label),
                                    },
                                )
                                # If exact-date row missing, update latest snapshot for the label
                                if int(getattr(res1, "rowcount", 0) or 0) == 0:
                                    _sess.execute(
                                        _sqltext(
                                            """
                                            UPDATE cvar_snapshot
                                            SET return_as_of = :ra,
                                                return_annual = :rann,
                                                updated_at = now()
                                            WHERE symbol = :sym
                                              AND alpha_label = :lbl
                                              AND as_of_date = (
                                                SELECT max(as_of_date) FROM cvar_snapshot WHERE symbol = :sym AND alpha_label = :lbl
                                              )
                                            """
                                        ),
                                        {
                                            "ra": return_as_of,
                                            "rann": annual_return,
                                            "sym": symb,
                                            "lbl": int(label),
                                        },
                                    )
                            _sess.commit()
                        except Exception:
                            _sess.rollback()
                            try:
                                _sess.close()
                            except Exception:
                                pass
                            return {"ok": 0, "bad": 1, "updated": 0}
                        # Update insufficient_history flag and ValidationFlags
                        try:
                            from services.validation_integration import process_ticker_validation
                            
                            _ps = _sess.query(_PS).filter(_PS.symbol == symb).one_or_none()
                            if _ps is not None:
                                # Process detailed ValidationFlags using the load_prices result
                                try:
                                    validation_data = res if res else {"success": False, "code": "load_error"}
                                    
                                    process_ticker_validation(
                                        symbol=symb,
                                        validation_data=validation_data,
                                        country=_ps.country,
                                        db_session=_sess
                                    )
                                    
                                    # The validation integration handles insufficient_history sync
                                    _v("Processed ValidationFlags for %s", symb)
                                    
                                except Exception as e:
                                    # Fallback to basic insufficient_history flag
                                    logger.warning(f"ValidationFlags processing failed for {symb}, using fallback: {e}")
                                    _ps.insufficient_history = 1 if mark_insufficient_due_to_few else 0
                                    _sess.merge(_ps)
                                    
                            _sess.commit()
                        except Exception:
                            _sess.rollback()
                            try:
                                _sess.close()
                            except Exception:
                                pass
                            return {"ok": 0, "bad": 1, "updated": 0}
                        try:
                            _sess.close()
                        except Exception:
                            pass
                        return {"ok": 1, "bad": 0, "updated": 1}
                else:
                    # dry-run: include summary and decision preview
                    try:
                        summ_obj = anom.get("summary") if isinstance(anom, dict) else (res.get("summary") if isinstance(res, dict) else None)
                        dl = None
                        if isinstance(summ_obj, dict):
                            dl = summ_obj.get("dropped_points_last_252")
                        try:
                            limit = int(_os.getenv("NVAR_MAX_DROP_LAST252", "10"))
                        except Exception:
                            limit = 10
                        would = False
                        reason = "ok"
                        if dl is not None and int(dl) > limit:
                            would = True
                            reason = "dropped_last_252"
                        elif mark_insufficient_due_to_few:
                            would = True
                            reason = "too_few_returns"
                        # Ensure annual_return is present in summary for dry_run preview
                        try:
                            if isinstance(summ_obj, dict):
                                ann_blk = summ_obj.get("annual_return")
                                if not isinstance(ann_blk, dict):
                                    ann_after: float | None
                                    try:
                                        if _ts_ann1y is not None:
                                            tail_log = (
                                                rets_log[-252:]
                                                if rets_log.size > 252
                                                else rets_log
                                            )
                                            ann_after = _ts_ann1y(tail_log)  # type: ignore[misc]
                                        else:
                                            tail = (
                                                simple[-252:]
                                                if simple.size > 252
                                                else simple
                                            )
                                            if tail.size > 0:
                                                prod = float(_np.prod(1.0 + tail))
                                                ann_after = float(prod - 1.0)
                                            else:
                                                ann_after = None
                                    except Exception:
                                        ann_after = None
                                    summ_obj["annual_return"] = {
                                        "before_fix": None,
                                        "after_fix": ann_after,
                                    }
                        except Exception:
                            pass
                        dl_i = None
                        try:
                            if isinstance(dl, (int, float, str)):
                                dl_i = int(float(dl))
                        except Exception:
                            dl_i = None
                        has_dl = bool(dl_i is not None and dl_i > 0)
                    except Exception:
                        summ_obj = None
                        would = False
                        reason = "ok"
                        dl_i = None
                        has_dl = False
                    try:
                        _sess.close()
                    except Exception:
                        pass
                    # surface diagnostics from summary for dry-run convenience
                    _last3 = None
                    _cov3 = None
                    _liq3 = None
                    try:
                        if isinstance(summ_obj, dict):
                            _last3 = summ_obj.get("last_252")
                            _cov3 = summ_obj.get("coverage_by_year")
                            _liq3 = summ_obj.get("liquidity_decision")
                    except Exception:
                        pass
                    return {
                        "ok": 1,
                        "bad": 0,
                        "updated": 0,
                        "symbol": symb,
                        "summary": summ_obj,
                        "insufficient_history_decision": {
                            "would_mark": bool(would),
                            "reason": reason,
                            "dropped_points_recent": dl_i,
                            "has_dropped_points_recent": bool(has_dl),
                        },
                        "last_252": _last3,
                        "coverage_by_year": _cov3,
                        "per_year": _cov3,
                        "liquidity_decision": _liq3,
                        "as_of_date": res.get("as_of_date"),
                        "as_of": res.get("as_of_date"),
                        "decision": {
                            "would_mark": bool(would),
                            "reason": reason,
                            "dropped_points_recent": dl_i,
                            "has_dropped_points_recent": bool(has_dl),
                        },
                    }

            if _req_workers <= 1:
                # Sequential path for full, ordered logs
                for i, s, t in items:
                    try:
                        res_j = _job(i, s, t)
                        ok += _to_int(res_j.get("ok"))
                        bad += _to_int(res_j.get("bad"))
                        updated += _to_int(res_j.get("updated"))
                        if int(dry_run) != 0 and isinstance(res_j, dict):
                            sym_out = res_j.get("symbol") or s
                            ihd = res_j.get("insufficient_history_decision")
                            if not isinstance(ihd, dict):
                                ihd = {"would_mark": False, "reason": "none"}
                            details.append(
                                {
                                    "symbol": sym_out,
                                "summary": res_j.get("summary"),
                                    "insufficient_history_decision": ihd,
                                    "last_252": res_j.get("last_252"),
                                    "coverage_by_year": res_j.get(
                                        "coverage_by_year"
                                    ),
                                    "per_year": (res_j.get("per_year") if isinstance(res_j, dict) else None) or res_j.get("coverage_by_year"),
                                    "liquidity_decision": res_j.get(
                                        "liquidity_decision"
                                    ),
                                    "as_of_date": res_j.get("as_of_date"),
                                    "as_of": res_j.get("as_of_date"),
                                    "decision": ihd,
                                }
                            )
                            # Handle mark counts and loader error tracking
                            try:
                                if bool(ihd.get("would_mark")):
                                    marked += 1
                                    rs = str(ihd.get("reason") or "unknown")
                                    reasons[rs] = reasons.get(rs, 0) + 1
                                # Track loader errors in summary
                                _summ = res_j.get("summary")
                                if isinstance(_summ, dict):
                                    err_s = _summ.get("error")
                                    if isinstance(err_s, str) and err_s:
                                        load_errors += 1
                            except Exception:
                                pass
                    except Exception as exc:
                        try:
                            logger.exception("%s: job failed", s)
                        except Exception:
                            pass
                        bad += 1
                        skipped += 1
                        if int(dry_run) != 0:
                            details.append({
                                "symbol": s,
                                "summary": {"error": str(exc)},
                                "insufficient_history_decision": {"would_mark": False, "reason": "none"},
                            })
            else:
                # Parallel branch disabled due to stability; fallback to sequential processing
                for i, s, t in items:
                    try:
                        res_j = _job(i, s, t)
                        ok += _to_int(res_j.get("ok"))
                        bad += _to_int(res_j.get("bad"))
                        updated += _to_int(res_j.get("updated"))
                        if int(dry_run) != 0 and isinstance(res_j, dict):
                            sym_out = res_j.get("symbol")
                            ihd = res_j.get("insufficient_history_decision")
                            if not isinstance(ihd, dict):
                                ihd = {"would_mark": False, "reason": "none"}
                            details.append(
                                {
                                    "symbol": sym_out,
                                    "summary": res_j.get("summary"),
                                    "insufficient_history_decision": ihd,
                                    "last_252": res_j.get("last_252"),
                                    "coverage_by_year": res_j.get("coverage_by_year"),
                                    "per_year": (res_j.get("per_year") if isinstance(res_j, dict) else None) or res_j.get("coverage_by_year"),
                                    "liquidity_decision": res_j.get("liquidity_decision"),
                                    "as_of_date": res_j.get("as_of_date"),
                                    "as_of": res_j.get("as_of_date"),
                                    "decision": ihd,
                                }
                            )
                    except Exception as exc:
                        bad += 1
                        skipped += 1
                        if int(dry_run) != 0:
                            details.append({
                                "symbol": None,
                                "summary": {"error": str(exc)},
                                "insufficient_history_decision": {"would_mark": False, "reason": "none"},
                            })
            # Recalibrate anchors after updates (optional)
            if int(recalibrate) != 0:
                try:
                    from services.compass_anchors import (
                        auto_calibrate_global_per_country_from_db as _cal_pc,
                        auto_calibrate_by_type_country_from_db as _cal_tc,
                        calibrate_special_sets as _cal_exp,
                    )
                    _r1 = _cal_pc()
                    logger.info("recalibrate per-country: %s", _r1)
                    _r2 = _cal_tc()
                    logger.info("recalibrate by-type-country: %s", _r2)
                    _r3 = _cal_exp()
                    logger.info("recalibrate experiment anchors: %s", _r3)
                except Exception as exc:
                    logger.warning("recalibration failed: %s", exc)
            out: dict[str, object] = {"total": total, "ok": ok, "bad": bad, "snapshots_updated": updated}
            if int(dry_run) != 0:
                out["items"] = list(details)
                _reason_texts = {
                    "fatal_anomaly": (
                        "Anomalies present (e.g., backward date, negative price)"
                    ),
                    "low_liquidity": (
                        "Loader liquidity decision flagged low or insufficient"
                    ),
                    "low_liquidity_fallback": (
                        "Few observations or high zero-share in last 252 days"
                    ),
                    "low_liquidity_unique_prices": (
                        "Too few unique prices in last 252 days"
                    ),
                    "low_liquidity_plateau_share": (
                        "Plateau share too high in last 252 days"
                    ),
                    "low_liquidity_min_trading_days": (
                        "Too few trading days in some recent months"
                    ),
                    "dropped_last_252": (
                        "Too many dropped points in last 252 days"
                    ),
                    "too_few_returns": "Too few returns to compute annual return",
                    "load_error": "Loader error",
                    "quota_exceeded": "Upstream API quota reached",
                    "none": "No issues",
                    "ok": "No issues",
                }
                out["summary"] = {
                    "marked_insufficient": marked,
                    "reasons": reasons,
                    "loader_errors": load_errors,
                    "skipped": skipped,
                    "reason_details": reason_details,
                    "dropped_points_recent_total": dropped_points_recent_total,
                    "dropped_points_recent_with_flag": dropped_points_recent_with_flag,
                    "reason_texts": _reason_texts,
                }
                if int(verbose) != 0:
                    try:
                        logger.info(
                            "dry-run summary: marked=%d loader_errors=%d reasons=%s",
                            int(marked),
                            int(load_errors),
                            reasons,
                        )
                    except Exception:
                        pass
            return out
        finally:
            try:
                sess.close()
            except Exception:
                pass

    if int(dry_run) == 0:
        # fire-and-forget background
        _th.Thread(target=_do_work, daemon=True).start()
        return JSONResponse({"started": True})
    else:
        res = _do_work()
        return JSONResponse(res)


@router.post("/experiments/anchors/calibrate")
def exp_calibrate_anchors(
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    summary = calibrate_special_sets()
    return JSONResponse({"ok": True, "summary": summary})


def _latest_snapshots(sess) -> Any:
    latest = (
        sess.query(
            CvarSnapshot.symbol.label("symbol"),
            func.max(CvarSnapshot.as_of_date).label("mx"),
        )
        .group_by(CvarSnapshot.symbol)
        .subquery()
    )
    q = (
        sess.query(CvarSnapshot, PriceSeries)
        .join(
            latest,
            and_(
                CvarSnapshot.symbol == latest.c.symbol,
                CvarSnapshot.as_of_date == latest.c.mx,
            ),
        )
        .outerjoin(PriceSeries, PriceSeries.symbol == CvarSnapshot.symbol)
        .filter(PriceSeries.valid == 1)  # Only use valid products
    )
    return q


@router.get("/experiments/products/take")
def exp_take_products(
    n: int = 20,
    category: str | None = None,
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        q = _latest_snapshots(sess)
        if category == "ALL-MUTUAL-FUND-ETF":
            q = q.filter(PriceSeries.instrument_type.in_(["Mutual Fund", "ETF"]))  # type: ignore
        if category == "US-MUTUAL-FUND-ETF":
            q = q.filter(PriceSeries.country == "US")  # type: ignore
            q = q.filter(PriceSeries.instrument_type.in_(["Mutual Fund", "ETF"]))  # type: ignore
        rows = q.limit(max(1, int(n))).all()
        items = []
        for r, ps in rows:
            items.append({
                "symbol": r.symbol,
                "name": getattr(ps, "name", None),
                "country": getattr(ps, "country", None),
                "type": getattr(ps, "instrument_type", None),
            })
        return JSONResponse({"items": items, "count": len(items)})
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.post("/experiments/score")
def exp_score(
    payload: dict[str, Any],
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    symbols = [str(s).strip() for s in payload.get("symbols", []) if str(s).strip()]
    anchor = str(payload.get("anchor", "GLOBAL:ALL")).strip() or "GLOBAL:ALL"
    lt_val = payload.get("lt")
    # Use configuration for default LT
    # Global config removed - using safe defaults only
    default_loss_tolerance = 0.25  # 25% - moderate risk tolerance
    
    try:
        LT = float(lt_val) if lt_val is not None else default_loss_tolerance
        if not (LT > 0):
            raise ValueError
    except Exception:
        LT = default_loss_tolerance
    if not symbols:
        raise HTTPException(400, "symbols required")

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        # Get anchors
        from core.models import CompassAnchor
        rec = (
            sess.query(CompassAnchor)
            .filter(CompassAnchor.category == anchor)
            .order_by(CompassAnchor.created_at.desc())  # type: ignore
            .first()
        )
        if rec is None:
            # Let client trigger calibration explicitly to avoid long requests
            raise HTTPException(409, f"Anchor not found: {anchor}")
        a_low = float(rec.mu_low)
        a_high = float(rec.mu_high)
        anchor_meta = {
            "category": rec.category,
            "version": rec.version,
            "mu_low": a_low,
            "mu_high": a_high,
        }

        # Fetch latest snapshots for requested symbols
        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .filter(CvarSnapshot.symbol.in_(symbols))
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        rows = (
            sess.query(CvarSnapshot)
            .join(
                latest,
                and_(
                    CvarSnapshot.symbol == latest.c.symbol,
                    CvarSnapshot.as_of_date == latest.c.mx,
                ),
            )
            .filter(CvarSnapshot.alpha_label == int(payload.get("alpha", 99)))
            .all()
        )
        # Compute worst-of-three CVaR and score
        try:
            from importlib import import_module as _import_module
            _comp = _import_module("nirvana_risk_core.compass")
            _score = getattr(_comp, "score")
        except Exception as exc:
            raise HTTPException(503, f"Compass core unavailable: {exc}")

        def _flt(x: Any) -> float | None:
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        out: list[dict[str, Any]] = []
        for r in rows:
            # Build numeric CVaR list with explicit float conversion for type safety
            _vals = [r.cvar_nig, r.cvar_ghst, r.cvar_evar]
            cvals: list[float] = []
            for v in _vals:
                try:
                    if v is not None:
                        cvals.append(float(v))
                except Exception:
                    pass
            worst = max(cvals) if cvals else None
            mu = _flt(r.return_annual)
            s_val = None
            if mu is not None and worst is not None:
                try:
                    sv = _score(float(mu), float(worst), a_low, a_high, LT, lambda_=2.25, round_to_int=True)
                    s_val = int(round(float(sv))) if isinstance(sv, (int, float)) else None
                except Exception:
                    s_val = None
            out.append({
                "symbol": r.symbol,
                "as_of": r.as_of_date.isoformat() if r.as_of_date else None,
                "return_annual": mu,
                "cvar_worst": worst,
                "compass_score": s_val,
            })
        # Sort by Compass Score descending; place missing scores at the end
        def _sort_key(d: dict[str, Any]):
            s = d.get("compass_score")
            return (s is None, -(int(s) if isinstance(s, (int, float)) else 0))
        out.sort(key=_sort_key)
        return JSONResponse({"items": out, "count": len(out), "anchor": anchor, "anchor_params": anchor_meta})
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/experiments/top")
def exp_top(
    anchor: str,
    limit: int = 20,
    country: str | None = None,
    types: str | None = None,
    candidate_cap: int = 5000,
    lt: float = 0.25,
    alpha: int = 99,
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    anchor = (anchor or "").strip()
    if not anchor:
        raise HTTPException(400, "anchor required")

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        # Load anchor
        from core.models import CompassAnchor
        rec = (
            sess.query(CompassAnchor)
            .filter(CompassAnchor.category == anchor)
            .order_by(CompassAnchor.created_at.desc())  # type: ignore
            .first()
        )
        if rec is None:
            raise HTTPException(409, f"Anchor not found: {anchor}")
        a_low = float(rec.mu_low)
        a_high = float(rec.mu_high)
        anchor_meta = {
            "category": rec.category,
            "version": rec.version,
            "mu_low": a_low,
            "mu_high": a_high,
        }

        # Latest snapshots with optional filters
        # Latest snapshot per symbol for the requested alpha label only
        latest = (
            sess.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("mx"),
            )
            .filter(CvarSnapshot.alpha_label == int(alpha))
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        q = (
            sess.query(CvarSnapshot, PriceSeries)
            .join(
                latest,
                and_(
                    CvarSnapshot.symbol == latest.c.symbol,
                    CvarSnapshot.as_of_date == latest.c.mx,
                ),
            )
            .outerjoin(PriceSeries, PriceSeries.symbol == CvarSnapshot.symbol)
            .filter(PriceSeries.insufficient_history == 0)  # type: ignore
        )
        if country:
            q = q.filter(PriceSeries.country == country)  # type: ignore
        if types:
            allowed = [t.strip() for t in types.split(",") if t.strip()]
            if allowed:
                q = q.filter(PriceSeries.instrument_type.in_(allowed))  # type: ignore

        # Take top candidates by return_annual as a proxy for speed, then score
        rows = (
            q.order_by(CvarSnapshot.return_annual.desc())  # type: ignore
            .limit(max(1, int(candidate_cap)))
            .all()
        )

        try:
            from importlib import import_module as _import_module
            _comp = _import_module("nirvana_risk_core.compass")
            _score = getattr(_comp, "score")
        except Exception as exc:
            raise HTTPException(503, f"Compass core unavailable: {exc}")

        def _flt(x: Any) -> float | None:
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        scored: list[tuple[str, int | None, float, float, str | None, str | None]] = []
        for r, _ps in rows:
            # Build numeric CVaR list with explicit float conversion for type safety
            _vals = [r.cvar_nig, r.cvar_ghst, r.cvar_evar]
            cvals: list[float] = []
            for v in _vals:
                try:
                    if v is not None:
                        cvals.append(float(v))
                except Exception:
                    pass
            worst = max(cvals) if cvals else None
            mu = _flt(r.return_annual)
            if mu is None or worst is None:
                continue
            try:
                LT = float(lt) if lt and lt > 0 else 0.25
                sv = _score(float(mu), float(worst), a_low, a_high, LT, lambda_=2.25, round_to_int=True)
                s_val = int(round(float(sv))) if isinstance(sv, (int, float)) else None
            except Exception:
                s_val = None
            if mu is not None and worst is not None:
                name = None
                country = None
                try:
                    name = getattr(_ps, "name", None)
                except Exception:
                    name = None
                try:
                    country = getattr(_ps, "country", None)
                except Exception:
                    country = None
                scored.append((r.symbol, s_val, float(mu), float(worst), name, country))

        # Sort primarily by presence of score (scored first), then score desc,
        # and finally by annual return desc to surface notable entries lacking score
        scored.sort(
            key=lambda t: (
                (t[1] is None),
                (-(t[1] if isinstance(t[1], (int, float)) else 0)),
                (-t[2] if isinstance(t[2], (int, float)) else 0),
            )
        )
        top = scored[: max(1, int(limit))]
        return JSONResponse({
            "items": [
                {"symbol": s, "name": nm, "country": co, "compass_score": v, "return_annual": mu, "cvar_worst": cw}
                for s, v, mu, cw, nm, co in top
            ],
            "count": len(top),
            "anchor": anchor,
            "anchor_params": anchor_meta,
            "alpha": int(alpha),
            "lt": float(lt) if lt and lt > 0 else 0.25,
            "lambda": 2.25,
        })
    finally:
        try:
            sess.close()
        except Exception:
            pass


