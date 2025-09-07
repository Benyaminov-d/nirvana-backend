from __future__ import annotations
import os
import threading
from datetime import date as _date
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore

from utils.auth import (
    basic_auth_if_configured as _basic_auth_if_configured,
    require_pub_or_basic as _require_pub_or_basic,
)
from services.domain.cvar_unified_service import CvarUnifiedService
from core.persistence import (
    upsert_annual_violation,
    get_existing_annual_violation_years,
    list_annual_violations,
)


router = APIRouter()


def _load_prices_series(symbol: str) -> List[tuple[str, float]]:
    from pathlib import Path as _Path
    import requests as _req  # type: ignore

    root = _Path(__file__).parents[2] / "data-snapshots"
    use_csv = os.getenv("LOCAL_CSV_DATA", "False").lower() in (
        "true",
        "1",
        "yes",
    )

    def _csv_path(sym: str) -> _Path:
        return root / (
            "SP500TR.csv"
            if sym == "SP500TR"
            else "Fidelity_mutual_funds20240430.csv"
        )

    if use_csv:
        p = _csv_path(symbol)
        if p.exists():
            try:
                import csv as _csv

                with p.open("r", encoding="utf-8") as f:
                    reader = _csv.DictReader(f)
                    rows = list(reader)
                rows.sort(
                    key=lambda r: str(r.get("Date") or "")
                )
                rows_list: List[tuple[str, float]] = []
                if symbol == "SP500TR":
                    for r in rows:
                        try:
                            d = str(r.get("Date"))
                            raw_v = r.get("SP500TR")
                            v = float(str(raw_v))
                            if v > 0:
                                rows_list.append((d, v))
                        except Exception:
                            continue
                else:
                    for r in rows:
                        try:
                            d = str(r.get("Date"))
                            raw_v2 = r.get(symbol)
                            v = float(str(raw_v2))
                            if v > 0:
                                rows_list.append((d, v))
                        except Exception:
                            continue
                return rows_list
            except Exception:
                pass

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        return []
    map_api = {
        "BTC": "BTC-USD.CC",
        "ETH": "ETH-USD.CC",
        "SP500TR": "SP500TR.INDX",
    }
    from utils.common import resolve_eodhd_endpoint_symbol
    endpoint_symbol = map_api.get(
        symbol, resolve_eodhd_endpoint_symbol(symbol)
    )
    url = f"https://eodhistoricaldata.com/api/eod/{endpoint_symbol}"
    try:
        resp = _req.get(
            url,
            params={"api_token": api_key, "fmt": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception:
        return []
    rows_out: List[tuple[str, float]] = []
    for row in raw:
        try:
            d = str(row.get("date"))
            p_adj = row.get("adjusted_close")
            p_val = (
                p_adj
                if (p_adj is not None and p_adj != 0)
                else row.get("close")
            )
            v = float(p_val)
            if v > 0:
                rows_out.append((d, v))
        except Exception:
            continue
    rows_out.sort(key=lambda t: t[0])
    return rows_out


def _year_return(prices: List[tuple[str, float]], year: int) -> float | None:
    if not prices:
        return None
    jan1 = f"{year}-01-01"
    dec31 = f"{year}-12-31"
    start: tuple[str, float] | None = None
    end: tuple[str, float] | None = None
    for d, v in prices:
        if d >= jan1:
            start = (d, v)
            break
    for d, v in reversed(prices):
        if d <= dec31:
            end = (d, v)
            break
    if start is None or end is None:
        return None
    try:
        s = float(start[1])
        e = float(end[1])
        if s > 0:
            return e / s - 1.0
    except Exception:
        return None
    return None


@router.get("/cvar/annual_test")
def cvar_annual_test(
    symbol: str = "SP500TR",
    start_year: int = 2007,
    end_year: int | None = None,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    svc = CvarUnifiedService()
    today = _date.today()
    if end_year is None:
        end_year = today.year - 1
    end_year = max(start_year, end_year)
    series = _load_prices_series(symbol)
    items: List[Dict] = []
    total_95 = 0
    total_99 = 0
    considered = 0

    for y in range(start_year, end_year + 1):
        asof = f"{y}-12-31"
        data = svc.get_cvar_data(
            symbol,
            force_recalculate=False,
            to_date=asof,
        )
        if not data.get("success"):
            continue

        def _worst(block: dict) -> float | None:
            try:
                vals = [
                    float(block["annual"].get("nig")),
                    float(block["annual"].get("ghst")),
                    float(block["annual"].get("evar")),
                ]
                vals = [v for v in vals if v == v]
                return max(vals) if vals else None
            except Exception:
                return None

        c95 = _worst(data.get("cvar95", {}))
        c99 = _worst(data.get("cvar99", {}))
        next_ret = _year_return(series, y + 1)
        if next_ret is None:
            continue
        considered += 1
        v95 = bool(next_ret < 0 and c95 is not None and (-next_ret) > c95)
        v99 = bool(next_ret < 0 and c99 is not None and (-next_ret) > c99)
        if v95:
            total_95 += 1
        if v99:
            total_99 += 1
        items.append(
            {
                "year": y,
                "as_of": asof,
                "cvar95": c95,
                "cvar99": c99,
                "next_year": y + 1,
                "next_return": next_ret,
                "viol95": v95,
                "viol99": v99,
            }
        )
    return {
        "symbol": symbol,
        "start_year": start_year,
        "end_year": end_year,
        "considered": considered,
        "violations95": total_95,
        "violations99": total_99,
        "items": items,
    }


_annual_jobs: Dict[str, Dict] = {}
_annual_jobs_lock = threading.Lock()


def _parse_products_arg(products: str) -> List[str]:
    val = (products or "").strip()
    if not val or val.lower() == "all":
        return []
    out: List[str] = []
    for tok in val.split(","):
        s = tok.strip()
        if s:
            out.append(s)
    return out


def _compute_annual_violations_for_symbol(
    symbol: str, start_year: int, end_year: int | None
) -> Dict:
    svc = CvarUnifiedService()
    today = _date.today()
    if end_year is None:
        end_year = today.year - 1
    end_year = max(start_year, end_year)
    series = _load_prices_series(symbol)
    items: List[Dict] = []
    total99 = 0
    considered = 0
    years = list(range(start_year, end_year + 1))
    try:
        existing_years = get_existing_annual_violation_years(
            symbol, start_year, end_year
        )
    except Exception:
        existing_years = set()
    missing_years = [y for y in years if y not in existing_years]

    for y in missing_years:
        asof = f"{y}-12-31"
        data = svc.get_cvar_data(symbol, force_recalculate=False, to_date=asof)
        if not data.get("success"):
            continue

        def _triple(block: dict) -> tuple[
            float | None, float | None, float | None, float | None
        ]:
            try:
                nig = float(block["annual"].get("nig"))
            except Exception:
                nig = None
            try:
                ghst = float(block["annual"].get("ghst"))
            except Exception:
                ghst = None
            try:
                evar = float(block["annual"].get("evar"))
            except Exception:
                evar = None
            vals = [v for v in (nig, ghst, evar) if v is not None and v == v]
            worst = max(vals) if vals else None
            return nig, ghst, evar, worst

        n99, g99, e99, w99 = _triple(data.get("cvar99", {}))
        next_ret = _year_return(series, y + 1)
        if next_ret is None:
            continue
        considered += 1
        viol99 = bool(next_ret < 0 and w99 is not None and (-next_ret) > w99)
        if viol99:
            total99 += 1
        try:
            from datetime import datetime as _dt

            as_of_date = _dt.strptime(asof, "%Y-%m-%d").date()
        except Exception:
            as_of_date = _date.fromisoformat(asof)
        try:
            upsert_annual_violation(
                symbol=symbol,
                year=y,
                as_of_date=as_of_date,
                next_year=y + 1,
                next_return=next_ret,
                cvar99_nig=n99,
                cvar99_ghst=g99,
                cvar99_evar=e99,
                cvar99_worst=w99,
                violated99=viol99,
            )
        except Exception:
            pass
        items.append(
            {
                "year": y,
                "as_of": asof,
                "cvar99_nig": n99,
                "cvar99_ghst": g99,
                "cvar99_evar": e99,
                "cvar99_worst": w99,
                "next_year": y + 1,
                "next_return": next_ret,
                "viol99": viol99,
            }
        )
    return {
        "symbol": symbol,
        "start_year": start_year,
        "end_year": end_year,
        "considered": considered,
        "violations99": total99,
        "items": items,
    }


@router.get("/annual")
def annual_page(
    _auth: None = Depends(_basic_auth_if_configured),
) -> HTMLResponse:
    from pathlib import Path as _Path

    page = (
        _Path(__file__).parents[2]
        / "frontend"
        / "templates"
        / "annual.html"
    )
    if page.exists():
        return HTMLResponse(page.read_text(), 200)
    raise HTTPException(404, "annual.html not found")


@router.get("/cvar/annual_test_start")
def cvar_annual_test_start(
    symbol: str = "SP500TR",
    start_year: int = 2007,
    end_year: int | None = None,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    job_id = os.urandom(8).hex()
    with _annual_jobs_lock:
        _annual_jobs[job_id] = {"status": "running", "progress": 0.0}
    t = threading.Thread(
        target=_annual_job_run,
        args=(job_id, symbol, start_year, end_year),
        daemon=True,
    )
    t.start()
    return {"job": job_id}


def _annual_job_run(
    job_id: str,
    symbol: str,
    start_year: int,
    end_year: int | None,
) -> None:
    try:
        svc = CvarUnifiedService()
        today = _date.today()
        if end_year is None:
            end_year = today.year - 1
        end_year = max(start_year, end_year)
        series = _load_prices_series(symbol)
        items: List[Dict] = []
        total_95 = 0
        total_99 = 0
        considered = 0
        years = list(range(start_year, end_year + 1))
        n = len(years)
        for idx, y in enumerate(years, start=1):
            asof = f"{y}-12-31"
            data = svc.get_cvar_data(
                symbol,
                force_recalculate=False,
                to_date=asof,
            )
            if not data.get("success"):
                with _annual_jobs_lock:
                    st = _annual_jobs.get(job_id)
                    if st:
                        st["progress"] = idx / max(1, n)
                continue

            def _worst(block: dict) -> float | None:
                try:
                    vals = [
                        float(block["annual"].get("nig")),
                        float(block["annual"].get("ghst")),
                        float(block["annual"].get("evar")),
                    ]
                    vals = [v for v in vals if v == v]
                    return max(vals) if vals else None
                except Exception:
                    return None

            c95 = _worst(data.get("cvar95", {}))
            c99 = _worst(data.get("cvar99", {}))
            next_ret = _year_return(series, y + 1)
            if next_ret is None:
                with _annual_jobs_lock:
                    st = _annual_jobs.get(job_id)
                    if st:
                        st["progress"] = idx / max(1, n)
                continue
            considered += 1
            v95 = bool(next_ret < 0 and c95 is not None and (-next_ret) > c95)
            v99 = bool(next_ret < 0 and c99 is not None and (-next_ret) > c99)
            if v95:
                total_95 += 1
            if v99:
                total_99 += 1
            items.append(
                {
                    "year": y,
                    "as_of": asof,
                    "cvar95": c95,
                    "cvar99": c99,
                    "next_year": y + 1,
                    "next_return": next_ret,
                    "viol95": v95,
                    "viol99": v99,
                }
            )
            with _annual_jobs_lock:
                st = _annual_jobs.get(job_id)
                if st:
                    st["progress"] = idx / max(1, n)
        payload = {
            "symbol": symbol,
            "start_year": start_year,
            "end_year": end_year,
            "considered": considered,
            "violations95": total_95,
            "violations99": total_99,
            "items": items,
        }
        with _annual_jobs_lock:
            st = _annual_jobs.get(job_id)
            if st is not None:
                st.update(
                    {
                        "status": "done",
                        "result": payload,
                        "progress": 1.0,
                    }
                )
    except Exception as exc:
        with _annual_jobs_lock:
            st = _annual_jobs.get(job_id)
            if st is not None:
                st.update({"status": "error", "error": str(exc)})


@router.get("/cvar/annual_test_status")
def cvar_annual_test_status(
    job: str,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    with _annual_jobs_lock:
        st = _annual_jobs.get(job)
        if not st:
            raise HTTPException(404, "job not found")
        return st


@router.get("/annual_violations")
def annual_violations(
    products: str = Query(
        "all",
        description="Comma-separated tickers or 'all'",
    ),
    start_year: int = 2007,
    end_year: int | None = None,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    syms = _parse_products_arg(products)
    if not syms:
        raise HTTPException(400, "no products to process")
    job_id = os.urandom(8).hex()
    with _annual_jobs_lock:
        _annual_jobs[job_id] = {
            "status": "running",
            "progress": 0.0,
            "total": len(syms),
            "symbols": syms,
        }

    def _run():
        try:
            import concurrent.futures as _cf

            results: List[Dict] = []
            max_workers = max(
                2,
                int(os.getenv("NVAR_ANNUAL_WORKERS", "8")),
            )
            with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_sym = {
                    ex.submit(
                        _compute_annual_violations_for_symbol,
                        s,
                        start_year,
                        end_year,
                    ): s
                    for s in syms
                }
                total = len(fut_to_sym)
                done_n = 0
                for fut in _cf.as_completed(fut_to_sym):
                    sym = fut_to_sym[fut]
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        results.append({"symbol": sym, "error": str(exc)})
                    done_n += 1
                    with _annual_jobs_lock:
                        st = _annual_jobs.get(job_id)
                        if st is not None:
                            st["progress"] = done_n / max(1, total)
            with _annual_jobs_lock:
                st = _annual_jobs.get(job_id)
                if st is not None:
                    st.update(
                        {
                            "status": "done",
                            "result": {"results": results},
                            "progress": 1.0,
                        }
                    )
        except Exception as exc:
            with _annual_jobs_lock:
                st = _annual_jobs.get(job_id)
                if st is not None:
                    st.update({"status": "error", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return {"job": job_id, "submitted": len(syms)}


@router.get("/annual_violations_db")
def annual_violations_db(
    symbol: str = Query("SP500TR"),
    start_year: int = 2007,
    end_year: int | None = None,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    today = _date.today()
    if end_year is None:
        end_year = today.year - 1
    end_year = max(start_year, end_year)
    rows = list_annual_violations(symbol, start_year, end_year)
    items: List[Dict] = []
    total99 = 0
    considered = 0
    for r in rows:
        try:
            y = int(r.year)
        except Exception:
            continue
        asof = (
            r.as_of_date.isoformat()
            if getattr(r, "as_of_date", None)
            else f"{y}-12-31"
        )
        next_ret = r.next_return
        w99 = r.cvar99_worst
        v99 = bool(r.violated99)
        if next_ret is not None:
            considered += 1
        if v99:
            total99 += 1
        items.append(
            {
                "year": y,
                "as_of": asof,
                "cvar99_nig": r.cvar99_nig,
                "cvar99_ghst": r.cvar99_ghst,
                "cvar99_evar": r.cvar99_evar,
                "cvar99_worst": w99,
                "next_year": r.next_year,
                "next_return": next_ret,
                "viol99": v99,
            }
        )
    return {
        "symbol": symbol,
        "start_year": start_year,
        "end_year": end_year,
        "considered": considered,
        "violations99": total99,
        "items": items,
    }


@router.get("/annual_violations_status")
def annual_violations_status(
    job: str,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    with _annual_jobs_lock:
        st = _annual_jobs.get(job)
        if not st:
            raise HTTPException(404, "job not found")
        return st


