from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, RedirectResponse

from utils.auth import basic_auth_if_configured as _basic_auth_if_configured


router = APIRouter()


@router.get("/export", response_class=RedirectResponse)
def export_page(
    _auth: None = Depends(_basic_auth_if_configured),
) -> RedirectResponse:
    # Always route to SPA implementation to avoid legacy template/JS drift
    return RedirectResponse(
        url="/spa/export",
        status_code=307,
    )


# ─────────────────────────── CVaR CSV/TXT ───────────────────────────
@router.get("/export/cvars")
def export_cvars_text(
    _auth: None = Depends(_basic_auth_if_configured),
) -> Response:
    """Text export equivalent of legacy /export_cvars."""
    from core.db import get_db_session
    from core.models import CvarSnapshot
    import io

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        rows = (
            sess.query(CvarSnapshot)
            .order_by(
                CvarSnapshot.symbol.asc(),
                CvarSnapshot.as_of_date.desc(),
                CvarSnapshot.alpha_label.asc(),
            )
            .all()
        )
        by_symbol: dict[str, dict[int, CvarSnapshot]] = {}
        for r in rows:
            slot = by_symbol.setdefault(r.symbol, {})
            if r.alpha_label not in slot:
                slot[r.alpha_label] = r
        out = io.StringIO()

        def fmt_month(d):
            try:
                return d.strftime("%b %Y")
            except Exception:
                return str(d)

        for sym in sorted(by_symbol.keys()):
            parts = by_symbol[sym]
            as_of = None
            start = None
            for pr in parts.values():
                d = pr.as_of_date
                s = pr.start_date
                if d is not None and (as_of is None or d > as_of):
                    as_of = d
                if s is not None and (start is None or s < start):
                    start = s
            header = f"{sym} \\ Data from "
            header += (fmt_month(start) if start is not None else "-")
            header += " to "
            header += (fmt_month(as_of) if as_of is not None else "-")
            out.write(header + "\n")

            def p(v: float | None) -> str:
                try:
                    if v is None:
                        return "-"
                    x = float(v)
                    return f"-{int(round(x * 100))}%"
                except Exception:
                    return "-"

            def line_for(label: int, title: str) -> None:
                rec = parts.get(label)
                out.write(title + "\n")
                if rec is None:
                    out.write("NIG: - | GHST: - | EVaR: -\n")
                    return
                out.write(
                    "NIG: "
                    + p(rec.cvar_nig)
                    + " | GHST: "
                    + p(rec.cvar_ghst)
                    + " | EVaR: "
                    + p(rec.cvar_evar)
                    + "\n"
                )

            line_for(50, "CVaR 50%")
            line_for(95, "CVaR 95%")
            line_for(99, "CVaR 99%")
            out.write("\n")

        data = out.getvalue().encode("utf-8")
        headers = {
            "Content-Disposition": "attachment; filename=export_cvars.txt",
        }
        return Response(content=data, media_type="text/plain", headers=headers)
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/export/cvars.csv")
def export_cvars_csv(
    _auth: None = Depends(_basic_auth_if_configured),
    levels: str = Query(
        "",
        description="Comma-separated alpha labels: 50,95,99",
    ),
    level: str = Query(
        "",
        description="Alias of 'levels'",
    ),
    products: str = Query(
        "",
        description=(
            "Comma-separated tickers to include; omit or 'all' for all"
        ),
    ),
    instrument_types: str = Query(
        "",
        description=(
            "Comma-separated instrument types to include "
            "(e.g., ETF,Mutual Fund)"
        ),
    ),
    latest: bool = Query(
        False,
        description=(
            "When true, include only the latest as_of per symbol"
        ),
    ),
    one_row: bool = Query(
        False,
        description=(
            "When true, output one row per symbol with per-alpha columns"
        ),
    ),
    worst_only: bool = Query(
        False,
        description=(
            "When true, include only worst-of-three CVaR:"
            "\n- per row: 'cvar'"
            "\n- one_row: 'cvar_50', 'cvar_95', 'cvar_99'"
        ),
    ),
    exclude: str = Query(
        "",
        description=(
            "Comma-separated column names to exclude from CSV."
            " When one_row is true, use suffixed names like cvar_nig_95 or cvar_95."
        ),
    ),
    sort: str = Query(
        "name_asc",
        description="Sort: name_asc|name_desc|cvar_desc|cvar_asc",
    ),
) -> Response:
    from core.db import get_db_session
    from core.models import CvarSnapshot
    from core.models import Symbols
    import io
    import csv as _csv
    from utils.common import (
        canonical_instrument_type as _canon_type,
    )  # type: ignore

    sess = get_db_session()
    if sess is None:
        raise HTTPException(501, "Database not configured")
    try:
        q = (
            sess.query(CvarSnapshot)
            .order_by(
                CvarSnapshot.symbol.asc(),
                CvarSnapshot.as_of_date.asc(),
                CvarSnapshot.alpha_label.asc(),
            )
        )
        allowed: set[int] = set()
        raw_levels = levels or level
        if raw_levels:
            for tok in raw_levels.split(","):
                t = tok.strip()
                if not t:
                    continue
                try:
                    v = int(t)
                except Exception:
                    continue
                if v in (50, 95, 99):
                    allowed.add(v)
        if allowed:
            q = q.filter(CvarSnapshot.alpha_label.in_(list(allowed)))

        # Optional symbol filter, supports special tokens:
        # 5STARS, 5STARSUS, 5STARSCA
        prod_raw = (products or "").strip()
        if prod_raw and prod_raw.lower() != "all":
            token = prod_raw.upper()
            if token == "5STARS":
                q = q.join(
                    Symbols,
                    Symbols.symbol == CvarSnapshot.symbol,
                ).filter(Symbols.five_stars == 1)
            elif token == "5STARSUS":
                q = (
                    q.join(Symbols, Symbols.symbol == CvarSnapshot.symbol)
                    .filter(Symbols.five_stars == 1)
                    .filter(
                        Symbols.country.in_(
                            [
                                "US",
                                "USA",
                                "United States",
                            ]
                        )
                    )  # type: ignore
                )
            elif token == "5STARSCA":
                q = (
                    q.join(Symbols, Symbols.symbol == CvarSnapshot.symbol)
                    .filter(Symbols.five_stars == 1)
                    .filter(Symbols.country == "Canada")  # type: ignore
                )
            else:
                sym_set = {
                    s.strip().upper()
                    for s in prod_raw.split(",")
                    if s and s.strip()
                }
                if sym_set:
                    q = q.filter(CvarSnapshot.symbol.in_(list(sym_set)))

        # Optional instrument types filter (e.g., ETF, Mutual Fund)
        types_raw = (instrument_types or "").strip()
        if types_raw:
            # Canonicalize and deduplicate provided types
            type_list = []
            seen_types: set[str] = set()
            for tok in types_raw.split(","):
                label = _canon_type(tok)
                if label:
                    if label not in seen_types:
                        type_list.append(label)
                        seen_types.add(label)
            if type_list:
                # Join Symbols if not already joined (safe to re-join by symbol
                # equality in SQLAlchemy)
                q = q.join(Symbols, Symbols.symbol == CvarSnapshot.symbol)
                q = q.filter(Symbols.instrument_type.in_(type_list))

        # Latest-only filter (per symbol) if requested
        if latest:
            from sqlalchemy import func, and_  # type: ignore

            latest_sub = (
                sess.query(
                    CvarSnapshot.symbol,
                    func.max(CvarSnapshot.as_of_date).label("mx"),
                )
                .group_by(CvarSnapshot.symbol)
                .subquery()
            )
            q = q.join(
                latest_sub,
                and_(
                    CvarSnapshot.symbol == latest_sub.c.symbol,
                    CvarSnapshot.as_of_date == latest_sub.c.mx,
                ),
            )

        rows = q.all()

        def _flt(x: object | None) -> float | None:
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        def _worst_of_three(r: CvarSnapshot) -> float | None:
            vals = [
                _flt(r.cvar_nig),
                _flt(r.cvar_ghst),
                _flt(r.cvar_evar),
            ]
            vals = [v for v in vals if v is not None and v == v]
            return max(vals) if vals else None

        # Determine selected alpha labels
        selected_labels: list[int]
        if allowed:
            selected_labels = sorted(list(allowed))
        else:
            selected_labels = [50, 95, 99]

        # Build output rows
        records: list[dict[str, object]] = []
        if not one_row:
            for r in rows:
                rec: dict[str, object] = {
                    "symbol": r.symbol,
                    "as_of": r.as_of_date.isoformat() if r.as_of_date else "",
                    "start_date": r.start_date.isoformat() if r.start_date else "",
                    "years": int(r.years) if r.years is not None else "",
                    "alpha_label": int(r.alpha_label) if r.alpha_label is not None else "",
                    "alpha": (f"{float(r.alpha):.6f}" if r.alpha is not None else ""),
                }
                if worst_only:
                    rec["cvar"] = _worst_of_three(r) or ""
                else:
                    rec["cvar_nig"] = _flt(r.cvar_nig) or ""
                    rec["cvar_ghst"] = _flt(r.cvar_ghst) or ""
                    rec["cvar_evar"] = _flt(r.cvar_evar) or ""
                records.append(rec)
            # Header order
            base = [
                "symbol",
                "as_of",
                "start_date",
                "years",
                "alpha_label",
                "alpha",
            ]
            value_cols = (
                ["cvar"]
                if worst_only
                else [
                    "cvar_nig",
                    "cvar_ghst",
                    "cvar_evar",
                ]
            )
            header = base + value_cols
        else:
            # one row per symbol
            from collections import defaultdict as _dd

            by_sym: dict[str, dict[int, CvarSnapshot]] = _dd(dict)
            for r in rows:
                try:
                    by_sym[r.symbol][int(r.alpha_label)] = r
                except Exception:
                    continue
            for sym, parts in sorted(by_sym.items(), key=lambda t: t[0]):
                # Determine target as_of
                asofs = [
                    getattr(parts.get(lbl), "as_of_date", None)
                    for lbl in selected_labels
                ]
                asofs = [a for a in asofs if a is not None]
                target_asof = max(asofs) if asofs else None
                as_of = target_asof.isoformat() if target_asof else ""
                rec2: dict[str, object] = {"symbol": sym, "as_of": as_of}
                for lbl in selected_labels:
                    row = parts.get(lbl)
                    if (
                        target_asof is not None
                        and getattr(row, "as_of_date", None) != target_asof
                    ):
                        # align to target date when latest-only requested upstream
                        # (missing labels for that date will be empty)
                        row = None
                    suf = f"_{lbl}"
                    if row is None:
                        if worst_only:
                            rec2[f"cvar{suf}"] = ""
                        else:
                            rec2[f"cvar_nig{suf}"] = ""
                            rec2[f"cvar_ghst{suf}"] = ""
                            rec2[f"cvar_evar{suf}"] = ""
                        continue
                    if worst_only:
                        rec2[f"cvar{suf}"] = _worst_of_three(row) or ""
                    else:
                        rec2[f"cvar_nig{suf}"] = _flt(row.cvar_nig) or ""
                        rec2[f"cvar_ghst{suf}"] = _flt(row.cvar_ghst) or ""
                        rec2[f"cvar_evar{suf}"] = _flt(row.cvar_evar) or ""
                records.append(rec2)
            # Header
            header = ["symbol", "as_of"]
            for lbl in selected_labels:
                if worst_only:
                    header.append(f"cvar_{lbl}")
                else:
                    header.extend(
                        [
                            f"cvar_nig_{lbl}",
                            f"cvar_ghst_{lbl}",
                            f"cvar_evar_{lbl}",
                        ]
                    )

        # Apply exclude filter
        exclude_set = {
            c.strip() for c in (exclude or "").split(",") if c and c.strip()
        }
        # Sorting
        def _row_cvar_value(rec: dict[str, object]) -> float:
            try:
                if one_row:
                    if worst_only:
                        vals = [
                            rec.get("cvar_50"),
                            rec.get("cvar_95"),
                            rec.get("cvar_99"),
                        ]
                    else:
                        vals = []
                        for lbl in selected_labels:
                            for fam in (
                                "cvar_nig_",
                                "cvar_ghst_",
                                "cvar_evar_",
                            ):
                                vals.append(rec.get(f"{fam}{lbl}"))
                else:
                    if worst_only:
                        vals = [rec.get("cvar")]
                    else:
                        vals = [
                            rec.get("cvar_nig"),
                            rec.get("cvar_ghst"),
                            rec.get("cvar_evar"),
                        ]
                nums = [float(v) for v in vals if v is not None and str(v) != ""]
                return max(nums) if nums else float("nan")
            except Exception:
                return float("nan")

        sort_mode = (sort or "name_asc").lower()
        if sort_mode in ("name_asc", "name_desc"):
            records.sort(key=lambda r: str(r.get("symbol", "")) or "")
            if sort_mode == "name_desc":
                records.reverse()
        elif sort_mode in ("cvar_desc", "cvar_asc"):
            records.sort(key=_row_cvar_value)
            if sort_mode == "cvar_desc":
                records.reverse()

        # Apply exclude after sorting
        if exclude_set:
            header = [c for c in header if c not in exclude_set]
            for rec in records:
                for col in list(rec.keys()):
                    if col in exclude_set:
                        rec.pop(col, None)

        # Write CSV
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(header)
        for rec in records:
            try:
                w.writerow([rec.get(k, "") for k in header])
            except Exception:
                continue
        data = buf.getvalue().encode("utf-8")
        headers = {
            "Content-Disposition": (
                "attachment; filename=export_cvars.csv"
            )
        }
        return Response(content=data, media_type="text/csv", headers=headers)
    finally:
        try:
            sess.close()
        except Exception:
            pass


# ─────────────────────────── Time series ───────────────────────────
@router.get("/export/timeseries")
def export_timeseries(
    products: str = Query(...),
    _auth: None = Depends(_basic_auth_if_configured),
) -> Response:
    syms = [s.strip() for s in (products or "").split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "no symbols provided")
    if len(syms) == 1:
        from routes.timeseries import get_time_series as _single

        return _single(ticker=syms[0])
    import io
    import zipfile as _zip
    buf = io.BytesIO()
    with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as zf:
        from routes.timeseries import get_time_series as _single

        for s in syms:
            try:
                resp = _single(ticker=s)
                zf.writestr(f"{s}_timeseries.csv", resp.body or b"")
            except Exception:
                zf.writestr(f"{s}_error.txt", "failed to fetch timeseries")
    data = buf.getvalue()
    headers = {
        "Content-Disposition": "attachment; filename=timeseries_export.zip",
    }
    return Response(content=data, media_type="application/zip", headers=headers)


