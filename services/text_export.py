from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _fmt_month_year(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%b %Y")
    except Exception:  # noqa: BLE001
        return date_str


def _ppct(x: Any) -> str:
    try:
        v = float(x)
        return f"-{int(round(v * 100))}%"
    except Exception:  # noqa: BLE001
        return "-"


def append_cvar_snapshot(
    project_root: Path,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    cvar50: Dict[str, Any],
    cvar95: Dict[str, Any],
    cvar99: Dict[str, Any],
) -> None:
    """Append a human-readable snapshot to project_root/calculated_data.txt"""
    path = project_root / "calculated_data.txt"

    s_from = _fmt_month_year(start_date)
    s_to = _fmt_month_year(end_date)

    l1 = f"{symbol} \\ Data from {s_from} to {s_to}\n"

    try:
        if isinstance(cvar50, dict):
            raw_a50 = cvar50.get("alpha")
            a50 = float(raw_a50) if raw_a50 is not None else float("nan")
        else:
            a50 = float("nan")
    except Exception:  # noqa: BLE001
        a50 = float("nan")
    if math.isfinite(a50) and 0 < a50 < 1:
        l1b = f"CVaR {int(round(a50 * 100))}%\n"
    else:
        l1b = "CVaR 50%\n"

    if isinstance(cvar50, dict):
        l1c = (
            f"NIG: {_ppct(cvar50.get('annual', {}).get('nig'))} | "
            f"GHST: {_ppct(cvar50.get('annual', {}).get('ghst'))} | "
            f"EVaR: {_ppct(cvar50.get('annual', {}).get('evar'))}\n"
        )
    else:
        l1c = ""

    l2 = "CVaR 95%\n"
    l3 = (
        f"NIG: {_ppct(cvar95.get('annual', {}).get('nig'))} | "
        f"GHST: {_ppct(cvar95.get('annual', {}).get('ghst'))} | "
        f"EVaR: {_ppct(cvar95.get('annual', {}).get('evar'))}\n"
    )
    l4 = "CVaR 99%\n"
    l5 = (
        f"NIG: {_ppct(cvar99.get('annual', {}).get('nig'))} | "
        f"GHST: {_ppct(cvar99.get('annual', {}).get('ghst'))} | "
        f"EVaR: {_ppct(cvar99.get('annual', {}).get('evar'))}\n\n"
    )

    if not path.exists():
        path.write_text("", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(l1)
        f.write(l1b)
        f.write(l1c)
        f.write(l2)
        f.write(l3)
        f.write(l4)
        f.write(l5)


