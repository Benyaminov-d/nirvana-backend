from __future__ import annotations

import os
from pathlib import Path
import pandas as pd  # type: ignore


def lambert_root() -> Path:
    """Resolve lambert CSV directory across dev and container.
    Order: env override, then backend/lambert, then repo-root/lambert.
    """
    lam_env = os.getenv("NIR_LAMBERT_DIR") or os.getenv("LAMBS_DIR")
    if lam_env:
        p = Path(str(lam_env))
        if p.exists():
            return p
    # services/ -> backend/
    backend_dir = Path(__file__).parents[1]
    p1 = backend_dir / "lambert"
    if p1.exists():
        return p1
    # repo-root/lambert (backend/..)
    p2 = backend_dir.parents[1] / "lambert"
    return p2


def lambert_nig_lookup(symbol: str) -> dict[str, float] | None:
    """Return Lambert NIG ETL targets for symbol, or None if unavailable.
    Looks into lambert/SP500TR_cvar_nig.csv or
    lambert/Fidelity_mutual_funds20240430_cvar_nig.csv
    """
    root = lambert_root()
    if symbol == "SP500TR":
        path = root / "SP500TR_cvar_nig.csv"
    else:
        path = root / "Fidelity_mutual_funds20240430_cvar_nig.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Fund" not in df.columns:
        return None
    row = df.loc[df["Fund"] == symbol]
    if row.empty:
        return None
    rec = row.iloc[0]
    out: dict[str, float] = {}
    for col in ("ETL_1", "ETL_5"):
        if col in df.columns:
            try:
                out[col] = float(rec[col])
            except Exception:
                pass
    return out if out else None


def lambert_nig_params(symbol: str) -> dict[str, float] | None:
    root = lambert_root()
    if symbol == "SP500TR":
        path = root / "SP500TR_cvar_nig.csv"
    else:
        path = root / "Fidelity_mutual_funds20240430_cvar_nig.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Fund" not in df.columns:
        return None
    row = df.loc[df["Fund"] == symbol]
    if row.empty:
        return None
    rec = row.iloc[0]
    out: dict[str, float] = {}
    for col in ("mu", "omega", "alpha1", "gamma1", "beta1"):
        if col in df.columns:
            try:
                out[col] = float(rec[col])
            except Exception:
                pass
    return out if out else None


def lambert_ghst_lookup(symbol: str) -> dict[str, float] | None:
    """Lambert GHST ETL targets (if available)."""
    root = lambert_root()
    if symbol == "SP500TR":
        path = root / "SP500TR_cvar_ghst.csv"
    else:
        path = root / "Fidelity_mutual_funds20240430_cvar_ghst.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Fund" not in df.columns:
        return None
    row = df.loc[df["Fund"] == symbol]
    if row.empty:
        return None
    rec = row.iloc[0]
    out: dict[str, float] = {}
    for col in ("ETL_1", "ETL_5", "shape", "skew"):
        if col in df.columns:
            try:
                out[col] = float(rec[col])
            except Exception:
                pass
    return out if out else None


def lambert_evar_lookup(symbol: str) -> dict[str, float] | None:
    """Lambert EVaR targets (if available).
    Files provide EVaR_99 and EVaR_99_1y. We expose ETL_1 mapped from
    EVaR_99_1y for consistency with UI selection for α=99. No 95% EVaR.
    """
    root = lambert_root()
    if symbol == "SP500TR":
        path = root / "SP500TR_evar.csv"
    else:
        path = root / "Fidelity_mutual_funds20240430_evar.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Fund" not in df.columns:
        return None
    row = df.loc[df["Fund"] == symbol]
    if row.empty:
        return None
    rec = row.iloc[0]
    out: dict[str, float] = {}
    # Map EVaR_99_1y → ETL_1 for α=99 one-year horizon
    if "EVaR_99_1y" in df.columns:
        try:
            out["ETL_1"] = float(rec["EVaR_99_1y"])
        except Exception:
            pass
    return out if out else None


