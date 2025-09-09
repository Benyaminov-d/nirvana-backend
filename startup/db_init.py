from __future__ import annotations

from pathlib import Path

from core.persistence import (
    init_db_if_configured,
    bootstrap_annual_violations_from_csv,
)
from core.db import get_db_session
from core.models import Symbols  # type: ignore
from utils.common import (  # type: ignore
    canonical_instrument_type as _canon_type,
)


def ensure_db_ready() -> bool:
    try:
        return init_db_if_configured()
    except Exception:
        return False


def bootstrap_annual_if_any(db_ready: bool) -> None:
    if not db_ready:
        return
    try:
        csv_path = (
            Path(__file__).parents[1]
            / "data"
            / "annual_cvar_violation_202508141906.csv"
        )
        if csv_path.exists():
            _ = bootstrap_annual_violations_from_csv(str(csv_path))
    except Exception:
        pass


def normalize_instrument_types() -> int:
    """One-time normalization of instrument_type values in DB at startup.

    Maps variants like 'fund', 'FUND' → 'Fund' per canonical dictionary.
    Returns number of rows updated (best-effort).
    """
    sess = get_db_session()
    if sess is None:
        return 0
    try:
        rows = sess.query(Symbols).all()  # type: ignore
        updated = 0
        for r in rows:
            try:
                new_val = _canon_type(getattr(r, "instrument_type", None))
                if new_val != getattr(r, "instrument_type", None):
                    r.instrument_type = new_val
                    updated += 1
            except Exception:
                continue
        if updated:
            sess.commit()
        return int(updated)
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return 0


def normalize_countries() -> int:
    """Normalize `Symbols.country` values to canonical labels.

    Maps variants like 'us', 'USA', 'United States' → 'US';
    'ca', 'CAN', 'Canada' → 'Canada'. Returns number of rows updated.
    """
    sess = get_db_session()
    if sess is None:
        return 0
    try:
        rows = sess.query(Symbols).all()  # type: ignore
        updated = 0

        def _canon_country(v: object | None) -> str | None:
            try:
                s = str(v).strip()
            except Exception:
                return None
            if not s:
                return None
            s_low = s.lower()
            if s_low in (
                "us",
                "usa",
                "united states",
                "united states of america",
            ):
                return "US"
            if s_low in ("ca", "can", "canada"):
                return "Canada"
            return s
        for r in rows:
            try:
                new_val = _canon_country(getattr(r, "country", None))
                if new_val != getattr(r, "country", None):
                    r.country = new_val
                    updated += 1
            except Exception:
                continue
        if updated:
            sess.commit()
        return int(updated)
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return 0
    finally:
        try:
            sess.close()
        except Exception:
            pass
