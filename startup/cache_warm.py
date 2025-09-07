from __future__ import annotations

from core.db import get_db_session
from core.models import CvarSnapshot


def warm_cache_from_db(db_ready: bool) -> None:
    if not db_ready:
        return
    from services.domain.cvar_unified_service import CvarUnifiedService  # lazy import

    try:
        sess = get_db_session()
        if sess is None:
            return
        try:
            rows = (
                sess.query(CvarSnapshot)
                .order_by(
                    CvarSnapshot.symbol.asc(),
                    CvarSnapshot.as_of_date.desc(),
                )
                .all()
            )
            by_symbol: dict[str, dict[int, CvarSnapshot]] = {}
            for r in rows:
                slot = by_symbol.setdefault(r.symbol, {})
                if r.alpha_label not in slot:
                    slot[r.alpha_label] = r
            calc = CvarUnifiedService()
            for sym, parts in by_symbol.items():
                any_block = parts.get(95) or parts.get(99) or parts.get(50)
                if any_block is None:
                    continue
                as_of = any_block.as_of_date.isoformat()

                def _triple(opt: CvarSnapshot | None) -> dict:
                    if opt is None:
                        return {"nig": None, "ghst": None, "evar": None}
                    return {"nig": opt.cvar_nig, "ghst": opt.cvar_ghst, "evar": opt.cvar_evar}

                payload = {
                    "success": True,
                    "as_of_date": as_of,
                    "start_date": None,
                    "data_summary": {},
                    "cached": True,
                    "cvar50": {"annual": _triple(parts.get(50)), "snapshot": _triple(parts.get(50)), "alpha": 0.5},
                    "cvar95": {"annual": _triple(parts.get(95)), "snapshot": _triple(parts.get(95)), "alpha": 0.05},
                    "cvar99": {"annual": _triple(parts.get(99)), "snapshot": _triple(parts.get(99)), "alpha": 0.01},
                }
                try:
                    calc.set_cached(sym, payload)
                except Exception:
                    continue
        finally:
            try:
                sess.close()
            except Exception:
                pass
    except Exception:
        pass
