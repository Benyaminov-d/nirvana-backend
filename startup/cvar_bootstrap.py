from __future__ import annotations

import os
import json as _json
import importlib

from core.db import get_db_session
from core.models import PriceSeries, CvarSnapshot
from utils import service_bus as _sb


def _build_symbol_list() -> list[str]:
    sess = get_db_session()
    if sess is None:
        return []
    try:
        q = sess.query(PriceSeries.symbol)
        # include_unknown=True
        q = q.filter(
            (PriceSeries.insufficient_history == 0)
            | (PriceSeries.insufficient_history.is_(None))
        )
        rows = q.all()
        return [s for (s,) in rows]
    except Exception:
        return []
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _enqueue_via_servicebus(symbols: list[str]) -> dict:
    conn = _sb.sb_connection_string()
    qname = _sb.sb_queue_name()
    if not (conn and qname):
        return {"mode": "none", "symbols": 0, "submitted": 0}
    submitted = 0
    corr_ids: list[str] = []
    try:
        sb_mod = importlib.import_module("azure.servicebus")
        ServiceBusClient = getattr(sb_mod, "ServiceBusClient")
        ServiceBusMessage = getattr(sb_mod, "ServiceBusMessage")
        batch_size = max(50, int(os.getenv("SB_BATCH", "100")))
        chunk_size = max(50, int(os.getenv("SB_SYMBOLS_PER_MSG", "100")))
        with ServiceBusClient.from_connection_string(conn) as client:
            sender = client.get_queue_sender(queue_name=qname)
            with sender:
                pending: list = []
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i:i + chunk_size]
                    for sym in chunk:
                        body = {
                            "symbol": sym,
                            "alphas": [0.99, 0.95, 0.50],
                            "force": True,
                        }
                        cid = str(sym)
                        pending.append(
                            ServiceBusMessage(
                                _json.dumps(body),
                                correlation_id=cid,
                            )
                        )
                        corr_ids.append(cid)
                        submitted += 3
                        if len(pending) >= batch_size:
                            sender.send_messages(pending)
                            pending.clear()
                if pending:
                    sender.send_messages(pending)
        return {
            "mode": "servicebus",
            "symbols": len(symbols),
            "submitted": submitted,
        }
    except Exception:
        return {"mode": "error", "symbols": 0, "submitted": 0}


def _compute_locally(symbols: list[str]) -> dict:
    try:
        from datetime import datetime as _dt
        from services.domain.cvar_unified_service import CvarUnifiedService
        from core.persistence import upsert_snapshot_row
    except Exception:
        return {"mode": "none", "symbols": 0, "updates": 0}
    svc = CvarUnifiedService()
    updated = 0
    for sym in symbols:
        try:
            data = svc.get_cvar_data(
                sym,
                force_recalculate=True,
                prefer_local=True,
            )
            if not isinstance(data, dict) or not data.get("success"):
                continue
            as_of_s = str(data.get("as_of_date"))
            try:
                as_of_date = _dt.fromisoformat(as_of_s).date()
            except Exception:
                continue
            years_val = None
            try:
                yv = data.get("summary", {}).get("years")
                years_val = float(yv) if yv is not None else None
            except Exception:
                years_val = None

            def _flt(x: object) -> float | None:
                try:
                    return float(x)  # type: ignore[arg-type]
                except Exception:
                    return None

            for label in (50, 95, 99):
                key = f"cvar{label}"
                blk = data.get(key) or {}
                ann = blk.get("annual") if isinstance(blk, dict) else {}
                alpha_val = None
                try:
                    cond = (
                        isinstance(blk, dict)
                        and (blk.get("alpha") is not None)
                    )
                    if cond:
                        alpha_val = float(blk.get("alpha"))  # type: ignore
                except Exception:
                    alpha_val = None
                upsert_snapshot_row(
                    symbol=sym,
                    as_of_date=as_of_date,
                    alpha_label=label,
                    alpha_conf=alpha_val,
                    years=years_val,
                    cvar_nig=_flt(
                        getattr(ann, "get", lambda *_: None)("nig")
                    ),
                    cvar_ghst=_flt(
                        getattr(ann, "get", lambda *_: None)("ghst")
                    ),
                    cvar_evar=_flt(
                        getattr(ann, "get", lambda *_: None)("evar")
                    ),
                    source="local_startup",
                    return_as_of=None,
                    return_annual=None,
                )
                updated += 1
        except Exception:
            continue
    return {"mode": "local", "symbols": len(symbols), "updates": updated}


def enqueue_all_if_snapshots_empty(db_ready: bool) -> dict | None:
    """If snapshots are empty, enqueue all symbols for CVaR calculation.

    Uses Azure Service Bus when configured; otherwise falls back to local.
    Returns a summary dict or None when skipped.
    """
    if not db_ready:
        return None
    sess = get_db_session()
    if sess is None:
        return None
    try:
        has_any = sess.query(CvarSnapshot.id).limit(1).all()
        if has_any:
            return None
    except Exception:
        return None
    finally:
        try:
            sess.close()
        except Exception:
            pass

    syms = _build_symbol_list()
    if not syms:
        return {"mode": "none", "symbols": 0}
    if _sb.sb_connection_string() and _sb.sb_queue_name():
        return _enqueue_via_servicebus(syms)
    return _compute_locally(syms)
