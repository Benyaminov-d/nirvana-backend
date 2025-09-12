from __future__ import annotations

import os
import importlib

from core.db import get_db_session
from core.models import Symbols, CvarSnapshot
from utils import service_bus as _sb


def _build_symbol_list() -> list[str]:
    sess = get_db_session()
    if sess is None:
        return []
    try:
        # Only include distinct symbols with ValidationFlags.valid = 1
        from core.models import ValidationFlags
        q = (
            sess.query(Symbols.symbol)
            .join(ValidationFlags, Symbols.symbol == ValidationFlags.symbol)
            .filter(ValidationFlags.valid == 1)
            .distinct()
        )

        # Log how many distinct symbols we're processing
        import logging
        logger = logging.getLogger(__name__)
        rows = q.all()
        syms = [s for (s,) in rows]
        logger.info(
            f"Found {len(syms)} valid symbols for CVaR calculation"
        )
        return syms
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error building symbol list: {e}")
        return []
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _enqueue_via_servicebus(symbols: list[str]) -> dict:
    conn = _sb.sb_connection_string()
    # Use CVaR calculations queue for bootstrap
    qname = _sb.sb_cvar_calculations_queue() or _sb.sb_queue_name()
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

        # Detailed logging
        import logging
        import json
        logger = logging.getLogger(__name__)
        logger.info(
            f"Initializing Service Bus connection: conn={conn[:20]}... "
            f"qname={qname}"
        )
        logger.info(
            f"Preparing to enqueue {len(symbols)} symbols "
            f"with batch_size={batch_size}, chunk_size={chunk_size}"
        )

        with ServiceBusClient.from_connection_string(conn) as client:
            sender = client.get_queue_sender(queue_name=qname)
            logger.info(f"Service Bus sender created for queue: {qname}")

            with sender:
                pending: list = []
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i:i + chunk_size]
                    logger.info(
                        f"Processing chunk {i//chunk_size + 1}/"
                        f"{(len(symbols) + chunk_size - 1)//chunk_size}: "
                        f"{len(chunk)} symbols"
                    )

                    for sym in chunk:
                        # Determine exchange for symbol
                        exchange = "US"  # default US

                        # Check for CA/UK from symbols table
                        session = get_db_session()
                        try:
                            symbol_info = (
                                session.query(Symbols)
                                .filter(Symbols.symbol == sym)
                                .first()
                            )
                            if symbol_info and symbol_info.country in [
                                "CA", "Canada"
                            ]:
                                exchange = "TO"
                                # logger.info(
                                #     f"Canadian symbol detected: {sym}, "
                                #     f"using exchange: {exchange}"
                                # )
                            elif symbol_info and symbol_info.country in [
                                "UK", "United Kingdom", "GB", "Great Britain"
                            ]:
                                exchange = "LSE"
                                # logger.info(
                                #     f"UK symbol detected: {sym}, "
                                #     f"using exchange: {exchange}"
                                # )
                        except Exception as ex:
                            logger.warning(
                                f"Error determining exchange for {sym}: {ex}"
                            )
                        finally:
                            session.close()

                        # Map exchange to suffix for downstream consumers (functions)
                        suf = None
                        try:
                            exu = str(exchange or "").strip().upper()
                            if exu == "TO":
                                suf = ".TO"
                            elif exu in ("LSE", "L"):
                                suf = ".LSE"
                        except Exception:
                            suf = None

                        body = {
                            "symbol": sym,
                            "exchange": exchange,
                            "alphas": [0.99, 0.95, 0.50],
                            "force": True,
                        }
                        if suf:
                            body["suffix"] = suf
                        cid = str(sym)
                        pending.append(
                            ServiceBusMessage(
                                json.dumps(body),
                                correlation_id=cid,
                                message_id=f"cvarreq-{sym}"
                            )
                        )
                        corr_ids.append(cid)
                        submitted += 3
                        if len(pending) >= batch_size:
                            logger.info(
                                f"Sending batch of {len(pending)} messages"
                            )
                            sender.send_messages(pending)
                            pending.clear()

                if pending:
                    logger.info(
                        f"Sending final batch of {len(pending)} messages"
                    )
                    sender.send_messages(pending)

                logger.info(
                    f"Successfully enqueued {submitted} messages for "
                    f"{len(symbols)} symbols"
                )

        return {
            "mode": "servicebus",
            "symbols": len(symbols),
            "submitted": submitted,
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Service Bus enqueue failed: {str(e)}")
        return {
            "mode": "error",
            "symbols": 0,
            "submitted": 0,
            "error": str(e),
        }


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
                    cvar_nig=_flt(getattr(ann, "get", lambda *_: None)("nig")),
                    cvar_ghst=_flt(getattr(ann, "get", lambda *_: None)("ghst")),
                    cvar_evar=_flt(getattr(ann, "get", lambda *_: None)("evar")),
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
    # Mark bootstrap start time for retry grace logic
    try:
        import time as _t
        os.environ["CVAR_BOOTSTRAP_START_TS"] = str(int(_t.time()))
    except Exception:
        pass
    if _sb.sb_connection_string() and _sb.sb_symbols_queue():
        return _enqueue_via_servicebus(syms)
    return _compute_locally(syms)
