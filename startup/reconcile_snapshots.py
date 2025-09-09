from __future__ import annotations

import json as _json
import logging as _log
import os
from typing import Dict, List, Tuple

from sqlalchemy import text as _text  # type: ignore

from core.db import get_db_session
from core.models import Symbols  # type: ignore
from utils.common import _eodhd_suffix_for  # type: ignore
from utils import service_bus as _sb


def reconcile_cvar_snapshots() -> Dict[str, int]:
    """One-time reconciliation to bind CVaR snapshots to instruments.

    Steps:
      1) For unambiguous symbols (exactly one instrument), set instrument_id
         on existing snapshots where it is NULL.
      2) For ambiguous symbols (multiple instruments with same symbol):
         delete all existing snapshots for that symbol and enqueue
         recalculation per instrument with suffix derived from
         (exchange,country).

    Controlled via startup; safe to run multiple times.
    """
    sess = get_db_session()
    if sess is None:
        return {
            "updated_unambiguous": 0,
            "deleted_ambiguous_rows": 0,
            "enqueued": 0,
        }

    updated_unambiguous = 0
    deleted_rows = 0
    enqueued = 0

    try:
        # Build groups by symbol
        rows = (
            sess.query(
                Symbols.symbol,
                Symbols.id,
                Symbols.exchange,
                Symbols.country,
            )  # type: ignore
            .order_by(Symbols.symbol.asc())
            .all()
        )
        sym_to_variants: Dict[str, List[Tuple[int, str | None, str | None]]] = {}
        for sym, rid, ex, co in rows:
            try:
                sym_to_variants.setdefault(str(sym), []).append(
                    (int(rid), ex, co)
                )
            except Exception:
                continue

        # Split into unambiguous and ambiguous
        unambig: List[Tuple[str, int]] = []
        ambig: Dict[
            str, List[Tuple[int, str | None, str | None]]
        ] = {}
        for sym, variants in sym_to_variants.items():
            if len(variants) == 1:
                unambig.append((sym, variants[0][0]))
            elif len(variants) > 1:
                ambig[sym] = variants

        # 1) Update snapshots for unambiguous
        try:
            if unambig:
                # Batch via SQL for speed
                for sym, inst_id in unambig:
                    try:
                        r = sess.execute(
                            _text(
                                "UPDATE cvar_snapshot SET instrument_id = :iid "
                                "WHERE symbol = :sym AND instrument_id IS NULL"
                            ),
                            {"iid": int(inst_id), "sym": str(sym)},
                        )
                        updated_unambiguous += int(
                            getattr(r, "rowcount", 0) or 0
                        )
                    except Exception:
                        continue
                sess.commit()
        except Exception:
            try:
                sess.rollback()
            except Exception:
                pass

        # 2) For ambiguous: delete and enqueue
        amb_syms = list(ambig.keys())
        if amb_syms:
            # Delete all snapshots for ambiguous symbols
            try:
                r = sess.execute(
                    _text(
                        "DELETE FROM cvar_snapshot WHERE symbol = ANY(:syms)"
                    ),
                    {"syms": amb_syms},
                )
                deleted_rows += int(getattr(r, "rowcount", 0) or 0)
                sess.commit()
            except Exception:
                try:
                    sess.rollback()
                except Exception:
                    pass

            # Enqueue recalculation per instrument variant
            conn = _sb.sb_connection_string()
            qname = _sb.sb_queue_name()
            if conn and qname:
                try:
                    from azure.servicebus import (  # type: ignore
                        ServiceBusClient,
                        ServiceBusMessage,
                    )
                    batch_size = max(50, int(os.getenv("SB_BATCH", "100")))
                    alphas = [0.99, 0.95, 0.50]
                    with ServiceBusClient.from_connection_string(conn) as client:
                        sender = client.get_queue_sender(queue_name=qname)
                        with sender:
                            pending: List[ServiceBusMessage] = []
                            for sym, variants in ambig.items():
                                for rid, ex, co in variants:
                                    suf = _eodhd_suffix_for(ex, co)
                                    body = {
                                        "symbol": sym,
                                        "alphas": alphas,
                                        "force": True,
                                        "suffix": suf,
                                    }
                                    pending.append(
                                        ServiceBusMessage(_json.dumps(body))
                                    )
                                    enqueued += len(alphas)
                                    if len(pending) >= batch_size:
                                        sender.send_messages(pending)
                                        pending.clear()
                            if pending:
                                sender.send_messages(pending)
                except Exception as exc:
                    try:
                        _log.warning(
                            "reconcile: enqueue failed: %s",
                            exc,
                        )
                    except Exception:
                        pass
            else:
                try:
                    _log.info(
                        "reconcile: SB disabled; skipped enqueue"
                    )
                except Exception:
                    pass

        return {
            "updated_unambiguous": int(updated_unambiguous),
            "deleted_ambiguous_rows": int(deleted_rows),
            "enqueued": int(enqueued),
        }
    finally:
        try:
            sess.close()
        except Exception:
            pass


