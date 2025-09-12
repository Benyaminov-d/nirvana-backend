from __future__ import annotations

import json
import logging
import os
from datetime import date as _date
from typing import Any, Dict

from core.db import get_db_session
from core.models import CompassInputs

_log = logging.getLogger("sb_compass_consumer")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s sb_compass_consumer %(levelname)s: %(message)s"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False


def _conn() -> str | None:
    return os.getenv("SB_CONNECTION") or os.getenv("SERVICEBUS_CONNECTION")


def _queue() -> str | None:
    return os.getenv("COMPASS_RESULTS_QUEUE")


def start_compass_results_consumer() -> None:
    conn = _conn()
    q = _queue()
    if not conn or not q:
        _log.info("Compass results consumer disabled (missing SB connection or queue)")
        return

    try:
        from azure.servicebus import ServiceBusClient  # type: ignore
    except Exception:
        _log.warning("azure.servicebus not available; compass consumer disabled")
        return

    _log.info("Compass results consumer starting: queue=%s", q)
    with ServiceBusClient.from_connection_string(conn) as client:
        receiver = client.get_queue_receiver(queue_name=q, prefetch_count=int(os.getenv("SB_PREFETCH", "200")))
        with receiver:  # type: ignore
            while True:
                msgs = receiver.receive_messages(max_message_count=int(os.getenv("SB_MAX_BATCH", "50")), max_wait_time=5)
                if not msgs:
                    continue
                _log.info("Compass consumer received: %d messages", len(msgs))
                for msg in msgs:
                    payload: Dict[str, Any] = {}
                    try:
                        parts = list(getattr(msg, "body", []))  # type: ignore[attr-defined]
                        body_bytes = b"".join(p if isinstance(p, (bytes, bytearray)) else bytes(p) for p in parts) if parts else (
                            msg.get_body() if hasattr(msg, "get_body") else b""
                        )
                        decoded = body_bytes.decode("utf-8", "replace") if body_bytes else "{}"
                        payload = json.loads(decoded)
                    except Exception as e:
                        _log.warning("Failed to parse compass message: %s", e)
                        try:
                            receiver.complete_message(msg)
                        except Exception:
                            pass
                        continue

                    # Expected payload: { symbol, instrument_id, category_id, version_id, mu_i, L_i_99 }
                    sym = str(payload.get("symbol") or "").strip().upper()
                    instrument_id = payload.get("instrument_id")
                    category_id = payload.get("category_id") or payload.get("country")
                    version_id = payload.get("version_id")
                    mu_i = payload.get("mu_i")
                    L_i_99 = payload.get("L_i_99")

                    if not instrument_id or version_id is None or mu_i is None or L_i_99 is None:
                        try:
                            receiver.complete_message(msg)
                        except Exception:
                            pass
                        continue

                    sess = get_db_session()
                    if not sess:
                        _log.warning("DB session unavailable; abandoning compass message for redelivery")
                        try:
                            receiver.abandon_message(msg)
                        except Exception:
                            pass
                        continue

                    try:
                        # Upsert CompassInputs by instrument_id/category_id/version_id
                        existing = (
                            sess.query(CompassInputs)
                            .filter(CompassInputs.instrument_id == instrument_id,
                                    CompassInputs.category_id == category_id,
                                    CompassInputs.version_id == version_id)
                            .one_or_none()
                        )
                        if existing:
                            existing.mu_i = float(mu_i)
                            existing.L_i_99 = float(L_i_99)
                        else:
                            sess.add(CompassInputs(
                                instrument_id=instrument_id,
                                category_id=category_id,
                                version_id=version_id,
                                mu_i=float(mu_i),
                                L_i_99=float(L_i_99),
                                data_vendor=str(payload.get("data_vendor") or "EODHD"),
                                run_id=str(payload.get("run_id") or "sb-compass")
                            ))
                        sess.commit()
                        _log.info("Compass inputs upserted: symbol=%s inst=%s ver=%s", sym, instrument_id, version_id)
                        receiver.complete_message(msg)
                    except Exception as e:
                        try:
                            sess.rollback()
                        except Exception:
                            pass
                        _log.warning("Compass upsert failed for %s: %s", sym or instrument_id, e)
                        try:
                            receiver.abandon_message(msg)
                        except Exception:
                            pass
                    finally:
                        try:
                            sess.close()
                        except Exception:
                            pass


