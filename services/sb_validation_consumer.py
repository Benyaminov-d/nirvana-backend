from __future__ import annotations

import json
import logging
import os
from datetime import date as _date
from typing import Any, Dict

from core.db import get_db_session
from services.validation_integration import process_ticker_validation

_log = logging.getLogger("sb_validation_consumer")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s sb_validation_consumer %(levelname)s: %(message)s"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False


def _conn() -> str | None:
    return os.getenv("SB_CONNECTION") or os.getenv("SERVICEBUS_CONNECTION")


def _queue() -> str | None:
    return os.getenv("VALIDATION_RESULTS_QUEUE")


def start_validation_consumer() -> None:
    conn = _conn()
    q = _queue()
    if not conn or not q:
        _log.info("Validation consumer disabled (missing SB connection or queue name)")
        return

    try:
        from azure.servicebus import ServiceBusClient  # type: ignore
        from azure.servicebus.exceptions import ServiceBusConnectionError, ServiceBusError  # type: ignore
    except Exception:
        _log.warning("azure.servicebus not available; validation consumer disabled")
        return

    prefetch = int(os.getenv("SB_PREFETCH", "200"))
    max_batch = int(os.getenv("SB_MAX_BATCH", "50"))
    backoff_seconds = int(os.getenv("SB_RECONNECT_BACKOFF", "5"))

    _log.info("Validation consumer starting: queue=%s", q)

    while True:
        try:
            with ServiceBusClient.from_connection_string(conn) as client:
                with client.get_queue_receiver(queue_name=q, prefetch_count=prefetch) as receiver:  # type: ignore
                    while True:
                        try:
                            msgs = receiver.receive_messages(
                                max_message_count=max_batch, max_wait_time=5
                            )
                        except (ServiceBusConnectionError, ServiceBusError) as e:
                            _log.warning("Validation SB receive error, reconnecting: %s", e)
                            break  # break inner loop to recreate receiver/client

                        if not msgs:
                            continue
                        _log.info("Validation consumer received: %d messages", len(msgs))
                        for msg in msgs:
                            payload: Dict[str, Any] = {}
                            try:
                                parts = list(getattr(msg, "body", []))  # type: ignore[attr-defined]
                                body_bytes = b"".join(
                                    p if isinstance(p, (bytes, bytearray)) else bytes(p) for p in parts
                                ) if parts else (msg.get_body() if hasattr(msg, "get_body") else b"")
                                decoded = body_bytes.decode("utf-8", "replace") if body_bytes else "{}"
                                payload = json.loads(decoded)
                            except Exception as e:
                                _log.warning("Failed to parse validation message: %s", e)
                                try:
                                    receiver.complete_message(msg)
                                except Exception:
                                    pass
                                continue

                            sym = str(payload.get("symbol") or "").strip().upper()
                            as_of = str(payload.get("as_of") or payload.get("as_of_date") or "")
                            country = payload.get("country")
                            validation_data = payload
                            if not sym:
                                try:
                                    receiver.complete_message(msg)
                                except Exception:
                                    pass
                                continue

                            try:
                                if as_of:
                                    from datetime import datetime as _dt
                                    as_of_date = _dt.fromisoformat(as_of).date()
                                else:
                                    as_of_date = _date.today()
                            except Exception:
                                as_of_date = _date.today()

                            sess = get_db_session()
                            if not sess:
                                _log.warning("DB session unavailable; abandoning validation message for redelivery")
                                try:
                                    receiver.abandon_message(msg)
                                except Exception:
                                    pass
                                continue

                            try:
                                process_ticker_validation(
                                    symbol=sym,
                                    validation_data=validation_data,
                                    country=country,
                                    as_of_date=as_of_date,
                                    db_session=sess,
                                )
                                sess.commit()
                                _log.info(
                                    "Validation saved: symbol=%s as_of=%s",
                                    sym,
                                    as_of or as_of_date.isoformat(),
                                )
                                receiver.complete_message(msg)
                            except Exception as e:
                                try:
                                    sess.rollback()
                                except Exception:
                                    pass
                                _log.warning("Validation save failed for %s: %s", sym, e)
                                try:
                                    receiver.abandon_message(msg)
                                except Exception:
                                    pass
                            finally:
                                try:
                                    sess.close()
                                except Exception:
                                    pass
        except Exception as e:
            _log.warning("Validation SB client error, retrying in %ss: %s", backoff_seconds, e)
        try:
            import time as _t
            _t.sleep(backoff_seconds)
        except Exception:
            pass


