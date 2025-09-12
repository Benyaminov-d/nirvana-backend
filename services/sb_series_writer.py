from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from core.db import get_db_session
from core.models import Symbols
from repositories.time_series_repository import TimeSeriesRepository

_log = logging.getLogger("sb_series_writer")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s sb_series_writer %(levelname)s: %(message)s"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False


def _conn() -> str | None:
    return os.getenv("SB_CONNECTION") or os.getenv("SERVICEBUS_CONNECTION")


def _series_queue() -> str | None:
    return os.getenv("SERIES_WRITE_QUEUE")


def start_series_writer_consumer() -> None:
    """Start background consumer for series-write-q to persist raw series.

    Expected message schema (JSON):
      {
        "symbol": "AAPL",
        "as_of": "YYYY-MM-DD",
        "version_id": "AAPL-20250131T235959Z",
        "source_type": "raw",
        "series": [
          {"date": "YYYY-MM-DD", "price": 123.45, "volume": 100}, ...
        ]
      }
    """
    conn = _conn()
    q = _series_queue()
    if not conn or not q:
        _log.info("Series writer disabled (missing SB connection or queue name)")
        return

    try:
        from azure.servicebus import ServiceBusClient  # type: ignore
    except Exception:
        _log.warning("azure.servicebus not available; series writer disabled")
        return

    _log.info("Series writer consumer starting: queue=%s", q)
    repo = TimeSeriesRepository()

    def _resolve_symbol_id(symbol: str) -> int | None:
        sess = get_db_session()
        if not sess:
            return None
        try:
            row = (
                sess.query(Symbols.id)
                .filter(Symbols.symbol == symbol)
                .one_or_none()
            )
            return int(row[0]) if row else None
        except Exception:
            return None
        finally:
            try:
                sess.close()
            except Exception:
                pass

    with ServiceBusClient.from_connection_string(conn) as client:
        receiver = client.get_queue_receiver(queue_name=q, prefetch_count=int(os.getenv("SB_PREFETCH", "50")))
        with receiver:  # type: ignore
            while True:
                msgs = receiver.receive_messages(max_message_count=int(os.getenv("SB_MAX_BATCH", "10")), max_wait_time=5)
                if not msgs:
                    continue
                _log.info("Series writer received: %d messages", len(msgs))
                for msg in msgs:
                    try:
                        body_bytes = b"".join(list(getattr(msg, "body", []))) if hasattr(msg, "body") else msg.get_body()
                        payload: Dict[str, Any] = json.loads(body_bytes.decode("utf-8", "replace")) if body_bytes else {}
                    except Exception as e:
                        _log.warning("Failed to parse message body: %s", e)
                        try:
                            receiver.complete_message(msg)
                        except Exception:
                            pass
                        continue

                    symbol = str(payload.get("symbol") or "").strip().upper()
                    version_id = str(payload.get("version_id") or "").strip()
                    source_type = str(payload.get("source_type") or "raw").strip()
                    series: List[Dict[str, Any]] = payload.get("series") or []

                    if not symbol or not version_id or not series:
                        _log.info("Skip series message: missing symbol/version/series")
                        try:
                            receiver.complete_message(msg)
                        except Exception:
                            pass
                        continue

                    symbol_id = _resolve_symbol_id(symbol)
                    if not symbol_id:
                        _log.info("Skip series message: unknown symbol=%s", symbol)
                        try:
                            receiver.complete_message(msg)
                        except Exception:
                            pass
                        continue

                    sess = get_db_session()
                    if not sess:
                        _log.warning("DB session unavailable; abandoning message for redelivery")
                        try:
                            receiver.abandon_message(msg)
                        except Exception:
                            pass
                        continue

                    try:
                        written = repo.bulk_upsert(
                            sess,
                            symbol_id=symbol_id,
                            version_id=version_id,
                            source_type=source_type,
                            rows=(
                                {
                                    "date": r.get("date"),
                                    "price": r.get("price"),
                                    "volume": r.get("volume"),
                                }
                                for r in series
                            ),
                            chunk_size=int(os.getenv("PTS_CHUNK_SIZE", "5000")),
                        )
                        sess.commit()
                        _log.info("Series upserted: symbol=%s version=%s rows=%d", symbol, version_id, written)
                        receiver.complete_message(msg)
                    except Exception as e:
                        try:
                            sess.rollback()
                        except Exception:
                            pass
                        _log.warning("Series upsert failed for %s: %s", symbol, e)
                        try:
                            receiver.abandon_message(msg)
                        except Exception:
                            pass
                    finally:
                        try:
                            sess.close()
                        except Exception:
                            pass


