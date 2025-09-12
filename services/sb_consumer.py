from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date as _date
from typing import Any

from core.db import get_db_session
from core.persistence import (
    upsert_snapshot_row,
    insert_insufficient_data_event,
    upsert_price_last,
    save_cvar_result,
)

_logger = logging.getLogger("sb_consumer")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s sb_consumer %(levelname)s: %(message)s"))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)
_logger.propagate = False

_counters_lock = threading.Lock()
_sb_incoming_count: int = 0
_db_upsert_success: int = 0
_sb_success_events: int = 0
_sb_error_events: int = 0
_sb_abandoned_events: int = 0
_sb_empty_events: int = 0
_sb_recent: list[dict] = []
_sb_err_insufficient_history: int = 0
_sb_err_insufficient_data: int = 0
_sb_err_calc_failed: int = 0
_sb_err_other: int = 0
_sb_insufficient_data_raw: list[dict] = []


def _resolve_instrument_id(symbol: str, suffix: str | None) -> int | None:
    """Best-effort resolve Symbols.id for a symbol with optional suffix.

    Suffix examples: ".US", ".TO", ".V". When provided, we map common
    Canadian suffixes to country=Canada. Otherwise prefer US if multiple.
    """
    try:
        sess = get_db_session()
        if sess is None:
            return None
        from core.models import Symbols  # local import to avoid cycles
        q = sess.query(Symbols.id, Symbols.country).filter(Symbols.symbol == symbol)
        rows = q.all()
        if not rows:
            return None
        # If only one match – return it
        if len(rows) == 1:
            return int(rows[0][0])
        # If suffix provided, use it to prioritize country
        try:
            suf = (suffix or "").strip().upper()
        except Exception:
            suf = ""
        if suf in (".TO", ".V", ".CN", ".NE"):
            for rid, country in rows:
                try:
                    if str(country or "").strip().lower() in ("canada", "ca"):
                        return int(rid)
                except Exception:
                    continue
        # Default preference: US
        for rid, country in rows:
            try:
                co = str(country or "").strip().upper()
                if co in ("US", "USA", "UNITED STATES"):
                    return int(rid)
            except Exception:
                continue
        # Fallback: first row
        return int(rows[0][0])
    except Exception:
        return None
    finally:
        try:
            if 'sess' in locals() and sess is not None:
                sess.close()
        except Exception:
            pass


def _log_skips_enabled() -> bool:
    """Control verbosity of per-event skip logs via env SB_LOG_SKIPS.

    Defaults to disabled to avoid flooding logs when many symbols error.
    Enable by setting SB_LOG_SKIPS=1|true|yes.
    """
    try:
        return (os.getenv("SB_LOG_SKIPS", "0") or "").lower() in ("1", "true", "yes")
    except Exception:
        return False


def _log_payload_enabled() -> bool:
    """Enable verbose raw payload logs via env SB_LOG_PAYLOAD=1|true|yes."""
    try:
        return (os.getenv("SB_LOG_PAYLOAD", "0") or "").lower() in (
            "1",
            "true",
            "yes",
        )
    except Exception:
        return False


def get_counters_snapshot() -> dict:
    with _counters_lock:
        return {
            "sb_incoming": int(_sb_incoming_count),
            "db_upsert_success": int(_db_upsert_success),
            "sb_success": int(_sb_success_events),
            "sb_errors": int(_sb_error_events),
            "sb_abandoned": int(_sb_abandoned_events),
            "sb_empty": int(_sb_empty_events),
            "errors": {
                "insufficient_history": int(_sb_err_insufficient_history),
                "insufficient_data": int(_sb_err_insufficient_data),
                "calc_failed": int(_sb_err_calc_failed),
                "other": int(_sb_err_other),
            },
            "insufficient_data_raw": list(_sb_insufficient_data_raw),
            "recent": list(_sb_recent),
        }


def reset_counters() -> None:
    global _sb_incoming_count, _db_upsert_success
    with _counters_lock:
        _sb_incoming_count = 0
        _db_upsert_success = 0
        globals()["_sb_success_events"] = 0
        globals()["_sb_error_events"] = 0
        globals()["_sb_err_insufficient_history"] = 0
        globals()["_sb_err_insufficient_data"] = 0
        globals()["_sb_err_calc_failed"] = 0
        globals()["_sb_err_other"] = 0
        globals()["_sb_abandoned_events"] = 0
        globals()["_sb_empty_events"] = 0
        globals()["_sb_recent"] = []
        globals()["_sb_insufficient_data_raw"] = []


def _push_recent(item: dict) -> None:
    try:
        with _counters_lock:
            _sb_recent.append(item)
            if len(_sb_recent) > 200:
                del _sb_recent[0: len(_sb_recent) - 200]
    except Exception:
        pass


def _sb_conn() -> str | None:
    return os.getenv("SB_CONNECTION") or os.getenv("SERVICEBUS_CONNECTION")


def _sb_topic() -> str | None:
    return os.getenv("SB_TOPIC") or os.getenv("SERVICEBUS_TOPIC")


def _get_results_queue() -> str | None:
    return os.getenv("SB_RESULTS_QUEUE") or os.getenv("SERVICEBUS_RESULTS_QUEUE")


def start_consumer_loop() -> None:
    conn = _sb_conn()
    topic = _sb_topic()
    sub = os.getenv("SB_SUBSCRIPTION", "monitor")
    q_results = _get_results_queue()
    if not conn:
        _logger.info("SB consumer disabled (no connection)")
        return

    try:
        from azure.servicebus import ServiceBusClient  # type: ignore
    except Exception:
        _logger.warning("azure.servicebus not available; SB consumer disabled")
        return

    import time as __t  # type: ignore

    def _sleep(s: float) -> None:
        try:
            __t.sleep(max(0.0, float(s)))
        except Exception:
            pass

    try:
        base = float(os.getenv("SB_CONSUMER_RETRY_S", "0.5"))
    except Exception:
        base = 0.5
    try:
        cap = float(os.getenv("SB_CONSUMER_RETRY_CAP_S", "30"))
    except Exception:
        cap = 30.0

    backoff = base
    while True:
        try:
            with ServiceBusClient.from_connection_string(conn) as client:
                if topic and sub:
                    _logger.info(
                        "SB consumer starting: topic=%s sub=%s", topic, sub
                    )
                    receiver = client.get_subscription_receiver(
                        topic_name=topic,
                        subscription_name=sub,
                        prefetch_count=int(os.getenv("SB_PREFETCH", "200")),
                    )
                elif q_results:
                    _logger.info("SB consumer starting: queue=%s", q_results)
                    receiver = client.get_queue_receiver(
                        queue_name=q_results,
                        prefetch_count=int(os.getenv("SB_PREFETCH", "200")),
                    )
                else:
                    _logger.info("SB consumer disabled (no topic/sub or results queue)")
                    _sleep(backoff)
                    backoff = min(cap, max(base, backoff * 2.0))
                    continue

                with receiver:  # type: ignore
                    backoff = base
                    try:
                        from azure.servicebus import AutoLockRenewer  # type: ignore

                        _lock_renew_s = int(os.getenv("SB_LOCK_RENEW_S", "300"))
                        renewer = AutoLockRenewer(max_lock_renewal_duration=_lock_renew_s)
                    except Exception:
                        renewer = None  # type: ignore

                    while True:
                        msgs = receiver.receive_messages(
                            max_message_count=int(os.getenv("SB_MAX_BATCH", "50")),
                            max_wait_time=5,
                        )
                        if not msgs:
                            continue
                        _logger.info("SB received: %d messages", len(msgs))
                        for msg in msgs:
                            try:
                                if renewer is not None:
                                    renewer.register(receiver, msg)
                            except Exception:
                                pass

                            payload: dict[str, Any] = {}
                            try:
                                parts = list(
                                    getattr(msg, "body", [])
                                )  # type: ignore[attr-defined]
                                if parts:
                                    body_bytes = b"".join(
                                        p
                                        if isinstance(p, (bytes, bytearray))
                                        else bytes(p)
                                        for p in parts
                                    )
                                else:
                                    body_bytes = (
                                        msg.get_body()  # type: ignore[attr-defined]
                                        if hasattr(msg, "get_body")
                                        else bytes(str(msg), "utf-8")
                                    )
                                try:
                                    decoded_body = body_bytes.decode("utf-8", "replace")
                                    _logger.info(f"Raw message body: {decoded_body[:200]}")
                                    payload = json.loads(decoded_body) if body_bytes else {}
                                except Exception as e:
                                    _logger.error(f"Failed to parse message body: {str(e)}")
                                    payload = {}
                            except Exception:
                                payload = {}

                            # Timing flag
                            _timing = (os.getenv("SB_TIMING", "0") or "").lower() in ("1","true","yes")
                            import time as _t
                            _t0 = _t.perf_counter()
                            if _timing:
                                _logger.info("step:parse start symbol=%s", str(payload.get("symbol") or ""))

                            try:
                                with _counters_lock:
                                    globals()["_sb_incoming_count"] = globals().get("_sb_incoming_count", 0) + 1
                            except Exception:
                                pass

                            sym = str(payload.get("symbol") or "").upper().strip()
                            as_of = str(payload.get("as_of") or payload.get("as_of_date") or "")
                            if _timing:
                                _logger.info("step:parsed symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)

                            try:
                                if as_of:
                                    from datetime import datetime as _dt

                                    as_of_date = _dt.fromisoformat(as_of).date()
                                else:
                                    as_of_date = _date.today()
                            except Exception:
                                as_of_date = _date.today()

                            # Open one DB session per message to reduce churn
                            db_sess = None
                            try:
                                db_sess = get_db_session()
                            except Exception:
                                db_sess = None

                            years = payload.get("years")
                            src = str(payload.get("source") or "calc")
                            try:
                                alpha_in = float(payload.get("alpha"))
                            except Exception:
                                alpha_in = 0.99
                            alpha_bp = int(round(alpha_in * 100)) if alpha_in <= 1 else int(alpha_in)
                            alpha_label = 99 if alpha_bp >= 99 else (95 if alpha_bp >= 95 else 50)

                            # Persist last price if provided by producer (independent from CVaR upsert)
                            try:
                                if _timing:
                                    _logger.info("step:price_upsert start symbol=%s", sym)
                                pclose = payload.get("price_close")
                                if pclose is not None:
                                    curr = None
                                    try:
                                        if isinstance(payload.get("price_currency"), str):
                                            curr = str(payload.get("price_currency"))
                                    except Exception:
                                        curr = None
                                    try:
                                        price_val = float(pclose)
                                    except Exception:
                                        price_val = None
                                    if price_val is not None:
                                        upsert_price_last(
                                            symbol=sym,
                                            as_of_date=as_of_date,
                                            price_close=price_val,
                                            currency=curr,
                                            source=src,
                                            session=db_sess,
                                        )
                                if _timing:
                                    _logger.info("step:price_upsert done symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)
                            except Exception:
                                if _timing:
                                    _logger.info("step:price_upsert error symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)
                                pass

                            any_upsert = False
                            status = str(payload.get("status") or "")
                            code = str(payload.get("code") or "")
                            suffix_hint = None
                            try:
                                sh = payload.get("suffix")
                                if isinstance(sh, str):
                                    suffix_hint = sh.strip()
                            except Exception:
                                suffix_hint = None
                            # Resolve instrument_id once per message
                            inst_id = _resolve_instrument_id(sym, suffix_hint)
                            try:
                                if _timing:
                                    _logger.info("step:snapshot_upsert start symbol=%s", sym)
                                if (
                                    ("cvar_nig" in payload)
                                    or ("cvar_ghst" in payload)
                                    or ("cvar_evar" in payload)
                                ):
                                    ok = upsert_snapshot_row(
                                        symbol=sym,
                                        as_of_date=as_of_date,
                                        alpha_label=alpha_label,
                                        alpha_conf=(
                                            float(payload.get("alpha"))
                                            if payload.get("alpha") is not None
                                            else None
                                        ),
                                        years=years,
                                        cvar_nig=payload.get("cvar_nig"),
                                        cvar_ghst=payload.get("cvar_ghst"),
                                        cvar_evar=payload.get("cvar_evar"),
                                        source=src,
                                        start_date=payload.get("start_date"),
                                        return_as_of=(
                                            float(payload.get("return_as_of"))
                                            if payload.get("return_as_of") is not None
                                            else None
                                        ),
                                        return_annual=(
                                            float(payload.get("annual_return"))
                                            if payload.get("annual_return") is not None
                                            else None
                                        ),
                                        instrument_id=inst_id,
                                        session=db_sess,
                                    )
                                    if ok:
                                        try:
                                            with _counters_lock:
                                                globals()["_db_upsert_success"] = globals().get("_db_upsert_success", 0) + 1
                                        except Exception:
                                            pass
                                        any_upsert = True
                                if _timing:
                                    _logger.info("step:snapshot_upsert done symbol=%s any=%s elapsed_ms=%.1f", sym, any_upsert, ( _t.perf_counter()-_t0)*1000)
                            except Exception:
                                if _timing:
                                    _logger.info("step:snapshot_upsert error symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)
                                any_upsert = False

                            def _u(label_key: str, label_val: int) -> None:
                                if _timing:
                                    _logger.info("step:label_upsert start symbol=%s label=%d", sym, label_val)
                                blk = payload.get(label_key) or {}
                                ok2 = upsert_snapshot_row(
                                    symbol=sym,
                                    as_of_date=as_of_date,
                                    alpha_label=label_val,
                                    alpha_conf=(
                                        float(blk.get("alpha"))
                                        if (
                                            isinstance(blk, dict)
                                            and blk.get("alpha") is not None
                                        )
                                        else None
                                    ),
                                    years=years,
                                    cvar_nig=blk.get("nig"),
                                    cvar_ghst=blk.get("ghst"),
                                    cvar_evar=blk.get("evar"),
                                    source=src,
                                    start_date=payload.get("start_date"),
                                    return_as_of=(
                                        float(payload.get("return_as_of"))
                                        if payload.get("return_as_of")
                                        is not None
                                        else None
                                    ),
                                    return_annual=(
                                        float(payload.get("annual_return"))
                                        if payload.get("annual_return")
                                        is not None
                                        else None
                                    ),
                                    instrument_id=inst_id,
                                    session=db_sess,
                                )
                                if _timing:
                                    _logger.info("step:label_upsert done symbol=%s label=%d ok=%s elapsed_ms=%.1f", sym, label_val, ok2, ( _t.perf_counter()-_t0)*1000)
                                if ok2:
                                    try:
                                        with _counters_lock:
                                            globals()["_db_upsert_success"] = globals().get("_db_upsert_success", 0) + 1
                                    except Exception:
                                        pass
                                    nonlocal any_upsert  # type: ignore
                                    any_upsert = True

                            # If we used a dedicated session, commit once here before anomalies/ACK
                            try:
                                if db_sess is not None:
                                    db_sess.commit()
                            except Exception:
                                try:
                                    if db_sess is not None:
                                        db_sess.rollback()
                                except Exception:
                                    pass

                            try:
                                if isinstance(payload.get("cvar50"), dict):
                                    _u("cvar50", 50)
                                if isinstance(payload.get("cvar95"), dict):
                                    _u("cvar95", 95)
                                if isinstance(payload.get("cvar99"), dict):
                                    _u("cvar99", 99)
                            except Exception:
                                pass

                            # Commit after label upserts as well
                            try:
                                if db_sess is not None:
                                    db_sess.commit()
                            except Exception:
                                try:
                                    if db_sess is not None:
                                        db_sess.rollback()
                                except Exception:
                                    pass

                            if any_upsert:
                                try:
                                    if (os.getenv("NIR_SAVE_ANOMALIES", "1") or "").lower() not in ("0","off","false","no"):
                                        if _timing:
                                            _logger.info("step:anomalies start symbol=%s", sym)
                                        if (
                                            isinstance(payload, dict)
                                            and isinstance(payload.get("anomalies_report"), dict)
                                        ):
                                            save_cvar_result(
                                                sym,
                                                {
                                                    "as_of_date": as_of_date.isoformat(),
                                                    "data_summary": {},
                                                    "cvar50": {},
                                                    "cvar95": {},
                                                    "cvar99": {},
                                                    "anomalies_report": payload.get(
                                                        "anomalies_report"
                                                    ),
                                                },
                                            )
                                        if _timing:
                                            _logger.info("step:anomalies done symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)
                                except Exception:
                                    if _timing:
                                        _logger.info("step:anomalies error symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)
                                    pass
                                try:
                                    _logger.info(
                                        "SB upserted: symbol=%s as_of=%s alpha=%s",
                                        sym,
                                        as_of,
                                        (
                                            payload.get("alpha")
                                            if "alpha" in payload
                                            else ""
                                        ),
                                    )
                                except Exception:
                                    _logger.info("SB upserted: symbol=%s as_of=%s", sym, as_of)
                                # Brief success summary
                                try:
                                    def _fmt3(v: object) -> str:
                                        try:
                                            return f"{float(v):.3f}"
                                        except Exception:
                                            return "-"
                                    nig = payload.get("cvar_nig")
                                    ghst = payload.get("cvar_ghst")
                                    evar = payload.get("cvar_evar")
                                    blk = payload.get(f"cvar{alpha_label}")
                                    if (nig is None or ghst is None or evar is None) and isinstance(blk, dict):
                                        ann = blk.get("annual") if isinstance(blk.get("annual"), dict) else {}
                                        nig = ann.get("nig") if nig is None else nig
                                        ghst = ann.get("ghst") if ghst is None else ghst
                                        evar = ann.get("evar") if evar is None else evar
                                    _logger.info(
                                        "result: ok symbol=%s as_of=%s alpha=%d nig=%s ghst=%s evar=%s",
                                        sym,
                                        as_of,
                                        int(alpha_label),
                                        _fmt3(nig),
                                        _fmt3(ghst),
                                        _fmt3(evar),
                                    )
                                except Exception:
                                    pass
                                try:
                                    with _counters_lock:
                                        globals()["_sb_success_events"] = globals().get("_sb_success_events", 0) + 1
                                except Exception:
                                    pass
                                _push_recent({"symbol": sym, "as_of": as_of, "type": "success"})
                                try:
                                    sess = db_sess or get_db_session()
                                    if sess is not None:
                                        from core.models import Symbols

                                        row = (
                                            sess.query(Symbols)
                                            .filter(Symbols.symbol == sym)
                                            .one_or_none()
                                        )
                                        if row is not None:
                                            row.insufficient_history = 0
                                            sess.commit()
                                except Exception:
                                    pass
                            else:
                                yrs = payload.get("years")
                                miny = payload.get("min_years")
                                if _log_skips_enabled():
                                    _logger.info(
                                        "SB event skipped: symbol=%s status=%s code=%s years=%s min_years=%s",
                                        sym,
                                        status or "",
                                        code or "",
                                        (f"{float(yrs):.2f}" if yrs is not None else ""),
                                        (f"{float(miny):.2f}" if miny is not None else ""),
                                    )
                                if status or code:
                                    try:
                                        with _counters_lock:
                                            globals()["_sb_error_events"] = globals().get("_sb_error_events", 0) + 1
                                            if code == "insufficient_history":
                                                globals()["_sb_err_insufficient_history"] += 1
                                            elif code in ("insufficient_data", "insufficient_history"):
                                                globals()["_sb_err_insufficient_data"] += 1
                                                try:
                                                    raw_entry = {
                                                        "symbol": sym,
                                                        "as_of": as_of,
                                                        # reflect the code we saw, and include original payload
                                                        "code": code,
                                                        "raw": payload,
                                                        "diag": payload.get("diag") if isinstance(payload, dict) else None,
                                                    }
                                                    _sb_insufficient_data_raw.append(raw_entry)
                                                    if len(_sb_insufficient_data_raw) > 100:
                                                        del _sb_insufficient_data_raw[0 : len(_sb_insufficient_data_raw) - 100]
                                                    insert_insufficient_data_event(raw_entry)
                                                except Exception:
                                                    pass
                                            elif code == "calc_failed":
                                                globals()["_sb_err_calc_failed"] += 1
                                            else:
                                                globals()["_sb_err_other"] += 1
                                    except Exception:
                                        pass
                                    _push_recent({"symbol": sym, "as_of": as_of, "type": "error", "code": code or ""})
                                    # Brief error summary
                                    try:
                                        emsg = payload.get("error") or payload.get("message") or ""
                                        _logger.info(
                                            "result: error symbol=%s as_of=%s code=%s msg=%s",
                                            sym,
                                            as_of,
                                            (code or status or ""),
                                            str(emsg)[:120],
                                        )
                                        
                                        # Check if this is a temporary API error that should be retried
                                        retry_error = False
                                        max_retries = int(os.getenv("CVAR_MAX_RETRIES", "3"))
                                        
                                        # Get current retry count from message properties
                                        retry_count = 0
                                        try:
                                            if hasattr(msg, "application_properties") and msg.application_properties:
                                                retry_count = int(msg.application_properties.get("retry_count", 0))
                                        except Exception:
                                            retry_count = 0
                                            
                                        # Determine if error is retryable
                                        if any(err_text in str(emsg).lower() for err_text in [
                                            "max retries exceeded",
                                            "timeout",
                                            "connection error",
                                            "retries exhausted",
                                            "rate limit",
                                            "too many requests",
                                            "service unavailable",
                                            "internal server error",
                                            "502",
                                            "503",
                                            "504"
                                        ]):
                                            retry_error = True
                                        
                                        # Special case: handle 404 errors for Canadian and UK symbols
                                        if "404" in str(emsg).lower() and "provider returned 404" in str(emsg).lower():
                                            # Check if this is a Canadian or UK symbol
                                            from core.db import get_db_session
                                            from core.models import Symbols
                                            
                                            try:
                                                session = get_db_session()
                                                symbol_info = session.query(Symbols).filter(Symbols.symbol == sym).first()
                                                
                                                # Get symbol info from database
                                                if symbol_info:
                                                    # For Canadian symbols add .TO suffix
                                                    if symbol_info.country in ["CA", "Canada"]:
                                                        _logger.info(f"Detected 404 for Canadian symbol {sym}, will retry with .TO suffix")
                                                        retry_error = True
                                                        
                                                        # Modify the original message body to include exchange=TO
                                                        if hasattr(msg, "body") and msg.body:
                                                            try:
                                                                body_data = _json.loads(msg.body.decode('utf-8'))
                                                                body_data["exchange"] = "TO"
                                                                # We'll use this modified body when requeuing
                                                            except Exception:
                                                                pass
                                                    
                                                    # For UK symbols add .LSE suffix
                                                    elif symbol_info.country in ["UK", "United Kingdom", "GB", "Great Britain"]:
                                                        _logger.info(f"Detected 404 for UK symbol {sym}, will retry with .LSE suffix")
                                                        retry_error = True
                                                        
                                                        # Modify the original message body to include exchange=LSE
                                                        if hasattr(msg, "body") and msg.body:
                                                            try:
                                                                body_data = _json.loads(msg.body.decode('utf-8'))
                                                                body_data["exchange"] = "LSE"
                                                                # We'll use this modified body when requeuing
                                                            except Exception:
                                                                pass
                                                    
                                                    # For any other symbol with valid=1 that's getting a 404, mark it as invalid
                                                    elif symbol_info.valid == 1:
                                                        _logger.warning(f"Symbol {sym} is marked as valid=1 but got 404 error. Marking as invalid.")
                                                        try:
                                                            # Update validation flags to mark symbol as invalid
                                                            from services.validation_integration import process_ticker_validation
                                                            validation_data = {
                                                                "success": False,
                                                                "code": "insufficient_data",
                                                                "error": f"Provider returned 404 for {sym}"
                                                            }
                                                            process_ticker_validation(
                                                                symbol=sym,
                                                                validation_data=validation_data,
                                                                country=symbol_info.country
                                                            )
                                                        except Exception as ve:
                                                            _logger.error(f"Failed to update validation flags for {sym}: {ve}")
                                            except Exception as ex:
                                                _logger.warning(f"Error checking symbol country for {sym}: {ex}")
                                            finally:
                                                session.close()
                                            
                                        # Requeue if retryable and under max retries
                                        if retry_error and retry_count < max_retries:
                                            try:
                                                from services.infrastructure.azure_service_bus_client import AzureServiceBusClient, QueueMessage, MessagePriority
                                                
                                                # Increment retry count
                                                retry_count += 1
                                                
                                                # Create new message with same data but incremented retry count
                                                from utils.service_bus import sb_connection_string, _sb_results_queue
                                                
                                                # Get connection string from environment
                                                conn_str = sb_connection_string()
                                                queue_name = _sb_results_queue()
                                                
                                                if not conn_str:
                                                    _logger.error("Cannot retry: Service Bus connection string not configured")
                                                    # Try direct retry without Service Bus
                                                    _logger.info(f"Attempting direct retry for symbol {sym} with exchange suffix")
                                                    # TODO: Implement direct retry mechanism
                                                    # For now, just log the attempt
                                                    _logger.info(f"Direct retry not implemented yet for {sym}")
                                                    return
                                                    
                                                sb_client = AzureServiceBusClient(connection_string=conn_str, default_queue=queue_name)
                                                if sb_client.connect():
                                                    # Extract original message body
                                                    body_data = {}
                                                    try:
                                                        if hasattr(msg, "body"):
                                                            body_data = json.loads(msg.body.decode('utf-8'))
                                                            
                                                            # If we detected a symbol with 404 error, ensure we add appropriate exchange suffix
                                                            if "404" in str(emsg).lower() and sym and body_data.get("symbol") == sym:
                                                                from core.models import Symbols
                                                                symbol_info = session.query(Symbols).filter(Symbols.symbol == sym).first()
                                                                if symbol_info and symbol_info.country in ["CA", "Canada"]:
                                                                    body_data["exchange"] = "TO"
                                                                    _logger.info(f"Added exchange=TO to requeued message for Canadian symbol {sym}")
                                                                elif symbol_info and symbol_info.country in ["UK", "United Kingdom", "GB", "Great Britain"]:
                                                                    body_data["exchange"] = "LSE"
                                                                    _logger.info(f"Added exchange=LSE to requeued message for UK symbol {sym}")
                                                    except Exception:
                                                        body_data = {"symbol": sym}
                                                    
                                                    # Create new message with retry metadata
                                                    new_message = QueueMessage(
                                                        body=body_data,
                                                        priority=MessagePriority.MEDIUM,
                                                        metadata={"retry_count": retry_count, "original_error": str(emsg)[:100]}
                                                    )
                                                    
                                                    # Add delay before retry (exponential backoff)
                                                    import datetime
                                                    backoff_minutes = min(5 * (2 ** (retry_count - 1)), 60)  # Max 60 minutes
                                                    new_message.scheduled_enqueue_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=backoff_minutes)
                                                    
                                                    # Send to queue
                                                    if queue_name and sb_client.send_message(new_message, queue_name):
                                                        _logger.info(
                                                            "Requeued symbol=%s for retry %d/%d with %d minute delay",
                                                            sym, retry_count, max_retries, backoff_minutes
                                                        )
                                            except Exception as re:
                                                _logger.warning("Failed to requeue symbol=%s: %s", sym, str(re))
                                    except Exception:
                                        pass
                                    # Persist anomaly report if provided on error events
                                    try:
                                        if isinstance(payload.get("anomalies_report"), dict):
                                            from core.persistence import save_cvar_result
                                            save_cvar_result(
                                                sym,
                                                {
                                                    "as_of_date": as_of,
                                                    "anomalies_report": payload.get("anomalies_report"),
                                                },
                                            )
                                    except Exception:
                                        pass
                                    try:
                                        if code in ("insufficient_history", "insufficient_data"):
                                            sess = db_sess or get_db_session()
                                            if sess is not None:
                                                from core.models import Symbols
                                                from services.validation_integration import process_ticker_validation

                                                row = (
                                                    sess.query(Symbols)
                                                    .filter(Symbols.symbol == sym)
                                                    .one_or_none()
                                                )
                                                if row is not None:
                                                    # Process detailed validation flags
                                                    try:
                                                        validation_data = {
                                                            "success": False,
                                                            "code": code,
                                                            "error": payload.get("error") if isinstance(payload, dict) else str(payload),
                                                            "diag": payload.get("diag") if isinstance(payload, dict) else None
                                                        }
                                                        
                                                        # Include additional data if available
                                                        if isinstance(payload, dict):
                                                            if "years" in payload:
                                                                validation_data["years"] = payload["years"]
                                                            if "min_years" in payload:
                                                                validation_data["min_years"] = payload["min_years"]
                                                            if "anomalies_report" in payload:
                                                                validation_data["anomalies_report"] = payload["anomalies_report"]
                                                        
                                                        # Process with detailed ValidationFlags
                                                        process_ticker_validation(
                                                            symbol=sym,
                                                            validation_data=validation_data,
                                                            country=row.country,
                                                            db_session=sess
                                                        )
                                                    except Exception as e:
                                                        # Fallback to basic insufficient_history flag
                                                        _logger.warning(f"ValidationFlags processing failed for {sym}, using fallback: {e}")
                                                        row.insufficient_history = 1
                                                        sess.commit()
                                    except Exception:
                                        pass
                                    finally:
                                        try:
                                            if db_sess is not None:
                                                db_sess.close()
                                        except Exception:
                                            pass
                                else:
                                    try:
                                        with _counters_lock:
                                            globals()["_sb_empty_events"] = globals().get("_sb_empty_events", 0) + 1
                                    except Exception:
                                        pass
                                    _push_recent({"symbol": sym, "as_of": as_of, "type": "empty"})
                                    # Brief empty summary
                                    try:
                                        _logger.info(
                                            "result: empty symbol=%s as_of=%s",
                                            sym,
                                            as_of,
                                        )
                                    except Exception:
                                        pass

                            try:
                                if _timing:
                                    _logger.info("step:ack start symbol=%s", sym)
                                receiver.complete_message(msg)
                                if _timing:
                                    _logger.info("step:ack done symbol=%s elapsed_ms=%.1f", sym, ( _t.perf_counter()-_t0)*1000)
                            except Exception as exc:
                                try:
                                    _logger.warning(
                                        "SB complete_message failed: %s; will rely on redelivery",
                                        exc,
                                    )
                                except Exception:
                                    pass
                            finally:
                                try:
                                    if db_sess is not None:
                                        db_sess.close()
                                except Exception:
                                    pass
        except Exception:
            try:
                _logger.warning("SB consumer error; will retry")
                __t.sleep(1.0)
            except Exception:
                pass
            continue


