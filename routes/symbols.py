from fastapi import (  # type: ignore
    APIRouter,
    HTTPException,
    Query,
    Depends,
    BackgroundTasks,
)
from utils.auth import require_pub_or_basic as _require_pub_or_basic
from core.db import get_db_session
from core.models import PriceSeries, InstrumentAlias
from sqlalchemy import or_  # type: ignore
from sqlalchemy.sql import func  # type: ignore
from services.symbols_sync import sync_symbols_once
from services.domain.cvar_unified_service import CvarUnifiedService
from utils import service_bus as _sb
import os
import requests  # type: ignore
import time
import logging
from utils.common import parse_exchanges_env as _parse_exchanges_env


router = APIRouter()


@router.get("/tickers")
def list_tickers(_auth: None = Depends(_require_pub_or_basic)) -> dict:
    sess = get_db_session()
    if sess is None:
        return {"items": []}
    try:
        rows = (
            sess.query(PriceSeries)
            .filter(PriceSeries.insufficient_history == 0)  # type: ignore
            .order_by(PriceSeries.symbol.asc())  # type: ignore
            .all()
        )
        items = [
            {"symbol": r.symbol, "name": r.name or r.symbol}
            for r in rows
        ]
        return {"items": items}
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/symbols/search")
def symbols_search(
    q: str,
    limit: int = 10,
    ready_only: bool = True,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """Search symbols by exact/partial match across symbol, name,
    alternative names.

    - Case-insensitive match on symbol and name
    - JSON alternative_names matched via text search when possible
    - Optional readiness filter (insufficient_history == 0)
    """
    query = (q or "").strip()
    if not query:
        return {"items": []}
    sess = get_db_session()
    if sess is None:
        return {"items": []}
    try:
        # Normalize for case-insensitive contains
        needle = f"%{query.lower()}%"
        base = sess.query(PriceSeries)
        if ready_only:
            base = base.filter(
                PriceSeries.insufficient_history == 0  # type: ignore
            )

        # Build filters: symbol ILIKE, name ILIKE, alternative_names contains
        filters = [
            func.lower(PriceSeries.symbol).like(needle),  # type: ignore
            func.lower(PriceSeries.name).like(needle),  # type: ignore
        ]

        # Prefetch window for ranking:
        # fetch more rows, rank in Python, then cut to limit
        try:
            prefetch = int(limit)
        except Exception:
            prefetch = 10
        prefetch = max(20, min(50, prefetch))

        # Try JSONB contains on Postgres; fallback to LIKE on text (SQLite)
        try:
            if sess.bind and sess.bind.dialect.name in (
                "postgresql",
                "postgres",
            ):
                # alternative_names is JSONB array of strings;
                # check any element ilike via jsonb_array_elements_text
                from sqlalchemy import text as _text  # type: ignore

                sql = _text(
                    (
                        "SELECT id FROM price_series "
                        "WHERE (:ready_only = 0 OR insufficient_history = 0) "
                        "AND (lower(symbol) LIKE :needle OR "
                        "lower(COALESCE(name, '')) LIKE :needle OR "
                        "EXISTS (SELECT 1 FROM jsonb_array_elements_text("
                        "alternative_names) AS x(val) "
                        "WHERE lower(val) LIKE :needle) OR "
                        "EXISTS (SELECT 1 FROM instrument_alias ia "
                        "WHERE ia.instrument_id = price_series.id AND "
                        "lower(ia.alias) LIKE :needle)) "
                        "LIMIT :prefetch"
                    )
                )
                rows = sess.execute(
                    sql,
                    {
                        "needle": needle,
                        "prefetch": int(max(1, prefetch)),
                        "ready_only": 1 if ready_only else 0,
                    },
                ).fetchall()
                ids = [r[0] for r in rows]
                if ids:
                    recs = (
                        sess.query(PriceSeries)
                        .filter(
                            PriceSeries.id.in_(ids)  # type: ignore[arg-type]
                        )
                        .all()
                    )
                else:
                    recs = []
            else:
                # SQLite/text fallback: join aliases, LIKE across fields
                base = base.outerjoin(
                    InstrumentAlias,
                    InstrumentAlias.instrument_id == PriceSeries.id,
                )
                try:
                    alt_like = func.lower(
                        func.coalesce(
                            PriceSeries.alternative_names, ""
                        )
                    ).like(needle)  # type: ignore
                    filters.append(alt_like)
                except Exception:
                    pass
                try:
                    alias_like = func.lower(InstrumentAlias.alias).like(
                        needle
                    )  # type: ignore
                    filters.append(alias_like)
                except Exception:
                    pass
                recs = (
                    base.filter(or_(*filters))
                    .order_by(PriceSeries.symbol.asc())  # type: ignore
                    .limit(int(max(1, prefetch)))
                    .all()
                )
        except Exception:
            recs = (
                base.filter(or_(*filters))
                .order_by(PriceSeries.symbol.asc())  # type: ignore
                .limit(int(max(1, prefetch)))
                .all()
            )

        # Rank results:
        # exact=0, prefix=1, contains=2 (symbol prioritized over name/alias),
        # shorter name wins
        ql = query.lower()

        def _rank(rec: PriceSeries) -> tuple:
            try:
                sy = (rec.symbol or "").lower()
                nm = (rec.name or "").lower()

                def h(t: str) -> int:
                    if not t:
                        return 9
                    if t == ql:
                        return 0
                    if t.startswith(ql):
                        return 1
                    if ql in t:
                        return 2
                    return 9
                s_score = h(sy)
                n_score = h(nm)
                overall = min(s_score, n_score)
                return (overall, len(nm or sy), sy)
            except Exception:
                return (9, 9999, (rec.symbol or ""))

        recs_sorted = sorted(recs, key=_rank)[: int(max(1, limit))]
        items = [
            {
                "symbol": r.symbol,
                "name": (r.name or r.symbol),
                "country": r.country,
                "exchange": r.exchange,
            }
            for r in recs_sorted
        ]
        return {"items": items}
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.post("/symbols/sync")
@router.get("/symbols/sync")
def symbols_sync(
    exchanges: str | None = Query(
        None,
        description="Comma-separated exchanges to override env (e.g., US,CA)",
    ),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    if exchanges:
        import os
        os.environ["EODHD_EXCHANGES"] = exchanges
    try:
        n = sync_symbols_once(force=True)
    except Exception as exc:
        raise HTTPException(502, f"sync failed: {exc}")
    return {"processed": int(n)}


def _eodhd_search_best_us(sym: str) -> dict | None:
    api_key = (
        os.getenv("EODHD_API_TOKEN", "")
        or os.getenv("EODHD_API_KEY", "")
    )
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"https://eodhistoricaldata.com/api/search/{sym}",
            params={"api_token": api_key, "limit": 20},
            timeout=15,
        )
        if not resp.ok:
            return None
        data = resp.json() if resp.content else []
    except Exception:
        return None
    best = None
    for row in data or []:
        co = str(row.get("Country") or "").lower()
        code = str(row.get("Code") or "").upper()
        if co in ("us", "usa", "united states") and (
            code == sym
            or code.startswith(sym + ".")
            or code.startswith(sym + ":")
        ):
            best = row
            break
    return best


@router.post("/symbols/resync_and_recalc")
@router.get("/symbols/resync_and_recalc")
def symbols_resync_and_recalc(
    symbols: str | None = Query(
        None,
        description=(
            "Comma-separated symbols; if omitted, resync all US from EODHD"
        ),
    ),
    background_tasks: BackgroundTasks = None,
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """Re-enrich US symbols from EODHD and batch-recompute CVaR.

    Query: symbols=comma,separated
    """
    syms = [s.strip().upper() for s in (symbols or "").split(",") if s.strip()]
    # If no explicit list, fetch full US universe and build reconciliation set
    if not syms:
        exchanges = _parse_exchanges_env()
        if not exchanges:
            exchanges = ["US"]
        # fetch symbols for US only
        if "US" not in exchanges:
            exchanges = ["US"]
        try:
            from services.symbols_sync import (
                _fetch_symbols_from_eodhd,
            )  # type: ignore
        except Exception:
            _fetch_symbols_from_eodhd = None  # type: ignore
        fetched = []
        if _fetch_symbols_from_eodhd is not None:
            fetched = _fetch_symbols_from_eodhd(exchanges) or []
        syms = sorted(
            {
                str(it.get("Code") or "")
                .strip()
                .upper()
                for it in fetched
                if it.get("Code")
            }
        )

    def _job(job_syms: list[str]) -> None:
        logger = logging.getLogger("symbols_resync")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s symbols_resync %(levelname)s: %(message)s"
                )
            )
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info("start: symbols=%d", len(job_syms))
        sess = get_db_session()
        if sess is None:
            return
        created_us = 0
        updated_us = 0
        try:
            for idx, sym in enumerate(job_syms, start=1):
                best = _eodhd_search_best_us(sym)
                # Work with the US row specifically;
                # do not overwrite other countries
                rec = (
                    sess.query(PriceSeries)
                    .filter(
                        PriceSeries.symbol == sym,
                        PriceSeries.country == "US",
                    )
                    .one_or_none()
                )
                is_new = False
                if rec is None:
                    rec = PriceSeries(symbol=sym, country="US")
                    is_new = True
                try:
                    if isinstance(best, dict):
                        rec.name = str(best.get("Name") or rec.name or sym)
                        rec.exchange = str(
                            best.get("Exchange") or rec.exchange or ""
                        )
                        rec.currency = str(
                            best.get("Currency") or rec.currency or ""
                        )
                        rec.instrument_type = str(
                            best.get("Type") or rec.instrument_type or ""
                        )
                        rec.country = "US"
                    else:
                        rec.country = rec.country or "US"
                except Exception:
                    rec.country = rec.country or "US"
                sess.merge(rec)
                if is_new:
                    created_us += 1
                else:
                    updated_us += 1
                if (idx % 500) == 0:
                    sess.commit()
                    logger.info(
                        "progress: %d/%d (created_us=%d updated_us=%d)",
                        idx,
                        len(job_syms),
                        created_us,
                        updated_us,
                    )
            sess.commit()
        except Exception:
            try:
                sess.rollback()
            except Exception:
                pass
        finally:
            try:
                sess.close()
            except Exception:
                pass
        logger.info(
            "db done: created_us=%d updated_us=%d", created_us, updated_us
        )

        # Enqueue CVaR recalculation in batches if SB configured;
        # otherwise compute locally
        conn = _sb.sb_connection_string()
        qname = _sb.sb_queue_name()
        if conn and qname:
            import json as _json
            try:
                from azure.servicebus import (  # type: ignore
                    ServiceBusClient,
                    ServiceBusMessage,
                )
                batch_size = max(50, int(os.getenv("SB_BATCH", "100")))
                chunk_size = max(
                    50,
                    int(os.getenv("SB_SYMBOLS_PER_MSG", "100")),
                )
                submitted = 0
                with ServiceBusClient.from_connection_string(conn) as client:
                    sender = client.get_queue_sender(queue_name=qname)
                    with sender:
                        pending: list[ServiceBusMessage] = []
                        for i in range(0, len(job_syms), chunk_size):
                            chunk = job_syms[i:i + chunk_size]
                            for sym in chunk:
                                body = {
                                    "symbol": sym,
                                    "alphas": [0.99, 0.95, 0.50],
                                    "force": True,
                                }
                                pending.append(
                                    ServiceBusMessage(_json.dumps(body))
                                )
                                submitted += 3
                                if len(pending) >= batch_size:
                                    sender.send_messages(pending)
                                    logger.info(
                                        "enqueue: batch=%d submitted=%d",
                                        len(pending),
                                        submitted,
                                    )
                                    pending.clear()
                        if pending:
                            sender.send_messages(pending)
                            logger.info(
                                "enqueue: final batch=%d submitted=%d",
                                len(pending),
                                submitted,
                            )
                logger.info(
                    "enqueue done: symbols=%d submitted=%d",
                    len(job_syms),
                    submitted,
                )
            except Exception as exc:
                logger.warning(
                    "enqueue failed; fallback local compute: %s", exc
                )
                svc = CvarUnifiedService()
                for sym in job_syms:
                    try:
                        svc.get_cvar_data(sym, force_recalculate=True)
                        time.sleep(0.01)
                    except Exception:
                        continue
        else:
            logger.info("SB disabled; running local compute")
            svc = CvarUnifiedService()
            for sym in job_syms:
                try:
                    svc.get_cvar_data(sym, force_recalculate=True)
                    time.sleep(0.01)
                except Exception:
                    continue
        logger.info(
            "done: symbols=%d created_us=%d updated_us=%d",
            len(job_syms),
            created_us,
            updated_us,
        )

    # Initialize background task runner if not provided (some test contexts)
    if background_tasks is None:
        from fastapi import BackgroundTasks as _BT  # type: ignore
        background_tasks = _BT()
    background_tasks.add_task(_job, syms)
    return {"status": "Request sent"}
