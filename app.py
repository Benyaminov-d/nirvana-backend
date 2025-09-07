"""
Nirvana App API application setup.

- Registers route modules from `backend/routes/*`
- Configures static assets  
- Starts Service Bus consumer and nightly worker on startup
"""

import os
from pathlib import Path
import random
from datetime import datetime
import time as _time
import threading
import logging

from dotenv import load_dotenv  # type: ignore
from fastapi import FastAPI  # type: ignore
from fastapi import HTTPException  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.responses import RedirectResponse  # type: ignore
from fastapi.responses import PlainTextResponse  # type: ignore

from services.domain.cvar_unified_service import CvarUnifiedService
from core.models import PriceSeries
# AnnualCvarViolation model is used via persistence helpers; no direct use here
from core.db import get_db_session
from utils.common import (
    seconds_until as _seconds_until,
)
from core import state as _state
from services.symbols_sync import sync_symbols_once
from services.sb_consumer import (
    start_consumer_loop as _sb_start_loop,
)

from routes.annual import router as annual_router
from routes.cvar import router as cvar_router
from routes.metrics import router as metrics_router
from routes.persistence import router as persistence_router
from routes.ticker import router as ticker_router
from routes.timeseries import router as timeseries_router
from routes.export import router as export_router
from routes.explain import router as explain_router
from routes.symbols import router as symbols_router
from routes.products import router as products_router
# Compass router removed - proprietary API moved to compiled library
from routes.experiments import router as experiments_router
from routes.auth import router as auth_router
from routes.demo import router as demo_router
from routes.contact import router as contact_router
from routes.validation_analytics import router as validation_analytics_router
from routes.ticker_check import router as ticker_check_router
from routes.rag import router as rag_router
from routes.universe import router as universe_router
from routes.cvar_refactored_demo import router as cvar_refactored_router
from routes.ticker_refactored import router as ticker_refactored_router
from routes.ticker_country_specific import router as ticker_country_specific_router
from routes.refactoring_tools import router as refactoring_tools_router
from routes.application_services_demo import router as application_services_demo_router
from routes.infrastructure_demo import router as infrastructure_demo_router
from routes.data_access_demo import router as data_access_demo_router
from routes.shared_demo import router as shared_demo_router
from routes.domain_models_demo import router as domain_models_demo_router
from routes.debug_database import router as debug_database_router

# ───────────────────── env / init ─────────────────────
load_dotenv()
os.environ.setdefault("NVAR_LICENSE", "DEV")
# Enforce Lambert parity defaults at app startup (no app-side math)
os.environ.setdefault("NIR_LAMBERT_STRICT", "0")
os.environ.setdefault("NVAR_BOOTSTRAP_THRESHOLD", "0")
os.environ.setdefault("NVAR_BOOTSTRAP_BLOCK", "100")
os.environ.setdefault("NVAR_SYMBOLS_FILTER", "ready")

app = FastAPI(
    title="Nirvana App",
    version="1.0.0",
    description="Financial risk analysis and portfolio management platform",
)

app.include_router(annual_router, prefix="/api")
app.include_router(cvar_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(persistence_router, prefix="/api")
app.include_router(ticker_router, prefix="/api")
app.include_router(timeseries_router, prefix="/api")
app.include_router(symbols_router, prefix="/api")
app.include_router(export_router, prefix="/api")
app.include_router(products_router, prefix="/api")
app.include_router(explain_router, prefix="/api")
# app.include_router(compass_router, prefix="/api")  # Removed - proprietary
app.include_router(experiments_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(demo_router, prefix="/api")
app.include_router(contact_router, prefix="/api")
app.include_router(validation_analytics_router, prefix="/api")
app.include_router(rag_router)
app.include_router(ticker_check_router)
app.include_router(universe_router, prefix="/api")
app.include_router(cvar_refactored_router, prefix="/api")
app.include_router(ticker_refactored_router, prefix="/api")
app.include_router(ticker_country_specific_router, prefix="/api")
app.include_router(refactoring_tools_router, prefix="/api")
app.include_router(application_services_demo_router, prefix="/api")
app.include_router(infrastructure_demo_router, prefix="/api")
app.include_router(data_access_demo_router, prefix="/api")
app.include_router(shared_demo_router, prefix="/api")
app.include_router(domain_models_demo_router, prefix="/api")
app.include_router(debug_database_router, prefix="/api")

# Static files (HTML, PDFs, images) are now served by the frontend container
# under /spa/public/static/ - backend no longer serves static content

# No /assets mounting here; assets are served by the frontend container.
# Optional: if SPA build is present alongside backend (dev fallback),
# expose it under /spa
FRONT = Path(__file__).parent.parent / "frontend"
SPA_DIST = FRONT / "spa" / "dist"
if SPA_DIST.exists():
    app.mount("/spa", StaticFiles(directory=SPA_DIST), name="spa")


# Convenience redirects for legacy UI paths without /api prefix
@app.get("/cvar")
def _redir_cvar() -> RedirectResponse:
    return RedirectResponse(url="/api/cvar", status_code=307)


@app.get("/cvar/calculator")
def _redir_cvar_calc() -> RedirectResponse:
    return RedirectResponse(url="/api/cvar/calculator", status_code=307)


# Legal content endpoints  
_ROOT = Path(__file__).parent
_EULA_MD = _ROOT / "rag" / "member_eula.md"


@app.get("/api/member-eula.md", response_class=PlainTextResponse)
def member_eula_markdown() -> PlainTextResponse:
    try:
        if _EULA_MD.exists():
            return PlainTextResponse(_EULA_MD.read_text(encoding="utf-8"))
    except Exception:
        pass
    raise HTTPException(404, "member EULA markdown not found")


# Ticker infra
_nightly_thread: threading.Thread | None = None
# import for side effects elsewhere; keep module-level state implicit
_symbols_last_sync: str | None = None
_sb_consumer_thread: threading.Thread | None = None
_sb_logger = logging.getLogger("sb_consumer")
if not _sb_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s sb_consumer %(levelname)s: %(message)s"
        )
    )
    _sb_logger.addHandler(_h)
_sb_logger.setLevel(logging.INFO)
_sb_logger.propagate = False

# Symbols sync logger
_sym_logger = logging.getLogger("symbols_sync")
if not _sym_logger.handlers:
    _h2 = logging.StreamHandler()
    _h2.setFormatter(
        logging.Formatter(
            "%(asctime)s symbols_sync %(levelname)s: %(message)s"
        )
    )
    _sym_logger.addHandler(_h2)
_sym_logger.setLevel(logging.INFO)
_sym_logger.propagate = False

# Reduce noisy Azure Service Bus connection-state logs
for _azure_logger_name in (
    "azure",
    "azure.servicebus",
    "azure.servicebus._pyamqp",
    "uamqp",
):
    try:
        logging.getLogger(_azure_logger_name).setLevel(logging.WARNING)
    except Exception:
        pass


def _start_sb_consumer_if_configured() -> None:
    """Start background Service Bus consumer loop using isolated module."""
    try:
        _sb_start_loop()
    except Exception:
        pass


def _db_symbols(
    *,
    ready_only: bool | None = None,
    include_unknown: bool | None = None,
    five_stars: bool = False,
) -> list[str]:
    """Return symbol list from DB with optional filters.

    Filters:
      - ready_only=True: only where insufficient_history == 0
      - include_unknown=True: include NULL (unknown) along with ready
      - five_stars=True: restrict to five_stars==1

    Env fallback when args are None:
      NVAR_SYMBOLS_FILTER = all | ready | ready_or_unknown
    """
    try:
        sess = get_db_session()
        if sess is None:
            raise RuntimeError("no session")
        from core.models import PriceSeries
        # Determine filter mode
        mode = (os.getenv("NVAR_SYMBOLS_FILTER", "all") or "all").lower()
        if ready_only is None and include_unknown is None:
            if mode == "ready":
                ready_only = True
                include_unknown = False
            elif mode in ("ready_or_unknown", "ready_unknown"):
                ready_only = True
                include_unknown = True
            else:
                ready_only = False
                include_unknown = False
        # Base query
        q = sess.query(PriceSeries.symbol)
        if five_stars:
            q = q.filter(PriceSeries.five_stars == 1)  # type: ignore
        # Apply readiness filter
        if ready_only:
            if include_unknown:
                # insufficient_history in (0, NULL)
                q = q.filter(
                    (PriceSeries.insufficient_history == 0)
                    | (PriceSeries.insufficient_history.is_(None))
                )  # type: ignore
            else:
                q = q.filter(
                    PriceSeries.insufficient_history == 0
                )  # type: ignore
        rows = q.all()  # type: ignore
        syms = [s for (s,) in rows]
        if syms:
            return syms
    except Exception:
        pass
    return ["SP500TR", "BTC", "ETH"]


def _build_symbol_pool(n: int) -> list[str]:
    syms = list(_db_symbols())
    random.shuffle(syms)
    return syms[:n] if n > 0 else syms


def _sync_symbols_once(force: bool = False) -> int:
    """Fetch EODHD symbols for configured exchanges and upsert into DB.

    - If force is False, skip when API key missing or DB unavailable
    - Returns number of processed records
    """
    global _symbols_last_sync
    sess = get_db_session()
    if sess is None:
        return 0
    try:
        n = sync_symbols_once(force=True if force else False)
        try:
            _symbols_last_sync = datetime.now().isoformat(
                timespec="seconds"
            )
        except Exception:
            pass
        return int(n)
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _sync_symbols_if_empty() -> None:
    sess = get_db_session()
    if sess is None:
        return
    try:
        has_any = (
            sess
            .query(PriceSeries.id)
            .limit(1)
            .all()
        )
        if not has_any:
            _sym_logger.info("price_series empty; running initial sync")
            _sync_symbols_once(force=True)
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _run_nightly_once(count: int, sleep_between_sec: float) -> None:
    svc = CvarUnifiedService()
    pool = _build_symbol_pool(count)
    for sym in pool:
        try:
            # ensure symbol is tracked for feed
            with _state.ticker_lock:
                _state.submitted_symbols.add(sym)
            # force fresh calc to pull new EOD data
            svc.get_cvar_data(sym, force_recalculate=True)
        except Exception:
            pass
        if sleep_between_sec > 0:
            _time.sleep(sleep_between_sec)
    _state.nightly_last_run = datetime.now().isoformat(timespec="seconds")


def _nightly_thread_main() -> None:
    enabled = os.getenv("NVAR_TICKER_NIGHTLY_ENABLED", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if not enabled:
        return
    try:
        hour = int(os.getenv("NVAR_TICKER_NIGHTLY_HOUR", "2"))
    except Exception:
        hour = 2
    try:
        minute = int(os.getenv("NVAR_TICKER_NIGHTLY_MINUTE", "0"))
    except Exception:
        minute = 0
    try:
        count = int(os.getenv("NVAR_TICKER_NIGHTLY_COUNT", "0"))
    except Exception:
        count = 0
    try:
        sleep_between = float(os.getenv("NVAR_TICKER_NIGHTLY_SLEEP", "1.0"))
    except Exception:
        sleep_between = 1.0

    # loop forever; sleep until next run, then refresh symbols and recalc
    while True:
        secs = _seconds_until(hour, minute)
        secs = max(1, secs)
        _time.sleep(secs)
        # Daily symbol sync from EODHD before nightly recalc
        try:
            _sym_logger.info("nightly symbols sync starting")
            _sync_symbols_once(force=False)
        except Exception:
            _sym_logger.warning("nightly symbols sync failed")
        _run_nightly_once(count, sleep_between)


@app.on_event("startup")
def _start_nightly_worker() -> None:
    # Delegate startup tasks to dedicated module to keep app.py clean
    try:
        from startup import run_startup_tasks
        run_startup_tasks()
    except Exception as exc:
        import logging
        _startup_logger = logging.getLogger(__name__)
        _startup_logger.error("STARTUP FAILED: %s", exc, exc_info=True)

    # Start nightly thread after basic startup
    global _nightly_thread
    if _nightly_thread is None or not _nightly_thread.is_alive():
        t = threading.Thread(target=_nightly_thread_main, daemon=True)
        _nightly_thread = t
        t.start()

    # Start SB topic consumer (persist results into DB and refresh cache)
    global _sb_consumer_thread
    if _sb_consumer_thread is None or not _sb_consumer_thread.is_alive():
        c = threading.Thread(
            target=_start_sb_consumer_if_configured,
            daemon=True,
        )
        _sb_consumer_thread = c
        c.start()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
