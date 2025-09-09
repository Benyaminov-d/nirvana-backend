from __future__ import annotations

from datetime import datetime as _dt
from pathlib import Path as _P

from core.db import get_db_session
from core.models import Symbols, Symbols


def bootstrap_symbols_if_empty(db_ready: bool) -> None:
    """Bootstrap symbols from EODHD if table is empty."""
    if not db_ready:
        return
    sess = get_db_session()
    if sess is None:
        return
    try:
        has_any = sess.query(Symbols.id).limit(1).all()
        if not has_any:
            from app import _sync_symbols_once as _sync_symbols_once  # lazy import
            _sync_symbols_once(force=True)
    finally:
        try:
            sess.close()
        except Exception:
            pass


def bootstrap_eodhd_symbols_by_exchanges(db_ready: bool) -> int:
    """Bootstrap symbols from EODHD for configured exchanges, regardless of table state.
    
    This ensures new exchanges (like LSE) get their symbols loaded even if 
    the price_series table already has data from other sources.
    
    Returns:
        Number of symbols processed
    """
    if not db_ready:
        return 0
    
    import logging
    _logger = logging.getLogger("startup.eodhd_symbols")
    
    try:
        from app import _sync_symbols_once as _sync_symbols_once  # lazy import
        count = _sync_symbols_once(force=True)
        _logger.info("EODHD symbols bootstrap completed, processed %d symbols", count)
        return count
    except Exception as e:
        _logger.error("EODHD symbols bootstrap failed: %s", str(e))
        return 0

