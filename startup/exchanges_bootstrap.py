from __future__ import annotations

import logging

from services.exchanges_sync import sync_exchanges_once, should_sync_exchanges

_logger = logging.getLogger("startup.exchanges")


def bootstrap_exchanges_if_needed(db_ready: bool) -> int:
    """Bootstrap exchanges from EODHD API if needed.
    
    Runs exchange sync if:
    - EODHD_SYNC_EXCHANGES env var is set to 1/true/yes, OR
    - Exchange table is empty
    
    Args:
        db_ready: Whether the database is ready for operations
        
    Returns:
        Number of exchanges synchronized, 0 if skipped
    """
    if not db_ready:
        _logger.info("Database not ready, skipping exchanges bootstrap")
        return 0
    
    try:
        if should_sync_exchanges():
            _logger.info("Starting exchanges bootstrap from EODHD")
            count = sync_exchanges_once(force=False)
            _logger.info("Exchanges bootstrap completed, synced %d exchanges", count)
            return count
        else:
            _logger.info("Exchanges bootstrap skipped (table not empty and env flag not set)")
            return 0
            
    except Exception as e:
        _logger.error("Exchanges bootstrap failed: %s", str(e))
        return 0
