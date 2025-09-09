import os
import logging
from typing import List, Dict

from core.db import get_db_session
from core.persistence import upsert_symbols_bulk
from utils.common import parse_exchanges_env as _parse_exchanges_env
from utils.common import should_include_instrument_type, should_exclude_exchange


_sym_logger = logging.getLogger("symbols_sync")


def _fetch_symbols_from_eodhd(exchanges: List[str]) -> List[Dict]:
    try:
        import requests  # type: ignore
    except Exception:
        return []
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        return []
    items: List[Dict] = []
    for ex in exchanges:
        try:
            _sym_logger.info("fetching exchange=%s", ex)
            url = (
                "https://eodhistoricaldata.com/api/"
                f"exchange-symbol-list/{ex}"
            )
            resp = requests.get(
                url,
                params={"api_token": api_key, "fmt": "json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                _sym_logger.info(
                    "fetched items=%d exchange=%s", len(data), ex
                )
                for it in data:
                    if isinstance(it, dict):
                        # Get instrument type for filtering
                        instrument_type = (
                            it.get("Type")
                            or it.get("TypeCode")
                            or it.get("type")
                        )
                        
                        # Get exchange for filtering
                        exchange = it.get("Exchange") or ex
                        
                        # Apply instrument type filter (ETF & Mutual Fund only)
                        if not should_include_instrument_type(instrument_type):
                            continue
                        
                        # Apply exchange exclusion filter (exclude PINK exchange)
                        if should_exclude_exchange(exchange):
                            _sym_logger.debug(
                                "Excluding symbol %s from exchange %s", 
                                it.get("Code", "UNKNOWN"), 
                                exchange
                            )
                            continue
                        
                        items.append(
                            {
                                "Code": it.get("Code"),
                                "Name": it.get("Name"),
                                "Country": it.get("Country"),
                                "Exchange": exchange,
                                "Currency": it.get("Currency"),
                                "Type": instrument_type,
                                "Isin": it.get("Isin") or it.get("ISIN"),
                            }
                        )
        except Exception:
            _sym_logger.warning("fetch failed exchange=%s", ex)
            continue
    return items


def sync_symbols_once(force: bool = False) -> int:
    """Fetch EODHD symbols for configured exchanges and upsert into DB.

    - If force is False, skip when API key missing or DB unavailable
    - Returns number of processed records
    """
    sess = get_db_session()
    if sess is None:
        return 0
    try:
        exchanges = _parse_exchanges_env()
        items = _fetch_symbols_from_eodhd(exchanges)
        if not items:
            _sym_logger.info("no items fetched; exchanges=%s", exchanges)
            return 0
        n = upsert_symbols_bulk(items)
        _sym_logger.info(
            "upserted items=%d exchanges=%s", int(n), exchanges
        )
        return int(n)
    finally:
        try:
            sess.close()
        except Exception:
            pass


