import os
import logging
from typing import List, Dict

from core.db import get_db_session
from core.models import Exchange


_exchange_logger = logging.getLogger("exchanges_sync")


def _fetch_exchanges_from_eodhd() -> List[Dict]:
    """Fetch exchanges list from EODHD API.

    Returns list of exchange dictionaries or empty list on error.
    """
    try:
        import requests  # type: ignore
    except Exception:
        return []

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        _exchange_logger.warning("EODHD_API_KEY not set")
        return []

    try:
        _exchange_logger.info("Fetching exchanges list from EODHD")
        url = "https://eodhistoricaldata.com/api/exchanges-list"
        resp = requests.get(
            url,
            params={"api_token": api_key, "fmt": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, list):
            _exchange_logger.info(
                "Fetched %d exchanges from EODHD", len(data)
            )
            return data
        else:
            _exchange_logger.warning(
                "Unexpected response format from EODHD exchanges API"
            )
            return []
    except Exception as e:
        _exchange_logger.warning(
            "Failed to fetch exchanges from EODHD: %s", str(e)
        )
        return []


def _upsert_exchanges_bulk(exchanges: List[Dict]) -> int:
    """Insert or update exchanges in the database.

    Args:
        exchanges: List of exchange dictionaries from EODHD API

    Returns:
        Number of exchanges processed
    """
    session = get_db_session()
    if session is None:
        return 0
    
    processed = 0
    try:
        from datetime import datetime
        now = datetime.utcnow()

        for exchange_data in exchanges:
            try:
                code = str(exchange_data.get("Code", "")).strip()
                if not code:
                    continue

                # Check if exchange already exists
                existing = (
                    session.query(Exchange)
                    .filter(Exchange.code == code)
                    .one_or_none()
                )
                
                if existing is None:
                    # Create new exchange
                    exchange = Exchange(
                        code=code,
                        name=exchange_data.get("Name"),
                        operating_mic=exchange_data.get("OperatingMIC"),
                        country=exchange_data.get("Country"),
                        currency=exchange_data.get("Currency"),
                        country_iso2=exchange_data.get("CountryISO2"),
                        country_iso3=exchange_data.get("CountryISO3"),
                        created_at=now,
                        updated_at=now,
                    )
                else:
                    # Update existing exchange
                    existing.name = (
                        exchange_data.get("Name") or existing.name
                    )
                    existing.operating_mic = (
                        exchange_data.get("OperatingMIC") or
                        existing.operating_mic
                    )
                    existing.country = (
                        exchange_data.get("Country") or existing.country
                    )
                    existing.currency = (
                        exchange_data.get("Currency") or existing.currency
                    )
                    existing.country_iso2 = (
                        exchange_data.get("CountryISO2") or
                        existing.country_iso2
                    )
                    existing.country_iso3 = (
                        exchange_data.get("CountryISO3") or
                        existing.country_iso3
                    )
                    existing.updated_at = now
                    exchange = existing

                session.merge(exchange)
                processed += 1

                # Commit periodically to reduce transaction size
                if (processed % 50) == 0:
                    session.commit()

            except Exception as e:
                _exchange_logger.warning(
                    "Failed to process exchange %s: %s",
                    exchange_data.get("Code", "unknown"), str(e)
                )
                continue

        session.commit()
        return processed
    except Exception as e:
        _exchange_logger.error("Failed to upsert exchanges: %s", str(e))
        try:
            session.rollback()
        except Exception:
            pass
        return processed
    finally:
        try:
            session.close()
        except Exception:
            pass


def sync_exchanges_once(force: bool = False) -> int:
    """Fetch EODHD exchanges and upsert into DB.

    Args:
        force: If True, sync even if API key missing or DB unavailable

    Returns:
        Number of exchanges processed
    """
    session = get_db_session()
    if session is None and not force:
        return 0

    try:
        exchanges_data = _fetch_exchanges_from_eodhd()
        if not exchanges_data:
            _exchange_logger.info("No exchanges data fetched")
            return 0
        
        count = _upsert_exchanges_bulk(exchanges_data)
        _exchange_logger.info("Upserted %d exchanges", count)
        return count
    except Exception as e:
        _exchange_logger.error("Exchange sync failed: %s", str(e))
        return 0
    finally:
        if session:
            try:
                session.close()
            except Exception:
                pass


def should_sync_exchanges() -> bool:
    """Check if exchanges sync should run.

    Returns True if:
    - EODHD_SYNC_EXCHANGES env var is set to 1/true/yes, OR
    - Exchange table is empty
    """
    # Check env flag first
    sync_flag = os.getenv("EODHD_SYNC_EXCHANGES", "").lower()
    if sync_flag in ("1", "true", "yes"):
        return True

    # Check if table is empty
    session = get_db_session()
    if session is None:
        return False

    try:
        has_any = session.query(Exchange.id).limit(1).all()
        return len(has_any) == 0
    except Exception:
        return False
    finally:
        try:
            session.close()
        except Exception:
            pass
