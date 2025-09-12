"""
Helper functions for price series operations.

This module provides price series helper functions that were previously
in core.persistence.py but are now moved to a dedicated service module.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

from core.db import get_db_session
from core.models import Symbols
from utils.common import canonical_instrument_type, db_base_symbol

logger = logging.getLogger(__name__)


def upsert_price_series_item(
    *,
    code: str,
    name: Optional[str] = None,
    country: Optional[str] = None,
    exchange: Optional[str] = None,
    currency: Optional[str] = None,
    instrument_type: Optional[str] = None,
    isin: Optional[str] = None
) -> None:
    """
    Insert or update a single Symbols row.
    
    Wrapper around SymbolsRepository.upsert_symbol for backward compatibility.
    
    Args:
        code: Symbol code (required)
        name: Symbol name
        country: Country code
        exchange: Exchange code
        currency: Currency code
        instrument_type: Instrument type
        isin: ISIN code
    """
    try:
        symbol_code = db_base_symbol(code)
        if not symbol_code:
            return
        
        # Process country code
        def _infer_country(country_in, exchange_in, symbol_in):
            """Infer country from inputs."""
            if country_in:
                return country_in
                
            # Infer from exchange
            try:
                ex = str(exchange_in or "").strip().upper()
                if ex in ("TSX", "TSXV", "CSE", "NEO", "TO", "V"):
                    return "Canada"
                if ex in ("NYSE", "NASDAQ", "ARCA", "BATS", "OTC", "NYSEMKT"):
                    return "US"
            except Exception:
                pass
                
            # Infer from symbol suffix
            try:
                s = (symbol_in or "").upper().strip()
                for suf in (".TO", ".V", ".CN", ".NE", ":TO", ":V", ":CN", ":NE"):
                    if s.endswith(suf):
                        return "Canada"
                for suf in (".US", ":US"):
                    if s.endswith(suf):
                        return "US"
            except Exception:
                pass
                
            # Default
            return "US"
            
        country_val = _infer_country(country, exchange, code)
        
        # Get canonical instrument type
        type_val = canonical_instrument_type(instrument_type) if instrument_type else None
        
        # Directly use model to avoid code/symbol confusion with repository
        session = get_db_session()
        if not session:
            return
        
        try:
            # Check if symbol exists
            existing = (
                session.query(Symbols)
                .filter(Symbols.symbol == symbol_code)
                .first()
            )
            
            now = datetime.utcnow()
            
            if existing:
                # Update existing symbol
                existing.name = name or symbol_code
                existing.exchange = exchange or ""
                existing.country = country_val
                if currency:
                    existing.currency = currency
                if type_val:
                    existing.instrument_type = type_val
                if isin:
                    existing.isin = isin
                existing.updated_at = now
            else:
                # Create new symbol
                new_symbol = Symbols(
                    symbol=symbol_code,
                    name=name or symbol_code,
                    exchange=exchange or "",
                    country=country_val,
                    currency=currency,
                    instrument_type=type_val,
                    isin=isin,
                    created_at=now,
                    updated_at=now
                )
                session.add(new_symbol)
            
            session.commit()
        except Exception as e:
            logger.error(f"Database error in upsert_price_series_item: {e}")
            session.rollback()
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to upsert symbol {code}: {e}")


def upsert_price_series_bulk(items: List[Dict[str, Any]]) -> int:
    """
    Upsert many symbol rows. Returns count processed.
    
    Wrapper around batch symbol operations for backward compatibility.
    
    Expected keys per item: Code, Name, Country, Exchange, Currency, Type, Isin
    
    Args:
        items: List of dictionaries with symbol information
        
    Returns:
        Number of symbols processed
    """
    if not items:
        return 0
        
    processed = 0
    session = get_db_session()
    if not session:
        return 0
        
    try:
        now = datetime.utcnow()
        
        for item in items:
            try:
                code = str(item.get("Code") or "").strip()
                if not code:
                    continue
                    
                try:
                    # Process directly here to avoid overhead of multiple function calls
                    symbol_code = db_base_symbol(code)
                    if not symbol_code:
                        continue
                        
                    # Infer country
                    country_val = item.get("Country")
                    if not country_val:
                        # Apply country inferencing logic
                        exchange_val = item.get("Exchange")
                        if exchange_val and exchange_val.upper() in ("TSX", "TSXV", "CSE", "NEO", "TO", "V"):
                            country_val = "Canada"
                        elif exchange_val and exchange_val.upper() in ("NYSE", "NASDAQ", "ARCA", "BATS", "OTC", "NYSEMKT"):
                            country_val = "US"
                        else:
                            for suf in (".TO", ".V", ".CN", ".NE", ":TO", ":V", ":CN", ":NE"):
                                if symbol_code.upper().endswith(suf):
                                    country_val = "Canada"
                                    break
                            else:
                                for suf in (".US", ":US"):
                                    if symbol_code.upper().endswith(suf):
                                        country_val = "US"
                                        break
                                else:
                                    country_val = "US"  # Default
                    
                    # Get canonical instrument type
                    type_val = canonical_instrument_type(item.get("Type")) if item.get("Type") else None
                    
                    # Find existing or create new
                    existing = (
                        session.query(Symbols)
                        .filter(Symbols.symbol == symbol_code)
                        .first()
                    )
                    
                    if existing:
                        # Update fields only if new value is provided
                        if item.get("Name"):
                            existing.name = item.get("Name")
                        existing.country = country_val
                        if item.get("Exchange"):
                            existing.exchange = item.get("Exchange")
                        if item.get("Currency"):
                            existing.currency = item.get("Currency")
                        if type_val:
                            existing.instrument_type = type_val
                        if item.get("Isin"):
                            existing.isin = item.get("Isin")
                        existing.updated_at = now
                    else:
                        # Create new record
                        new_symbol = Symbols(
                            symbol=symbol_code,
                            name=item.get("Name") or symbol_code,
                            country=country_val,
                            exchange=item.get("Exchange") or "",
                            currency=item.get("Currency"),
                            instrument_type=type_val,
                            isin=item.get("Isin"),
                            created_at=now,
                            updated_at=now
                        )
                        session.add(new_symbol)
                except Exception as e:
                    logger.error(f"Error processing item in bulk: {e}")
                    continue
                
                # Apply five_stars flag if present as part of the same transaction
                try:
                    mark_star = False
                    v = item.get("five_stars") if "five_stars" in item else item.get("FiveStars")
                    if isinstance(v, str):
                        mark_star = v.strip().lower() in ("1", "true", "yes")
                    elif isinstance(v, (int, float)):
                        mark_star = bool(v)
                    elif isinstance(v, bool):
                        mark_star = v
                        
                    if mark_star:
                        # Apply to the same object we just created/updated
                        if existing:
                            existing.five_stars = 1
                        elif new_symbol:  # Local variable from above
                            new_symbol.five_stars = 1
                except Exception as e:
                    logger.warning(f"Failed to set five_stars for {code}: {e}")
                
                processed += 1
                # Commit periodically to reduce transaction size
                if (processed % 500) == 0:
                    session.commit()
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                continue
                
        # Final commit
        session.commit()
        logger.info(f"Processed {processed} symbols in bulk operation")
        return processed
    except Exception as e:
        logger.error(f"Bulk symbol upsert failed: {e}")
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
