"""
Symbols Repository for handling symbol and exchange operations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from repositories.base_repository import BaseRepository
from core.models import Symbols, Exchange, InstrumentAlias
import logging

logger = logging.getLogger(__name__)


class SymbolsRepository(BaseRepository[Symbols]):
    """Repository for Symbols operations."""
    
    def __init__(self):
        super().__init__(Symbols)
    
    def get_by_code(self, code: str) -> Optional[Symbols]:
        """Get symbol by code."""
        def query_func(session: Session) -> Optional[Symbols]:
            return (
                session.query(Symbols)
                .filter(Symbols.code == code)
                .first()
            )
        
        return self.execute_query(query_func)
    
    def get_symbols_by_exchange(self, exchange_code: str) -> List[Symbols]:
        """Get all symbols for an exchange."""
        def query_func(session: Session) -> List[Symbols]:
            return (
                session.query(Symbols)
                .filter(Symbols.exchange == exchange_code)
                .all()
            )
        
        return self.execute_query(query_func) or []
    
    def get_symbols_by_type(self, instrument_type: str) -> List[Symbols]:
        """Get symbols by instrument type."""
        def query_func(session: Session) -> List[Symbols]:
            return (
                session.query(Symbols)
                .filter(Symbols.type == instrument_type)
                .all()
            )
        
        return self.execute_query(query_func) or []
    
    def upsert_symbol(
        self,
        code: str,
        name: str,
        exchange: str,
        country: Optional[str] = None,
        currency: Optional[str] = None,
        instrument_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Insert or update symbol information."""
        def query_func(session: Session) -> bool:
            existing = (
                session.query(Symbols)
                .filter(Symbols.code == code, Symbols.exchange == exchange)
                .first()
            )
            
            if existing:
                # Update existing
                existing.name = name
                existing.country = country
                existing.currency = currency
                existing.type = instrument_type
                existing.updated_at = datetime.utcnow()
                
                # Update any additional fields
                for key, value in kwargs.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # Create new symbol
                new_symbol = Symbols(
                    code=code,
                    name=name,
                    exchange=exchange,
                    country=country,
                    currency=currency,
                    type=instrument_type,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    **kwargs
                )
                session.add(new_symbol)
            
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def search_symbols(
        self,
        query: str,
        limit: int = 50,
        exchange_filter: Optional[str] = None
    ) -> List[Symbols]:
        """Search symbols by code or name."""
        def query_func(session: Session) -> List[Symbols]:
            search_query = session.query(Symbols)
            
            # Search in both code and name
            search_pattern = f"%{query.upper()}%"
            search_query = search_query.filter(
                or_(
                    Symbols.code.ilike(search_pattern),
                    Symbols.name.ilike(search_pattern)
                )
            )
            
            # Apply exchange filter if provided
            if exchange_filter:
                search_query = search_query.filter(Symbols.exchange == exchange_filter)
            
            return search_query.limit(limit).all()
        
        return self.execute_query(query_func) or []
    
    def get_exchange_info(self, exchange_code: str) -> Optional[Exchange]:
        """Get exchange information."""
        def query_func(session: Session) -> Optional[Exchange]:
            return (
                session.query(Exchange)
                .filter(Exchange.code == exchange_code)
                .first()
            )
        
        return self.execute_query(query_func)
    
    def get_all_exchanges(self) -> List[Exchange]:
        """Get all exchanges."""
        def query_func(session: Session) -> List[Exchange]:
            return session.query(Exchange).all()
        
        return self.execute_query(query_func) or []
    
    def upsert_exchange(
        self,
        code: str,
        name: str,
        country: Optional[str] = None,
        timezone: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Insert or update exchange information."""
        def query_func(session: Session) -> bool:
            existing = (
                session.query(Exchange)
                .filter(Exchange.code == code)
                .first()
            )
            
            if existing:
                # Update existing
                existing.name = name
                existing.country = country
                existing.timezone = timezone
                existing.updated_at = datetime.utcnow()
                
                for key, value in kwargs.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # Create new exchange
                new_exchange = Exchange(
                    code=code,
                    name=name,
                    country=country,
                    timezone=timezone,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    **kwargs
                )
                session.add(new_exchange)
            
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def create_instrument_alias(
        self,
        symbol: str,
        alias: str,
        alias_type: str = "ticker"
    ) -> bool:
        """Create instrument alias."""
        def query_func(session: Session) -> bool:
            # Check if alias already exists
            existing = (
                session.query(InstrumentAlias)
                .filter(
                    InstrumentAlias.symbol == symbol,
                    InstrumentAlias.alias == alias
                )
                .first()
            )
            
            if not existing:
                new_alias = InstrumentAlias(
                    symbol=symbol,
                    alias=alias,
                    alias_type=alias_type,
                    created_at=datetime.utcnow()
                )
                session.add(new_alias)
                session.commit()
            
            return True
        
        return self.execute_query(query_func) or False
    
    def get_symbol_aliases(self, symbol: str) -> List[InstrumentAlias]:
        """Get all aliases for a symbol."""
        def query_func(session: Session) -> List[InstrumentAlias]:
            return (
                session.query(InstrumentAlias)
                .filter(InstrumentAlias.symbol == symbol)
                .all()
            )
        
        return self.execute_query(query_func) or []
    
    def resolve_symbol_from_alias(self, alias: str) -> Optional[str]:
        """Resolve symbol from alias."""
        def query_func(session: Session) -> Optional[str]:
            result = (
                session.query(InstrumentAlias.symbol)
                .filter(InstrumentAlias.alias == alias)
                .first()
            )
            return result[0] if result else None
        
        return self.execute_query(query_func)
    
    def get_symbol_stats(self) -> Dict[str, Any]:
        """Get symbol statistics."""
        def query_func(session: Session) -> Dict[str, Any]:
            total_symbols = session.query(Symbols).count()
            
            from sqlalchemy import func
            
            # Count by exchange
            exchange_counts = {}
            exchange_results = (
                session.query(Symbols.exchange, func.count(Symbols.id))
                .group_by(Symbols.exchange)
                .all()
            )
            for exchange, count in exchange_results:
                exchange_counts[exchange] = count
            
            # Count by type
            type_counts = {}
            type_results = (
                session.query(Symbols.type, func.count(Symbols.id))
                .group_by(Symbols.type)
                .all()
            )
            for symbol_type, count in type_results:
                type_counts[symbol_type or "Unknown"] = count
            
            # Count by country
            country_counts = {}
            country_results = (
                session.query(Symbols.country, func.count(Symbols.id))
                .group_by(Symbols.country)
                .all()
            )
            for country, count in country_results:
                country_counts[country or "Unknown"] = count
            
            return {
                "total_symbols": total_symbols,
                "by_exchange": exchange_counts,
                "by_type": type_counts,
                "by_country": country_counts
            }
        
        return self.execute_query(query_func) or {}
