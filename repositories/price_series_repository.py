"""
Price Series Repository for handling price and symbol data operations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_

from repositories.base_repository import BaseRepository
from core.models import PriceSeries, ValidationFlags
from services.infrastructure.redis_cache_service import (
    get_cache_service, 
    CacheKeyType, 
    StandardCachePolicies
)
import logging

logger = logging.getLogger(__name__)


class PriceSeriesRepository(BaseRepository[PriceSeries]):
    """Repository for PriceSeries operations."""
    
    def __init__(self):
        super().__init__(PriceSeries)
        self.cache_service = get_cache_service()
    
    def get_by_symbol(self, symbol: str) -> Optional[PriceSeries]:
        """Get PriceSeries by symbol."""
        def query_func(session: Session) -> Optional[PriceSeries]:
            return (
                session.query(PriceSeries)
                .filter(PriceSeries.symbol == symbol)
                .first()
            )
        
        return self.execute_query(query_func)
    
    def get_symbols_by_filters(
        self,
        five_stars: bool = False,
        ready_only: bool = True,
        include_unknown: bool = False,
        country: Optional[str] = None,
        instrument_types: Optional[List[str]] = None,
        exclude_exchanges: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """Get symbols filtered by various criteria."""
        def query_func(session: Session) -> List[str]:
            query = session.query(PriceSeries.symbol).filter(PriceSeries.valid == 1)
            
            # Five stars filter
            if five_stars:
                query = query.filter(PriceSeries.five_stars == 1)
            
            # Country filter
            if country:
                query = query.filter(PriceSeries.country == country)
            
            # Instrument types filter
            if instrument_types:
                query = query.filter(PriceSeries.instrument_type.in_(instrument_types))
            
            # Exclude exchanges filter
            if exclude_exchanges:
                query = query.filter(~PriceSeries.exchange.in_(exclude_exchanges))
            
            # Readiness filter
            if ready_only:
                if include_unknown:
                    query = query.filter(
                        or_(
                            PriceSeries.insufficient_history == 0,
                            PriceSeries.insufficient_history.is_(None)
                        )
                    )
                else:
                    query = query.filter(PriceSeries.insufficient_history == 0)
            
            # Apply limit
            if limit and limit > 0:
                query = query.limit(limit)
            
            rows = query.all()
            return [s[0] for s in rows]
        
        return self.execute_query(query_func) or []
    
    def get_symbols_with_country_info(self, symbols: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """Get symbols with their country and instrument type info."""
        def query_func(session: Session) -> List[Tuple[str, Optional[str], Optional[str]]]:
            # Process in chunks to avoid parameter limits
            chunk_size = 500
            results = []
            
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                chunk_results = (
                    session.query(
                        PriceSeries.symbol,
                        PriceSeries.country,
                        PriceSeries.instrument_type
                    )
                    .filter(PriceSeries.symbol.in_(chunk))
                    .all()
                )
                results.extend(chunk_results)
            
            return results
        
        return self.execute_query(query_func) or []
    
    def get_symbol_exchange_country(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Get exchange and country for a symbol."""
        def query_func(session: Session) -> Tuple[Optional[str], Optional[str]]:
            result = (
                session.query(PriceSeries.exchange, PriceSeries.country)
                .filter(PriceSeries.symbol == symbol)
                .first()
            )
            return result if result else (None, None)
        
        return self.execute_query(query_func) or (None, None)
    
    def get_ambiguous_symbols(self, symbol: str) -> List[Tuple[str, Optional[str]]]:
        """Get all records for potentially ambiguous symbols."""
        def query_func(session: Session) -> List[Tuple[str, Optional[str]]]:
            return (
                session.query(PriceSeries.exchange, PriceSeries.country)
                .filter(PriceSeries.symbol == symbol)
                .all()
            )
        
        return self.execute_query(query_func) or []
    
    def update_insufficient_history(self, symbol: str, insufficient_history: int) -> bool:
        """Update insufficient_history flag for a symbol."""
        def query_func(session: Session) -> bool:
            record = (
                session.query(PriceSeries)
                .filter(PriceSeries.symbol == symbol)
                .first()
            )
            
            if record:
                record.insufficient_history = insufficient_history
                session.commit()
                return True
            return False
        
        return self.execute_query(query_func) or False
    
    def get_five_stars_symbols(
        self,
        country: Optional[str] = None,
        instrument_types: Optional[List[str]] = None
    ) -> List[Tuple[str, Optional[str]]]:
        """Get five-star symbols with their instrument types with Redis caching."""
        # Create cache key from parameters
        cache_key_parts = ["five_stars"]
        if country:
            cache_key_parts.append(f"country_{country}")
        if instrument_types:
            # Sort for consistent cache key
            sorted_types = sorted(instrument_types)
            cache_key_parts.append(f"types_{'_'.join(sorted_types)}")
        
        # Try cache first
        cached_result = self.cache_service.get(
            CacheKeyType.FIVE_STARS_FILTER, 
            *cache_key_parts
        )
        if cached_result is not None:
            logger.debug(f"Cache hit for five stars: {cache_key_parts}")
            return cached_result
        
        # Execute database query
        def query_func(session: Session) -> List[Tuple[str, Optional[str]]]:
            query = (
                session.query(PriceSeries.symbol, PriceSeries.instrument_type)
                .filter(PriceSeries.five_stars == 1)
            )
            
            if country:
                query = query.filter(PriceSeries.country == country)
            
            if instrument_types:
                query = query.filter(PriceSeries.instrument_type.in_(instrument_types))
            
            return query.all()
        
        result = self.execute_query(query_func) or []
        
        # Cache the result
        if result:
            success = self.cache_service.set(
                CacheKeyType.FIVE_STARS_FILTER,
                result,
                StandardCachePolicies.FIVE_STARS_FILTER,
                *cache_key_parts
            )
            if success:
                logger.debug(f"Cached {len(result)} five stars symbols")
        
        return result
    
    def get_country_mix_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze country and instrument type distribution for given symbols."""
        def query_func(session: Session) -> Dict[str, Any]:
            if not symbols:
                return {"countries": {}, "instrument_types": {}, "total": 0}
            
            # Get info for provided symbols
            results = (
                session.query(
                    PriceSeries.symbol,
                    PriceSeries.country,
                    PriceSeries.instrument_type
                )
                .filter(PriceSeries.symbol.in_(symbols))
                .all()
            )
            
            country_counts = {}
            type_counts = {}
            
            for symbol, country, instrument_type in results:
                country = (country or "Unknown").strip()
                instrument_type = (instrument_type or "Unknown").strip()
                
                country_counts[country] = country_counts.get(country, 0) + 1
                type_counts[instrument_type] = type_counts.get(instrument_type, 0) + 1
            
            return {
                "countries": country_counts,
                "instrument_types": type_counts,
                "total": len(results)
            }
        
        return self.execute_query(query_func) or {"countries": {}, "instrument_types": {}, "total": 0}
    
    def check_symbol_exists(self, symbol: str) -> bool:
        """Check if symbol exists in PriceSeries."""
        def query_func(session: Session) -> bool:
            return (
                session.query(PriceSeries)
                .filter(PriceSeries.symbol == symbol)
                .count() > 0
            )
        
        return self.execute_query(query_func) or False
    
    def get_symbol_name(self, symbol: str) -> Optional[str]:
        """Get friendly name for a symbol."""
        def query_func(session: Session) -> Optional[str]:
            result = (
                session.query(PriceSeries.name)
                .filter(PriceSeries.symbol == symbol)
                .first()
            )
            return result[0] if result and result[0] else None
        
        return self.execute_query(query_func)
