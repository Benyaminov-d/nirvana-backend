"""
Query Builders - Reusable database query patterns and optimizations.

This module provides optimized, reusable query builders that eliminate N+1 problems
and standardize common data access patterns across the application.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, date
import logging
from sqlalchemy.orm import Session, Query  # type: ignore
from sqlalchemy.sql import func, and_, or_  # type: ignore

from core.models import Symbols, CvarSnapshot, CompassAnchor
from core.db import get_db_session

logger = logging.getLogger(__name__)


class CvarQueryBuilder:
    """
    Optimized query builder for CVaR and Symbols data.
    
    Eliminates N+1 queries by providing batch operations and optimized JOINs.
    """
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_db_session()
        if not self.session:
            raise RuntimeError("Database session not available")
    
    def close(self) -> None:
        """Close the database session."""
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                logger.warning(f"Failed to close session: {e}")
    
    def get_symbols_with_filters(
        self,
        five_stars: bool = False,
        ready_only: bool = True,
        include_unknown: bool = False,
        country: Optional[str] = None,
        instrument_types: Optional[List[str]] = None,
        exclude_exchanges: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get symbols matching common filter criteria.
        
        This replaces multiple scattered queries with a single optimized query.
        """
        query = self.session.query(Symbols.symbol).distinct()
        
        # Five stars filter
        if five_stars:
            query = query.filter(Symbols.five_stars == 1)
        
        # History sufficiency filter
        if ready_only:
            if include_unknown:
                query = query.filter(
                    or_(
                        Symbols.insufficient_history == 0,
                        Symbols.insufficient_history.is_(None)
                    )
                )
            else:
                query = query.filter(Symbols.insufficient_history == 0)
        
        # Country filter
        if country:
            query = query.filter(Symbols.country == country)
        
        # Instrument types filter
        if instrument_types:
            query = query.filter(Symbols.instrument_type.in_(instrument_types))
        
        # Exchange exclusion filter
        if exclude_exchanges:
            query = query.filter(~Symbols.exchange.in_(exclude_exchanges))
        
        # Apply limit
        if limit and limit > 0:
            query = query.limit(limit)
        
        try:
            results = query.all()
            return [symbol for (symbol,) in results]
        except Exception as e:
            logger.error(f"Symbol query failed: {e}")
            return []
    
    def get_latest_cvar_snapshots_batch(
        self,
        symbols: List[str],
        alpha_labels: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, Any]]:
        """
        Get latest CVaR snapshots for multiple symbols in a single query.
        
        Returns: {symbol: {alpha_label: snapshot_row}}
        
        This eliminates N+1 queries when getting snapshots for multiple symbols.
        """
        if not symbols:
            return {}
        
        if alpha_labels is None:
            alpha_labels = [50, 95, 99]
        
        # Subquery to get latest snapshot per symbol/alpha combination
        latest_subquery = (
            self.session.query(
                CvarSnapshot.symbol.label("symbol"),
                CvarSnapshot.alpha_label.label("alpha_label"),
                func.max(CvarSnapshot.as_of_date).label("max_date")
            )
            .filter(
                CvarSnapshot.symbol.in_(symbols),
                CvarSnapshot.alpha_label.in_(alpha_labels)
            )
            .group_by(CvarSnapshot.symbol, CvarSnapshot.alpha_label)
            .subquery()
        )
        
        # Main query to get full snapshot data
        query = (
            self.session.query(CvarSnapshot)
            .join(
                latest_subquery,
                and_(
                    CvarSnapshot.symbol == latest_subquery.c.symbol,
                    CvarSnapshot.alpha_label == latest_subquery.c.alpha_label,
                    CvarSnapshot.as_of_date == latest_subquery.c.max_date
                )
            )
        )
        
        try:
            snapshots = query.all()
            
            # Organize by symbol and alpha
            result: Dict[str, Dict[int, Any]] = {}
            for snapshot in snapshots:
                if snapshot.symbol not in result:
                    result[snapshot.symbol] = {}
                result[snapshot.symbol][snapshot.alpha_label] = snapshot
            
            return result
            
        except Exception as e:
            logger.error(f"Batch snapshot query failed: {e}")
            return {}
    
    def get_symbols_with_cvar_data(
        self,
        country: Optional[str] = None,
        five_stars: bool = False,
        alpha_label: int = 99,
        limit: int = 20,
        instrument_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get symbols with their latest CVaR data in a single optimized query.
        
        This replaces the scattered queries in ticker_feed and similar functions.
        """
        # Latest snapshot subquery
        latest_subquery = (
            self.session.query(
                CvarSnapshot.symbol.label("symbol"),
                func.max(CvarSnapshot.as_of_date).label("max_date")
            )
            .filter(CvarSnapshot.alpha_label == alpha_label)
            .group_by(CvarSnapshot.symbol)
            .subquery()
        )
        
        # Main query with JOIN
        query = (
            self.session.query(
                CvarSnapshot.symbol,
                CvarSnapshot.as_of_date,
                CvarSnapshot.cvar_nig,
                CvarSnapshot.cvar_ghst,
                CvarSnapshot.cvar_evar,
                Symbols.name,
                Symbols.country,
                Symbols.instrument_type,
                Symbols.five_stars
            )
            .join(
                latest_subquery,
                and_(
                    CvarSnapshot.symbol == latest_subquery.c.symbol,
                    CvarSnapshot.as_of_date == latest_subquery.c.max_date
                )
            )
            .join(Symbols, Symbols.symbol == CvarSnapshot.symbol)
            .filter(
                CvarSnapshot.alpha_label == alpha_label,
                Symbols.insufficient_history == 0
            )
        )
        
        # Apply filters
        if country:
            query = query.filter(Symbols.country == country)
        
        if five_stars:
            query = query.filter(Symbols.five_stars == 1)
        
        if instrument_types:
            query = query.filter(Symbols.instrument_type.in_(instrument_types))
        
        # Order and limit
        query = query.order_by(CvarSnapshot.as_of_date.desc()).limit(limit)
        
        try:
            results = query.all()
            
            return [
                {
                    "symbol": row.symbol,
                    "name": row.name or row.symbol,
                    "country": row.country,
                    "instrument_type": row.instrument_type,
                    "as_of_date": row.as_of_date.isoformat() if row.as_of_date else None,
                    "five_stars": bool(row.five_stars),
                    "cvar_data": {
                        "nig": float(row.cvar_nig) if row.cvar_nig is not None else None,
                        "ghst": float(row.cvar_ghst) if row.cvar_ghst is not None else None,
                        "evar": float(row.cvar_evar) if row.cvar_evar is not None else None,
                        "worst": self._calculate_worst_cvar(row.cvar_nig, row.cvar_ghst, row.cvar_evar)
                    }
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Symbols with CVaR data query failed: {e}")
            return []
    
    def get_symbol_info_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get basic info for multiple symbols in a single query.
        
        Returns: {symbol: {name, country, instrument_type, etc}}
        """
        if not symbols:
            return {}
        
        try:
            query = (
                self.session.query(Symbols)
                .filter(Symbols.symbol.in_(symbols))
            )
            
            results = query.all()
            
            return {
                row.symbol: {
                    "name": row.name or row.symbol,
                    "country": row.country,
                    "instrument_type": row.instrument_type,
                    "exchange": row.exchange,
                    "five_stars": bool(row.five_stars) if row.five_stars is not None else False,
                    "insufficient_history": row.insufficient_history,
                    "valid": row.valid
                }
                for row in results
            }
            
        except Exception as e:
            logger.error(f"Symbol info batch query failed: {e}")
            return {}
    
    def search_symbols(
        self,
        query_text: str,
        limit: int = 10,
        ready_only: bool = True,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search symbols by name or ticker with optimized query.
        """
        if not query_text.strip():
            return []
        
        search_pattern = f"%{query_text.upper()}%"
        
        query = (
            self.session.query(
                Symbols.symbol,
                Symbols.name,
                Symbols.country,
                Symbols.instrument_type,
                Symbols.exchange
            )
            .filter(
                or_(
                    Symbols.symbol.ilike(search_pattern),
                    Symbols.name.ilike(search_pattern)
                )
            )
        )
        
        if ready_only:
            query = query.filter(Symbols.insufficient_history == 0)
        
        if country:
            query = query.filter(Symbols.country == country)
        
        query = query.order_by(Symbols.symbol).limit(limit)
        
        try:
            results = query.all()
            
            return [
                {
                    "symbol": row.symbol,
                    "name": row.name or row.symbol,
                    "country": row.country,
                    "instrument_type": row.instrument_type,
                    "exchange": row.exchange
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Symbol search query failed: {e}")
            return []
    
    def get_duplicate_symbols(self) -> List[Tuple[str, int, List[str]]]:
        """
        Find symbols that exist in multiple countries.
        
        Returns: [(symbol, country_count, [countries])]
        """
        try:
            # Find symbols with multiple countries
            duplicate_query = (
                self.session.query(
                    Symbols.symbol,
                    func.count(func.distinct(Symbols.country)).label('country_count')
                )
                .group_by(Symbols.symbol)
                .having(func.count(func.distinct(Symbols.country)) > 1)
                .all()
            )
            
            result = []
            for symbol, count in duplicate_query:
                # Get all countries for this symbol in a single query
                countries = (
                    self.session.query(func.distinct(Symbols.country))
                    .filter(Symbols.symbol == symbol)
                    .all()
                )
                country_list = [c[0] for c in countries if c[0]]
                result.append((symbol, count, country_list))
            
            return result
            
        except Exception as e:
            logger.error(f"Duplicate symbols query failed: {e}")
            return []
    
    def get_stale_snapshots(
        self,
        symbols: List[str],
        max_age_days: int = 7
    ) -> Set[str]:
        """
        Find symbols with stale or missing snapshot data.
        
        Returns set of symbols that need recalculation.
        """
        if not symbols:
            return set()
        
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow().date() - timedelta(days=max_age_days)
            
            # Get latest snapshot dates for all symbols
            latest_dates = (
                self.session.query(
                    CvarSnapshot.symbol,
                    func.max(CvarSnapshot.as_of_date).label('latest_date')
                )
                .filter(CvarSnapshot.symbol.in_(symbols))
                .group_by(CvarSnapshot.symbol)
                .all()
            )
            
            # Find symbols with missing or stale data
            fresh_symbols = set()
            for symbol, latest_date in latest_dates:
                if latest_date and latest_date >= cutoff_date:
                    fresh_symbols.add(symbol)
            
            # Stale symbols = all symbols - fresh symbols
            return set(symbols) - fresh_symbols
            
        except Exception as e:
            logger.error(f"Stale snapshots query failed: {e}")
            return set(symbols)  # Assume all stale on error
    
    @staticmethod
    def _calculate_worst_cvar(nig: Optional[float], ghst: Optional[float], evar: Optional[float]) -> Optional[float]:
        """Calculate worst-case CVaR from the three methods."""
        values = []
        for val in [nig, ghst, evar]:
            if val is not None:
                try:
                    float_val = float(val)
                    if float_val == float_val:  # Check for NaN
                        values.append(float_val)
                except (ValueError, TypeError):
                    continue
        
        return max(values) if values else None


class CompassQueryBuilder:
    """
    Query builder for Compass anchor and scoring operations.
    """
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_db_session()
        if not self.session:
            raise RuntimeError("Database session not available")
    
    def close(self) -> None:
        """Close the database session."""
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                logger.warning(f"Failed to close session: {e}")
    
    def get_latest_anchor(self, category: str) -> Optional[Dict[str, Any]]:
        """Get latest anchor for a category."""
        try:
            anchor = (
                self.session.query(CompassAnchor)
                .filter(CompassAnchor.category == category)
                .order_by(CompassAnchor.created_at.desc())
                .first()
            )
            
            if anchor:
                return {
                    "category": anchor.category,
                    "version": anchor.version,
                    "mu_low": float(anchor.mu_low),
                    "mu_high": float(anchor.mu_high),
                    "created_at": anchor.created_at.isoformat() if anchor.created_at else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Latest anchor query failed for {category}: {e}")
            return None
    
    def get_candidates_for_scoring(
        self,
        alpha: int = 99,
        country: Optional[str] = None,
        instrument_types: Optional[List[str]] = None,
        limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Get candidate symbols with their CVaR data for Compass scoring.
        
        This replaces the complex query in experiments.py with an optimized version.
        """
        try:
            # Latest snapshot subquery
            latest_subquery = (
                self.session.query(
                    CvarSnapshot.symbol.label("symbol"),
                    func.max(CvarSnapshot.as_of_date).label("max_date")
                )
                .filter(CvarSnapshot.alpha_label == alpha)
                .group_by(CvarSnapshot.symbol)
                .subquery()
            )
            
            # Main query
            query = (
                self.session.query(
                    CvarSnapshot,
                    Symbols.name,
                    Symbols.country,
                    Symbols.instrument_type
                )
                .join(
                    latest_subquery,
                    and_(
                        CvarSnapshot.symbol == latest_subquery.c.symbol,
                        CvarSnapshot.as_of_date == latest_subquery.c.max_date
                    )
                )
                .outerjoin(Symbols, Symbols.symbol == CvarSnapshot.symbol)
                .filter(
                    CvarSnapshot.alpha_label == alpha,
                    Symbols.insufficient_history == 0
                )
            )
            
            # Apply filters
            if country:
                query = query.filter(Symbols.country == country)
            
            if instrument_types:
                query = query.filter(Symbols.instrument_type.in_(instrument_types))
            
            query = query.limit(limit)
            
            results = query.all()
            
            return [
                {
                    "symbol": row.CvarSnapshot.symbol,
                    "name": row.name or row.CvarSnapshot.symbol,
                    "country": row.country,
                    "instrument_type": row.instrument_type,
                    "as_of_date": row.CvarSnapshot.as_of_date.isoformat() if row.CvarSnapshot.as_of_date else None,
                    "cvar_nig": float(row.CvarSnapshot.cvar_nig) if row.CvarSnapshot.cvar_nig is not None else None,
                    "cvar_ghst": float(row.CvarSnapshot.cvar_ghst) if row.CvarSnapshot.cvar_ghst is not None else None,
                    "cvar_evar": float(row.CvarSnapshot.cvar_evar) if row.CvarSnapshot.cvar_evar is not None else None,
                    "years": float(row.CvarSnapshot.years) if row.CvarSnapshot.years is not None else None
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Candidates query failed: {e}")
            return []


# Context manager for automatic session cleanup
class QueryBuilderContext:
    """
    Context manager that provides query builders with automatic session cleanup.
    
    Usage:
        with QueryBuilderContext() as (cvar_qb, compass_qb):
            symbols = cvar_qb.get_symbols_with_filters(country="US")
            # Session automatically closed
    """
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self.cvar_qb: Optional[CvarQueryBuilder] = None
        self.compass_qb: Optional[CompassQueryBuilder] = None
    
    def __enter__(self) -> Tuple[CvarQueryBuilder, CompassQueryBuilder]:
        session = self.session or get_db_session()
        if not session:
            raise RuntimeError("Database session not available")
        
        self.cvar_qb = CvarQueryBuilder(session)
        self.compass_qb = CompassQueryBuilder(session)
        
        return self.cvar_qb, self.compass_qb
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cvar_qb:
            self.cvar_qb.close()
        if self.compass_qb:
            self.compass_qb.close()
