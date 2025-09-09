"""
CVaR Repository for handling all CVaR-related database operations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from repositories.base_repository import BaseRepository
from core.models import CvarSnapshot, Symbols
import logging

logger = logging.getLogger(__name__)


class CvarRepository(BaseRepository[CvarSnapshot]):
    """Repository for CVaR snapshot operations."""
    
    def __init__(self):
        super().__init__(CvarSnapshot)
    
    def get_latest_by_symbol(self, symbol: str) -> List[CvarSnapshot]:
        """Get latest CVaR snapshots for a symbol (all alphas)."""
        def query_func(session: Session) -> List[CvarSnapshot]:
            return (
                session.query(CvarSnapshot)
                .filter(CvarSnapshot.symbol == symbol)
                .order_by(CvarSnapshot.as_of_date.desc())
                .all()
            )
        
        return self.execute_query(query_func) or []
    
    def get_latest_by_symbol_and_alpha(self, symbol: str, alpha_label: int) -> Optional[CvarSnapshot]:
        """Get latest CVaR snapshot for specific symbol and alpha."""
        def query_func(session: Session) -> Optional[CvarSnapshot]:
            return (
                session.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.symbol == symbol,
                    CvarSnapshot.alpha_label == alpha_label
                )
                .order_by(CvarSnapshot.as_of_date.desc())
                .first()
            )
        
        return self.execute_query(query_func)
    
    def get_symbols_with_fresh_data(self, max_age_days: int = 7) -> List[str]:
        """Get symbols that have fresh CVaR data (within max_age_days)."""
        def query_func(session: Session) -> List[str]:
            from datetime import date, timedelta
            cutoff_date = date.today() - timedelta(days=max_age_days)
            
            # Get latest date per symbol
            latest_per_symbol = (
                session.query(
                    CvarSnapshot.symbol.label("symbol"),
                    func.max(CvarSnapshot.as_of_date).label("latest_date")
                )
                .group_by(CvarSnapshot.symbol)
                .subquery()
            )
            
            # Filter symbols with recent data
            fresh_symbols = (
                session.query(latest_per_symbol.c.symbol)
                .filter(latest_per_symbol.c.latest_date >= cutoff_date)
                .all()
            )
            
            return [s[0] for s in fresh_symbols]
        
        return self.execute_query(query_func) or []
    
    def get_symbols_needing_update(self, max_age_days: int = 7) -> List[str]:
        """Get symbols that need CVaR data updates."""
        def query_func(session: Session) -> List[str]:
            from datetime import date, timedelta
            cutoff_date = date.today() - timedelta(days=max_age_days)
            
            # Get all symbols from Symbols
            all_symbols_query = session.query(Symbols.symbol).distinct()
            all_symbols = {s[0] for s in all_symbols_query.all()}
            
            # Get symbols with recent CVaR data  
            fresh_symbols = set(self.get_symbols_with_fresh_data(max_age_days))
            
            # Return symbols that need updates
            return list(all_symbols - fresh_symbols)
        
        return self.execute_query(query_func) or []
    
    def upsert_snapshot(
        self,
        symbol: str,
        as_of_date: date,
        alpha_label: int,
        alpha_conf: Optional[float] = None,
        years: Optional[float] = None,
        cvar_nig: Optional[float] = None,
        cvar_ghst: Optional[float] = None,
        cvar_evar: Optional[float] = None,
        source: str = "unknown",
        return_as_of: Optional[float] = None,
        return_annual: Optional[float] = None
    ) -> bool:
        """Insert or update CVaR snapshot."""
        def query_func(session: Session) -> bool:
            # Check if snapshot already exists
            existing = (
                session.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.symbol == symbol,
                    CvarSnapshot.as_of_date == as_of_date,
                    CvarSnapshot.alpha_label == alpha_label
                )
                .first()
            )
            
            if existing:
                # Update existing
                existing.alpha_conf = alpha_conf
                existing.years = years
                existing.cvar_nig = cvar_nig
                existing.cvar_ghst = cvar_ghst
                existing.cvar_evar = cvar_evar
                existing.source = source
                existing.return_as_of = return_as_of
                existing.return_annual = return_annual
                existing.updated_at = datetime.utcnow()
            else:
                # Create new snapshot
                # First, get instrument_id from Symbols
                price_series = (
                    session.query(Symbols)
                    .filter(Symbols.symbol == symbol)
                    .first()
                )
                
                if not price_series:
                    logger.warning(f"Symbols not found for symbol: {symbol}")
                    return False
                
                new_snapshot = CvarSnapshot(
                    symbol=symbol,
                    instrument_id=price_series.id,
                    as_of_date=as_of_date,
                    alpha_label=alpha_label,
                    alpha_conf=alpha_conf,
                    years=years,
                    cvar_nig=cvar_nig,
                    cvar_ghst=cvar_ghst,
                    cvar_evar=cvar_evar,
                    source=source,
                    return_as_of=return_as_of,
                    return_annual=return_annual,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(new_snapshot)
            
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def get_five_stars_batch(
        self, 
        alpha_label: int = 99, 
        country: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get five-star symbols with latest CVaR data."""
        def query_func(session: Session) -> List[Dict[str, Any]]:
            # Get latest per symbol for specific alpha
            latest_per_symbol = (
                session.query(
                    CvarSnapshot.symbol.label("symbol"),
                    func.max(CvarSnapshot.as_of_date).label("mx")
                )
                .filter(CvarSnapshot.alpha_label == alpha_label)
                .group_by(CvarSnapshot.symbol)
                .subquery()
            )
            
            # Join with latest snapshots and Symbols
            query = (
                session.query(CvarSnapshot, Symbols)
                .join(
                    latest_per_symbol,
                    and_(
                        CvarSnapshot.symbol == latest_per_symbol.c.symbol,
                        CvarSnapshot.as_of_date == latest_per_symbol.c.mx
                    )
                )
                .outerjoin(Symbols, Symbols.symbol == CvarSnapshot.symbol)
                .filter(CvarSnapshot.alpha_label == alpha_label)
                .filter(Symbols.five_stars == 1)
            )
            
            # Apply country filter
            if country:
                query = query.filter(Symbols.country == country)
            else:
                # Default to US
                query = query.filter(
                    Symbols.country.in_(["US", "USA", "United States"])
                )
            
            # Apply limit
            if limit and limit > 0:
                query = query.limit(limit)
            
            results = []
            for snapshot, price_series in query.all():
                # Calculate worst-case CVaR
                vals = [
                    snapshot.cvar_nig,
                    snapshot.cvar_ghst, 
                    snapshot.cvar_evar
                ]
                valid_vals = [v for v in vals if v is not None and v == v and v >= 0]
                worst = max(valid_vals) if valid_vals else None
                
                results.append({
                    "symbol": snapshot.symbol,
                    "name": getattr(price_series, "name", None) if price_series else None,
                    "as_of": snapshot.as_of_date.isoformat() if snapshot.as_of_date else None,
                    "alpha": alpha_label,
                    "value": worst
                })
            
            return results
        
        return self.execute_query(query_func) or []
    
    def get_evar_analysis(
        self, 
        country: Optional[str] = None,
        types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze when EVaR is the largest among NIG, GHST, and EVaR."""
        def query_func(session: Session) -> Dict[str, Any]:
            # Base query joining snapshots with price series
            query = (
                session.query(CvarSnapshot)
                .join(Symbols, CvarSnapshot.instrument_id == Symbols.id)
            )
            
            # Apply country filter
            if country:
                country_list = [c.strip() for c in country.split(",") if c.strip()]
                if country_list:
                    query = query.filter(Symbols.country.in_(country_list))
            
            # Apply instrument types filter
            if types:
                query = query.filter(Symbols.instrument_type.in_(types))
            
            # Filter out NULL values
            query = query.filter(
                CvarSnapshot.cvar_nig.isnot(None),
                CvarSnapshot.cvar_ghst.isnot(None),
                CvarSnapshot.cvar_evar.isnot(None)
            )
            
            all_snapshots = query.all()
            
            if not all_snapshots:
                return {
                    "total_records": 0,
                    "evar_largest_count": 0,
                    "evar_largest_percentage": 0.0,
                    "message": "No CVaR snapshot records found with the specified filters"
                }
            
            # Count cases where EVaR is largest
            evar_largest_count = 0
            for snapshot in all_snapshots:
                nig = float(snapshot.cvar_nig)
                ghst = float(snapshot.cvar_ghst)  
                evar = float(snapshot.cvar_evar)
                
                if evar > nig and evar > ghst:
                    evar_largest_count += 1
            
            total_records = len(all_snapshots)
            percentage = (evar_largest_count / total_records) * 100 if total_records > 0 else 0.0
            
            return {
                "total_records": total_records,
                "evar_largest_count": evar_largest_count,
                "evar_largest_percentage": round(percentage, 2),
                "filters_applied": {
                    "country": country,
                    "types": types
                },
                "message": f"EVaR is largest in {evar_largest_count} out of {total_records} records ({percentage:.2f}%)"
            }
        
        return self.execute_query(query_func) or {}
