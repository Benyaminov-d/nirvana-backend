"""
Refactored Ticker Routes - Clean architecture demonstration.

This module shows how ticker-related functionality can be implemented 
using repository pattern and domain services.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Dict, Any, List
import logging

from repositories import PriceSeriesRepository, CvarRepository
from utils.auth import require_pub_or_basic as _require_pub_or_basic

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/ticker/info-refactored")
def ticker_info_refactored(
    symbol: str = Query(..., description="Symbol to get information for"),
    include_cvar: bool = Query(False, description="Include latest CVaR data"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Get comprehensive ticker information using new repository architecture.
    
    Demonstrates:
    - Clean data access through repositories
    - No direct SQL queries in route handler
    - Separation of concerns
    - Easy testing and maintenance
    """
    
    # Initialize repositories
    price_repo = PriceSeriesRepository()
    cvar_repo = CvarRepository()
    
    try:
        # Get basic ticker information
        price_series = price_repo.get_by_symbol(symbol)
        
        if not price_series:
            raise HTTPException(404, f"Symbol {symbol} not found")
        
        # Build basic response
        ticker_info = {
            "symbol": symbol,
            "name": price_series.name,
            "exchange": price_series.exchange,
            "country": price_series.country,
            "currency": price_series.currency,
            "instrument_type": price_series.instrument_type,
            "five_stars": bool(price_series.five_stars),
            "valid": bool(price_series.valid),
            "insufficient_history": price_series.insufficient_history,
            "data_source": "refactored_repository_pattern"
        }
        
        # Add CVaR data if requested
        if include_cvar:
            try:
                cvar_snapshots = cvar_repo.get_latest_by_symbol(symbol)
                
                if cvar_snapshots:
                    cvar_data = {}
                    for snapshot in cvar_snapshots:
                        alpha_key = f"alpha_{snapshot.alpha_label}"
                        cvar_data[alpha_key] = {
                            "alpha_label": snapshot.alpha_label,
                            "alpha_conf": snapshot.alpha_conf,
                            "as_of_date": snapshot.as_of_date.isoformat() if snapshot.as_of_date else None,
                            "years": snapshot.years,
                            "cvar_nig": snapshot.cvar_nig,
                            "cvar_ghst": snapshot.cvar_ghst,
                            "cvar_evar": snapshot.cvar_evar,
                            "return_as_of": snapshot.return_as_of,
                            "return_annual": snapshot.return_annual,
                            "source": snapshot.source
                        }
                    
                    ticker_info["cvar_data"] = cvar_data
                    ticker_info["cvar_snapshots_count"] = len(cvar_snapshots)
                else:
                    ticker_info["cvar_data"] = None
                    ticker_info["cvar_message"] = "No CVaR data available"
                    
            except Exception as e:
                logger.warning(f"Failed to fetch CVaR data for {symbol}: {e}")
                ticker_info["cvar_error"] = str(e)
        
        return ticker_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ticker info for {symbol}: {e}")
        raise HTTPException(500, f"Failed to retrieve ticker information: {str(e)}")


@router.get("/ticker/batch-info-refactored")
def ticker_batch_info_refactored(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    include_country: bool = Query(True, description="Include country information"),
    limit: int = Query(50, description="Maximum number of symbols to process"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Get batch ticker information using repository pattern.
    
    Demonstrates efficient batch processing with repositories.
    """
    
    # Parse symbols
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    if not symbol_list:
        raise HTTPException(400, "No valid symbols provided")
    
    # Apply limit
    if len(symbol_list) > limit:
        symbol_list = symbol_list[:limit]
    
    # Initialize repository
    price_repo = PriceSeriesRepository()
    
    try:
        # Get symbols with country info (efficient batch query)
        if include_country:
            symbol_data = price_repo.get_symbols_with_country_info(symbol_list)
        else:
            # Fallback to individual queries if no batch method needed
            symbol_data = []
            for sym in symbol_list:
                series = price_repo.get_by_symbol(sym)
                if series:
                    symbol_data.append((sym, series.country, series.instrument_type))
        
        # Process results
        results = []
        found_symbols = set()
        
        for symbol, country, instrument_type in symbol_data:
            results.append({
                "symbol": symbol,
                "country": country,
                "instrument_type": instrument_type,
                "found": True
            })
            found_symbols.add(symbol)
        
        # Add missing symbols
        for symbol in symbol_list:
            if symbol not in found_symbols:
                results.append({
                    "symbol": symbol,
                    "country": None,
                    "instrument_type": None,
                    "found": False
                })
        
        # Get summary statistics
        country_analysis = price_repo.get_country_mix_analysis([s[0] for s in symbol_data])
        
        return {
            "requested_symbols": len(symbol_list),
            "found_symbols": len(found_symbols),
            "missing_symbols": len(symbol_list) - len(found_symbols),
            "results": results,
            "country_analysis": country_analysis,
            "data_source": "refactored_repository_pattern"
        }
        
    except Exception as e:
        logger.error(f"Batch ticker info failed: {e}")
        raise HTTPException(500, f"Batch processing failed: {str(e)}")


@router.get("/ticker/five-stars-refactored")
def five_stars_refactored(
    country: Optional[str] = Query(None, description="Country filter"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated instrument types"),
    limit: int = Query(100, description="Maximum results"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Get five-star symbols using repository pattern.
    
    Demonstrates clean filtering and data access.
    """
    
    # Initialize repositories
    price_repo = PriceSeriesRepository()
    cvar_repo = CvarRepository()
    
    try:
        # Parse instrument types
        types_list = None
        if instrument_types:
            types_list = [t.strip() for t in instrument_types.split(",") if t.strip()]
        
        # Get five-star symbols using repository
        five_star_data = price_repo.get_five_stars_symbols(
            country=country,
            instrument_types=types_list
        )
        
        # Apply limit
        if limit > 0:
            five_star_data = five_star_data[:limit]
        
        # Get CVaR data for these symbols
        symbols_with_cvar = []
        symbols_without_cvar = []
        
        for symbol, instrument_type in five_star_data:
            # Check if symbol has recent CVaR data
            cvar_snapshots = cvar_repo.get_latest_by_symbol(symbol)
            
            symbol_info = {
                "symbol": symbol,
                "instrument_type": instrument_type,
                "has_cvar_data": len(cvar_snapshots) > 0
            }
            
            if cvar_snapshots:
                # Add latest snapshot info
                latest = max(cvar_snapshots, key=lambda x: x.as_of_date or "")
                symbol_info.update({
                    "latest_cvar_date": latest.as_of_date.isoformat() if latest.as_of_date else None,
                    "cvar_source": latest.source,
                    "snapshots_count": len(cvar_snapshots)
                })
                symbols_with_cvar.append(symbol_info)
            else:
                symbols_without_cvar.append(symbol_info)
        
        return {
            "filters_applied": {
                "country": country,
                "instrument_types": types_list,
                "limit": limit
            },
            "total_five_stars": len(five_star_data),
            "with_cvar_data": len(symbols_with_cvar),
            "without_cvar_data": len(symbols_without_cvar),
            "symbols_with_cvar": symbols_with_cvar,
            "symbols_without_cvar": symbols_without_cvar,
            "data_source": "refactored_repository_pattern"
        }
        
    except Exception as e:
        logger.error(f"Five stars query failed: {e}")
        raise HTTPException(500, f"Five stars query failed: {str(e)}")


@router.get("/ticker/repository-demo")
def repository_demo(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate repository capabilities and architecture benefits.
    """
    
    # Initialize repositories
    price_repo = PriceSeriesRepository()
    cvar_repo = CvarRepository()
    
    try:
        # Get various statistics using repository methods
        fresh_symbols = cvar_repo.get_symbols_with_fresh_data(max_age_days=7)
        stale_symbols = cvar_repo.get_symbols_needing_update(max_age_days=7)
        
        five_star_us = price_repo.get_symbols_by_filters(
            five_stars=True,
            country="US",
            limit=10
        )
        
        return {
            "repository_capabilities": {
                "description": "Repository pattern provides clean data access abstraction",
                "benefits": [
                    "No SQL in business logic",
                    "Easy testing with mocks", 
                    "Consistent error handling",
                    "Centralized query logic",
                    "Type safety"
                ]
            },
            "data_freshness": {
                "symbols_with_fresh_cvar": len(fresh_symbols),
                "symbols_needing_update": len(stale_symbols),
                "fresh_sample": fresh_symbols[:5]
            },
            "five_star_sample": {
                "us_five_stars": five_star_us[:5],
                "count": len(five_star_us)
            },
            "architecture_notes": {
                "repositories_created": 6,
                "domain_services_created": 1,
                "routes_migrated": "In progress",
                "direct_sql_eliminated": "Repositories handle all DB access"
            }
        }
        
    except Exception as e:
        logger.error(f"Repository demo failed: {e}")
        raise HTTPException(500, f"Demo failed: {str(e)}")
