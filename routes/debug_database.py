"""
Debug database endpoint to check data availability and load symbols.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import logging

from core.db import get_db_session
from core.models import PriceSeries, CvarSnapshot
from utils.auth import require_pub_or_basic as _require_auth

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/debug/database/status")
def database_status(_auth: None = Depends(_require_auth)) -> Dict[str, Any]:
    """Check database status and data availability."""
    
    session = get_db_session()
    if not session:
        return {"error": "Database not available"}
    
    try:
        # Count symbols
        total_symbols = session.query(PriceSeries).count()
        
        # Count by country
        country_counts = {}
        country_results = (
            session.query(
                PriceSeries.country,
                session.query(PriceSeries.id).filter(
                    PriceSeries.country == PriceSeries.country
                ).count()
            )
            .group_by(PriceSeries.country)
            .all()
        )
        for country, count in country_results:
            country_counts[country or "NULL"] = count
        
        # Count five_stars
        five_stars_count = (
            session.query(PriceSeries)
            .filter(PriceSeries.five_stars == 1)
            .count()
        )
        
        # Count CVaR snapshots
        cvar_count = session.query(CvarSnapshot).count()
        
        # Sample symbols
        sample_symbols = session.query(PriceSeries).limit(10).all()
        sample_data = [
            {
                "symbol": s.symbol,
                "name": s.name,
                "country": s.country,
                "five_stars": s.five_stars,
                "instrument_type": s.instrument_type
            }
            for s in sample_symbols
        ]
        
        return {
            "total_symbols": total_symbols,
            "country_counts": country_counts,
            "five_stars_count": five_stars_count,
            "cvar_snapshots_count": cvar_count,
            "sample_symbols": sample_data,
            "database_available": True
        }
        
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return {"error": str(e)}
    finally:
        try:
            session.close()
        except Exception:
            pass


@router.post("/debug/database/load-symbols")
def load_symbols_from_csv(_auth: None = Depends(_require_auth)) -> Dict[str, Any]:
    """Load symbols from CSV files into database."""
    
    try:
        from startup.symbols_core_loader import SymbolsCoreLoader
        
        loader = SymbolsCoreLoader()
        results = loader.load_all_symbols()
        
        return {
            "success": True,
            "message": "Symbols loaded successfully",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to load symbols: {e}")
        return {
            "success": False,
            "error": str(e)
        }
