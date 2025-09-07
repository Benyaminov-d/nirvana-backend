"""
Refactored CVaR Routes - Demonstration of new architecture.

This file shows how the same functionality can be implemented using
the new repository pattern and domain services, eliminating direct
database access from route handlers.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
import logging

from services.domain.cvar_unified_service import CvarUnifiedService
from utils.auth import require_pub_or_basic as _require_pub_or_basic
from utils.common import parse_csv_list

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cvar/recalc-refactored")
@router.get("/cvar/recalc-refactored") 
def cvar_recalc_refactored(
    limit: int = Query(0, description="0=all, >0 to limit symbols"),
    products: str = Query("", description="Comma-separated symbols to process"),
    five_stars: bool = Query(False, description="Process only five-star symbols"),
    ready_only: bool = Query(True, description="Only symbols with sufficient history"),
    include_unknown: bool = Query(False, description="Include unknown status symbols"),
    country: Optional[str] = Query(None, description="Country filter"),
    types: Optional[str] = Query(None, description="Comma-separated instrument types"),
    exclude_exchange: Optional[str] = Query(None, description="Exchanges to exclude"),
    local: bool = Query(True, description="Process locally (demo always local)"),
    verbose: int = Query(0, description="Return detailed results"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """
    Refactored version of cvar_recalc_all using new architecture.
    
    This endpoint demonstrates:
    - No direct database access in route handler
    - Clean separation of concerns using domain service
    - Repository pattern for data access
    - Improved error handling and logging
    """
    
    # Initialize domain service
    cvar_service = CvarUnifiedService()
    
    try:
        # Get symbols to process
        if products:
            symbols = parse_csv_list(products) 
        else:
            # Parse instrument types
            instrument_types = None
            if types:
                instrument_types = [
                    t.strip() for t in types.split(",") if t.strip()
                ]
            
            # Parse excluded exchanges  
            exclude_exchanges = None
            if exclude_exchange:
                exclude_exchanges = [
                    e.strip() for e in exclude_exchange.split(",") if e.strip()
                ]
            
            # Use repository to get symbols (no direct SQL!)
            symbols = cvar_service.get_symbols_for_recalculation(
                five_stars=five_stars,
                ready_only=ready_only,
                include_unknown=include_unknown,
                country=country,
                instrument_types=instrument_types,
                exclude_exchanges=exclude_exchanges,
                limit=limit if limit > 0 else None
            )
        
        if not symbols:
            raise HTTPException(400, "No symbols to process")
        
        # Apply limit after symbol resolution
        if limit > 0:
            symbols = symbols[:limit]
        
        # Process symbols using domain service
        results = []
        successful_count = 0
        
        for symbol in symbols:
            result = cvar_service.process_symbol_calculation(
                symbol=symbol,
                force_recalculate=True
            )
            
            if result["status"] == "success":
                successful_count += 1
            
            # Add to detailed results if verbose
            if verbose > 0:
                results.append(result)
        
        # Return clean response
        response = {
            "mode": "refactored_local",
            "symbols_processed": len(symbols),
            "successful_calculations": successful_count,
            "failed_calculations": len(symbols) - successful_count,
            "architecture": "repository_pattern_with_domain_service"
        }
        
        if verbose > 0:
            response["detailed_results"] = results
        
        # Demonstrate cache info
        cache_info = cvar_service.get_cache_info()
        response["cache_info"] = cache_info
        
        logger.info(
            f"Refactored CVaR recalc: processed {len(symbols)} symbols, "
            f"{successful_count} successful"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refactored CVaR recalc failed: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


@router.get("/cvar/curve-refactored")
def cvar_curve_refactored(
    symbol: str = Query(..., description="Symbol to get CVaR for"),
    alpha: int = Query(95, description="Alpha level: 50, 95, or 99"),
    recalculate: bool = Query(False, description="Force recalculation"),
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """
    Refactored version of cvar_curve using new architecture.
    
    Demonstrates clean data flow without direct database access.
    """
    
    if alpha not in (50, 95, 99):
        alpha = 95
    
    # Initialize domain service
    cvar_service = CvarUnifiedService()
    
    try:
        # Get CVaR data using domain service
        data = cvar_service.get_cvar_data(
            symbol=symbol,
            force_recalculate=recalculate,
            prefer_local=True
        )
        
        if not data.get("success"):
            error_detail = data.get("error", "Calculation failed")
            code = data.get("code", "calc_failed")
            status = 422 if code == "insufficient_history" else 500
            raise HTTPException(status, error_detail)
        
        # Extract the requested alpha block
        alpha_key = f"cvar{alpha}"
        block = data.get(alpha_key, {})
        
        if not block:
            raise HTTPException(404, f"No data available for alpha {alpha}")
        
        # Build clean response
        return {
            "symbol": symbol,
            "alpha": alpha,
            "annual": block.get("annual", {}),
            "snapshot": block.get("snapshot", {}),
            "alpha_used": block.get("alpha"),
            "as_of": data.get("as_of_date"),
            "cached": data.get("cached", False),
            "architecture": "repository_pattern_with_domain_service",
            "data_source": "refactored_calculation_service"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refactored CVaR curve failed for {symbol}: {e}")
        raise HTTPException(500, f"Failed to get CVaR data: {str(e)}")


@router.get("/cvar/architecture-comparison")
def architecture_comparison(
    _auth: None = Depends(_require_pub_or_basic),
) -> dict:
    """
    Endpoint showing the difference between old and new architecture.
    """
    
    return {
        "old_architecture": {
            "description": "Direct database access in route handlers",
            "problems": [
                "SQL queries mixed with business logic",
                "No separation of concerns", 
                "Hard to test individual components",
                "Database sessions managed manually",
                "Error handling scattered throughout",
                "Code duplication across routes"
            ],
            "example_issues": [
                "35+ files directly use get_db_session()",
                "1,300+ lines in routes/cvar.py",
                "Duplicate CVaR logic in multiple places",
                "Mixed data access and presentation logic"
            ]
        },
        "new_architecture": {
            "description": "Clean separation with Repository pattern and Domain services",
            "benefits": [
                "Pure business logic in domain services",
                "Data access abstracted through repositories", 
                "Easy to test each layer independently",
                "Automatic session management",
                "Centralized error handling",
                "Single source of truth for each operation"
            ],
            "layers": {
                "routes": "Handle HTTP requests/responses only",
                "domain_services": "Pure business logic and orchestration", 
                "repositories": "Data access abstraction",
                "models": "Database entities"
            },
            "example_improvements": [
                "No direct SQL in route handlers",
                "Routes under 100 lines each",
                "Reusable business logic",
                "Clean testing interfaces"
            ]
        },
        "migration_status": {
            "repositories_created": True,
            "domain_services_created": "In progress",
            "routes_refactored": "Demo created",
            "old_code_removed": False
        }
    }
