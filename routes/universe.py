"""
Harvard Universe Management API

Provides REST endpoints for managing the frozen universe of products
for Nirvana's Harvard release.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.universe_manager import get_harvard_universe_manager
from core.universe_config import ProductCategory, UniverseFeatureFlags
from utils.auth import require_pub_or_basic as _require_pub_or_basic

router = APIRouter(prefix="/harvard-universe", tags=["Harvard Universe"])
_LOG = logging.getLogger(__name__)


class UniverseStatsResponse(BaseModel):
    """Response model for universe statistics."""
    total_products: int
    by_country: Dict[str, int]
    by_category: Dict[str, int]
    missing_mu: int
    missing_cvar: int
    last_updated: Optional[str] = None


class UniverseProductResponse(BaseModel):
    """Response model for universe product."""
    symbol: str
    name: str
    country: str
    instrument_type: str
    category: str
    has_mu: bool
    has_cvar: bool
    five_stars: bool = False
    special_lists: List[str] = []


class CompletenessRequest(BaseModel):
    """Request model for universe completeness check."""
    country: Optional[str] = Field(None, description="Process specific country only")
    force_refresh: bool = Field(False, description="Force complete refresh")


class CategoryManagementRequest(BaseModel):
    """Request model for category management."""
    country: str = Field(..., description="Country code (US, UK, CA)")
    category: str = Field(..., description="Product category")
    auto_complete: bool = Field(True, description="Auto-compute missing data")


@router.get("/stats", response_model=UniverseStatsResponse)
async def get_universe_stats():
    """Get comprehensive statistics about Harvard universe."""
    try:
        manager = get_harvard_universe_manager()
        stats = manager.get_universe_stats()
        
        return UniverseStatsResponse(
            total_products=stats.total_products,
            by_country=stats.by_country,
            by_category=stats.by_category,
            missing_mu=stats.missing_mu,
            missing_cvar=stats.missing_cvar,
            last_updated=stats.last_updated.isoformat() if stats.last_updated else None,
        )
        
    except Exception as exc:
        _LOG.error("Failed to get universe stats: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/products", response_model=List[UniverseProductResponse])
async def get_universe_products(
    country: Optional[str] = Query(None, description="Filter by country code"),
    limit: int = Query(1000, description="Maximum number of products to return"),
):
    """Get all products in Harvard universe."""
    try:
        manager = get_harvard_universe_manager()
        products = manager.get_universe_products(country)
        
        # Apply limit
        if limit > 0:
            products = products[:limit]
        
        return [
            UniverseProductResponse(
                symbol=p.symbol,
                name=p.name,
                country=p.country,
                instrument_type=p.instrument_type,
                category=p.category,
                has_mu=p.has_mu,
                has_cvar=p.has_cvar,
                five_stars=p.five_stars,
                special_lists=p.special_lists,
            )
            for p in products
        ]
        
    except Exception as exc:
        _LOG.error("Failed to get universe products: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ensure-completeness")
async def ensure_universe_completeness(request: CompletenessRequest):
    """
    Ensure universe has all required data (μ, CVaR) and anchors.
    
    This is the main maintenance operation for Harvard universe.
    """
    try:
        manager = get_harvard_universe_manager()
        
        # Set force refresh flag if requested
        if request.force_refresh:
            import os
            os.environ["HARVARD_FORCE_REFRESH"] = "true"
        
        report = manager.ensure_universe_completeness(request.country)
        
        return {
            "status": "completed",
            "report": report,
            "dry_run": UniverseFeatureFlags.dry_run_mode(),
        }
        
    except Exception as exc:
        _LOG.error("Failed to ensure universe completeness: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/validate")
async def validate_universe_integrity():
    """Validate Harvard universe integrity and consistency."""
    try:
        manager = get_harvard_universe_manager()
        validation_report = manager.validate_universe_integrity()
        
        # Return appropriate HTTP status based on validation
        if not validation_report["healthy"]:
            return {
                "status": "issues_found",
                "validation": validation_report,
            }
        
        return {
            "status": "healthy",
            "validation": validation_report,
        }
        
    except Exception as exc:
        _LOG.error("Failed to validate universe: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/validate/advanced")
async def validate_universe_advanced():
    """Perform comprehensive advanced validation of Harvard universe."""
    try:
        from services.universe_validator import get_harvard_validator
        
        validator = get_harvard_validator()
        validation_report = validator.validate_full_universe()
        
        # Return appropriate HTTP status based on validation
        summary = validation_report.get("summary", {})
        overall_health = summary.get("overall_health", "unknown")
        
        if overall_health == "validation_failed":
            raise HTTPException(status_code=500, detail="Validation process failed")
        
        return {
            "status": overall_health,
            "advanced_validation": validation_report,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        _LOG.error("Failed to perform advanced validation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/categories/add")
async def add_product_category(request: CategoryManagementRequest):
    """
    Add a product category to Harvard universe.
    
    Note: This currently logs the request but requires configuration changes.
    """
    try:
        # Validate category
        try:
            category_enum = ProductCategory(request.category.upper())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category: {request.category}"
            )
        
        manager = get_harvard_universe_manager()
        report = manager.add_product_category(
            country=request.country.upper(),
            category=category_enum,
            auto_complete=request.auto_complete,
        )
        
        return {
            "status": "logged",
            "message": "Category addition requires configuration update",
            "report": report,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        _LOG.error("Failed to add product category: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/categories/remove")
async def remove_product_category(request: CategoryManagementRequest):
    """
    Remove a product category from Harvard universe.
    
    Note: This currently logs the request but requires configuration changes.
    """
    try:
        # Validate category
        try:
            category_enum = ProductCategory(request.category.upper())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category: {request.category}"
            )
        
        manager = get_harvard_universe_manager()
        report = manager.remove_product_category(
            country=request.country.upper(),
            category=category_enum,
            recalibrate_anchors=True,
        )
        
        return {
            "status": "logged",
            "message": "Category removal requires configuration update",
            "report": report,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        _LOG.error("Failed to remove product category: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/config")
async def get_universe_configuration():
    """Get current Harvard universe configuration."""
    try:
        from core.universe_config import HarvardUniverseConfig
        
        config = HarvardUniverseConfig()
        enabled_countries = config.get_enabled_countries()
        
        config_data = {}
        for country_code, country_config in enabled_countries.items():
            config_data[country_code] = {
                "name": country_config.country_name,
                "categories": [cat.value for cat in country_config.categories],
                "compass_category": country_config.compass_category,
                "enabled": country_config.enabled,
                "min_market_cap": country_config.min_market_cap,
                "min_volume": country_config.min_volume,
                "min_history_days": country_config.min_history_days,
                "special_lists": country_config.special_lists,
            }
        
        return {
            "release_name": "Harvard",
            "countries": config_data,
            "feature_flags": {
                "auto_compute_mu": UniverseFeatureFlags.auto_compute_missing_mu(),
                "auto_compute_cvar": UniverseFeatureFlags.auto_compute_missing_cvar(),
                "auto_recalibrate_anchors": UniverseFeatureFlags.auto_recalibrate_anchors(),
                "dry_run_mode": UniverseFeatureFlags.dry_run_mode(),
                "max_workers": UniverseFeatureFlags.max_parallel_workers(),
            },
        }
        
    except Exception as exc:
        _LOG.error("Failed to get universe configuration: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/feature-flags")
async def get_feature_flags():
    """Get current feature flags for universe management."""
    return {
        "auto_compute_mu": UniverseFeatureFlags.auto_compute_missing_mu(),
        "auto_compute_cvar": UniverseFeatureFlags.auto_compute_missing_cvar(),
        "auto_recalibrate_anchors": UniverseFeatureFlags.auto_recalibrate_anchors(),
        "force_refresh": UniverseFeatureFlags.force_universe_refresh(),
        "dry_run_mode": UniverseFeatureFlags.dry_run_mode(),
        "max_workers": UniverseFeatureFlags.max_parallel_workers(),
    }


# Protected endpoints require authentication
@router.post("/admin/recalibrate-anchors")
async def recalibrate_anchors(
    country: Optional[str] = Query(None, description="Country to recalibrate"),
    _auth=_require_pub_or_basic,
):
    """
    Force recalibration of anchors for Harvard universe.
    
    This is an admin operation that should be used carefully.
    """
    try:
        from services.compass_anchors import auto_calibrate_from_db
        from core.universe_config import HarvardUniverseConfig
        
        config = HarvardUniverseConfig()
        compass_categories = config.get_compass_categories()
        
        if country:
            country_config = config.get_country_config(country.upper())
            if not country_config:
                raise HTTPException(status_code=400, detail=f"Invalid country: {country}")
            compass_categories = [country_config.compass_category]
        
        results = {}
        for compass_category in compass_categories:
            _LOG.info("Recalibrating anchors for %s", compass_category)
            success = auto_calibrate_from_db(compass_category)
            results[compass_category] = {
                "success": success,
                "timestamp": str(datetime.utcnow()),
            }
        
        return {
            "status": "completed",
            "recalibrated_categories": results,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        _LOG.error("Failed to recalibrate anchors: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
