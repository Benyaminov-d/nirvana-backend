"""Data validation utilities for startup integrity checks."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import func, or_

from core.db import get_db_session
from core.models import CvarSnapshot, Symbols, CompassAnchor, Symbols

logger = logging.getLogger(__name__)


def validate_cvar_data() -> Dict[str, Any]:
    """Validate CVaR data for consistency and completeness."""
    try:
        session = get_db_session()
        if not session:
            return {"success": False, "error": "Database not available"}
            
        issues = []
        warnings = []
        stats = {}
        
        # Total CVaR snapshots
        total_snapshots = session.query(func.count(CvarSnapshot.id)).scalar()
        stats["total_snapshots"] = total_snapshots
        
        if total_snapshots < 1000:
            issues.append(f"Low CVaR data count: {total_snapshots} (expected >1000)")
            
        # Recent snapshots (last 7 days)
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_count = session.query(func.count(CvarSnapshot.id))\
            .filter(CvarSnapshot.as_of_date >= recent_date).scalar()
        stats["recent_snapshots"] = recent_count
        
        if recent_count < 100:
            warnings.append(f"Few recent CVaR snapshots: {recent_count} in last 7 days")
            
        # Check for strange CVaR values - use the correct field names
        try:
            extreme_cvar = session.query(func.count(CvarSnapshot.id)).filter(
                or_(
                    CvarSnapshot.cvar_nig > 0.9,  # >90% annual loss unlikely
                    CvarSnapshot.cvar_nig < 0.01  # <1% annual loss unlikely
                )
            ).scalar()
            stats["extreme_cvar_values"] = extreme_cvar
            
            if extreme_cvar > 0:
                warnings.append(f"Found {extreme_cvar} suspicious CVaR values")
        except Exception as e:
            logger.warning(f"Could not check extreme CVaR values: {e}")
            
        # Check for strange return values
        try:
            extreme_returns = session.query(func.count(CvarSnapshot.id)).filter(
                or_(
                    CvarSnapshot.return_annual > 1.0,   # >100% annual return unlikely
                    CvarSnapshot.return_annual < -0.8   # <-80% annual return unlikely
                )
            ).scalar()
            stats["extreme_return_values"] = extreme_returns
            
            if extreme_returns > 0:
                warnings.append(f"Found {extreme_returns} suspicious return values")
        except Exception as e:
            logger.warning(f"Could not check extreme return values: {e}")
            
        # Check top markets coverage
        top_markets = ["US", "UK", "CA"]
        market_coverage = {}
        for market in top_markets:
            try:
                count = session.query(func.count(CvarSnapshot.id))\
                    .join(Symbols, Symbols.symbol == CvarSnapshot.symbol)\
                    .filter(Symbols.country == market).scalar()
                market_coverage[market] = count
                
                if market == "US" and count < 500:
                    issues.append(f"Low US market coverage: {count} snapshots (expected >500)")
                elif market in ["UK", "CA"] and count < 100:
                    warnings.append(f"Low {market} market coverage: {count} snapshots")
            except Exception as e:
                logger.warning(f"Could not check market coverage for {market}: {e}")
                market_coverage[market] = "error"
                
        stats["market_coverage"] = market_coverage
            
        # Check missing data
        try:
            symbols_without_cvar = session.query(func.count(Symbols.symbol))\
                .outerjoin(CvarSnapshot, Symbols.symbol == CvarSnapshot.symbol)\
                .filter(Symbols.valid == 1)\
                .filter(CvarSnapshot.id.is_(None))\
                .scalar()
            stats["symbols_without_cvar"] = symbols_without_cvar
            
            if symbols_without_cvar > 100:
                warnings.append(f"Many valid symbols without CVaR data: {symbols_without_cvar}")
        except Exception as e:
            logger.warning(f"Could not check symbols without CVaR: {e}")
            
        # Stale data check
        month_ago = datetime.utcnow() - timedelta(days=30)
        stale_snapshots = session.query(func.count(CvarSnapshot.id))\
            .filter(CvarSnapshot.as_of_date < month_ago).scalar()
        stats["stale_snapshots"] = stale_snapshots
        
        if stale_snapshots > 0.8 * total_snapshots:  # >80% snapshots older than 30 days
            warnings.append(f"Mostly stale CVaR data: {stale_snapshots} of {total_snapshots} older than 30 days")
            
        return {
            "success": len(issues) == 0,
            "critical_issues": issues,
            "warnings": warnings,
            "stats": stats
        }
            
    except Exception as e:
        logger.exception("Failed to validate CVaR data")
        return {"success": False, "error": str(e)}


def validate_symbols_data() -> Dict[str, Any]:
    """Validate symbols data for consistency and completeness."""
    try:
        session = get_db_session()
        if not session:
            return {"success": False, "error": "Database not available"}
            
        issues = []
        warnings = []
        stats = {}
        
        # Total symbols
        total_symbols = session.query(func.count(Symbols.id)).scalar()
        stats["total_symbols"] = total_symbols
        
        if total_symbols < 1000:
            issues.append(f"Low symbols count: {total_symbols} (expected >1000)")
            
        # Valid symbols
        valid_symbols = session.query(func.count(Symbols.id))\
            .filter(Symbols.valid == 1).scalar()
        stats["valid_symbols"] = valid_symbols
        
        if valid_symbols < 500:
            issues.append(f"Low valid symbols count: {valid_symbols} (expected >500)")
            
        # Symbols by type
        instrument_types = ["ETF", "Mutual Fund", "Common Stock"]
        type_counts = {}
        
        for instrument_type in instrument_types:
            try:
                # Use instrument_type field instead of type
                count = session.query(func.count(Symbols.id))\
                    .filter(Symbols.instrument_type == instrument_type)\
                    .scalar()
                type_counts[instrument_type] = count
                
                if count < 100 and instrument_type in ["ETF", "Common Stock"]:
                    warnings.append(f"Low {instrument_type} count: {count}")
            except Exception as e:
                logger.warning(f"Could not count symbols of type {instrument_type}: {e}")
                type_counts[instrument_type] = "error"
                
        stats["type_counts"] = type_counts
        
        # Symbols by country
        top_countries = ["US", "UK", "CA"]
        country_counts = {}
        
        for country in top_countries:
            try:
                count = session.query(func.count(Symbols.id))\
                    .filter(Symbols.country == country)\
                    .scalar()
                country_counts[country] = count
                
                if country == "US" and count < 500:
                    issues.append(f"Low US symbols count: {count} (expected >500)")
                elif country in ["UK", "CA"] and count < 100:
                    warnings.append(f"Low {country} symbols count: {count}")
            except Exception as e:
                logger.warning(f"Could not count symbols for country {country}: {e}")
                country_counts[country] = "error"
                
        stats["country_counts"] = country_counts
        
        # Data quality checks
        try:
            missing_names = session.query(func.count(Symbols.id))\
                .filter(Symbols.name.is_(None))\
                .scalar()
            stats["missing_names"] = missing_names
            
            if missing_names > 0.1 * total_symbols:  # >10% missing names
                warnings.append(f"Many symbols missing names: {missing_names}")
                
            missing_country = session.query(func.count(Symbols.id))\
                .filter(Symbols.country.is_(None))\
                .scalar()
            stats["missing_country"] = missing_country
            
            if missing_country > 0.1 * total_symbols:  # >10% missing country
                warnings.append(f"Many symbols missing country: {missing_country}")
                
            missing_type = session.query(func.count(Symbols.id))\
                .filter(Symbols.instrument_type.is_(None))\
                .scalar()
            stats["missing_type"] = missing_type
            
            if missing_type > 0.1 * total_symbols:  # >10% missing type
                warnings.append(f"Many symbols missing instrument type: {missing_type}")
        except Exception as e:
            logger.warning(f"Could not check data quality: {e}")
            
        return {
            "success": len(issues) == 0,
            "critical_issues": issues,
            "warnings": warnings,
            "stats": stats
        }
            
    except Exception as e:
        logger.exception("Failed to validate symbols data")
        return {"success": False, "error": str(e)}


def validate_harvard_universe() -> Dict[str, Any]:
    """Validate Harvard Universe data."""
    try:
        session = get_db_session()
        if not session:
            return {"success": False, "error": "Database not available"}
            
        issues = []
        warnings = []
        stats = {}
        
        # Check if Harvard Universe manager is available
        try:
            from services.universe_manager import get_harvard_universe_manager
            manager = get_harvard_universe_manager()
            
            # Get all Harvard Universe products
            all_products = manager.get_universe_products()
            stats["total_products"] = len(all_products)
            
            if len(all_products) < 500:
                issues.append(f"Low Harvard Universe product count: {len(all_products)} (expected >500)")
                
            # Group products by country
            products_by_country = {}
            for product in all_products:
                country = product.country
                products_by_country.setdefault(country, []).append(product)
                
            for country, products in products_by_country.items():
                stats[f"{country}_count"] = len(products)
                
                if country == "US" and len(products) < 500:
                    issues.append(f"Low US symbols count for Harvard Universe: {len(products)} (expected >500)")
                elif country in ["UK", "CA"] and len(products) < 100:
                    warnings.append(f"Low {country} symbols count for Harvard Universe: {len(products)}")
                    
            # Check if Harvard anchors exist
            for country in ["US", "UK", "CA"]:
                category = f"HARVARD-{country}"
                anchor = session.query(CompassAnchor)\
                    .filter(CompassAnchor.category == category)\
                    .first()
                    
                if not anchor:
                    issues.append(f"Missing Harvard Universe anchor for {country}: {category}")
                    
            # Check if global Harvard anchor exists
            global_anchor = session.query(CompassAnchor)\
                .filter(CompassAnchor.category == "GLOBAL-HARVARD")\
                .first()
                
            if not global_anchor:
                warnings.append("Missing global Harvard Universe anchor: GLOBAL-HARVARD")
                
        except ImportError:
            issues.append("Harvard Universe manager not available")
        except Exception as e:
            issues.append(f"Failed to validate Harvard Universe: {str(e)}")
            
        return {
            "success": len(issues) == 0,
            "critical_issues": issues,
            "warnings": warnings,
            "stats": stats
        }
            
    except Exception as e:
        logger.exception("Failed to validate Harvard Universe")
        return {"success": False, "error": str(e)}


def validate_all_data() -> Dict[str, Any]:
    """Validate all data integrity."""
    results = {
        "success": True,
        "critical_issues": [],
        "warnings": [],
        "validation_results": {}
    }
    
    # Validate CVaR data
    cvar_result = validate_cvar_data()
    results["validation_results"]["cvar_data"] = cvar_result
    
    if not cvar_result.get("success", False):
        results["success"] = False
        for issue in cvar_result.get("critical_issues", []):
            results["critical_issues"].append(f"cvar_data: {issue}")
            
    # Validate symbols data
    symbols_result = validate_symbols_data()
    results["validation_results"]["symbols_data"] = symbols_result
    
    if not symbols_result.get("success", False):
        results["success"] = False
        for issue in symbols_result.get("critical_issues", []):
            results["critical_issues"].append(f"symbols_data: {issue}")
            
    # Validate Harvard Universe
    harvard_result = validate_harvard_universe()
    results["validation_results"]["harvard_universe"] = harvard_result
    
    if not harvard_result.get("success", False):
        results["success"] = False
        for issue in harvard_result.get("critical_issues", []):
            results["critical_issues"].append(f"harvard_universe: {issue}")
            
    # Collect warnings
    for source, result in results["validation_results"].items():
        for warning in result.get("warnings", []):
            results["warnings"].append(f"{source}: {warning}")
            
    return results

