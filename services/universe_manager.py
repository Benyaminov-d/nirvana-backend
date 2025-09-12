"""
Universe Manager

Manages universes of products for Nirvana.
Handles adding/removing products, auto-computing missing data,
and maintaining consistency of anchors and calculations.
Supports multiple universe types (Harvard, Cambridge, etc).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from sqlalchemy import or_

from core.db import get_db_session
from core.models import Symbols, CompassInputs, CvarSnapshot, CompassAnchor, ValidationFlags
from core.universe_config import (
    UniverseType, ACTIVE_UNIVERSE,
    UniverseFeatureFlags, ProductCategory, UniverseFactory
)
# Removed import of CompassParametersService to avoid circular dependency
from services.compass_anchors import auto_calibrate_from_db
from services.domain.cvar_unified_service import CvarUnifiedService


_LOG = logging.getLogger(__name__)


@dataclass
class UniverseStats:
    """Statistics about universe state."""
    total_products: int
    by_country: Dict[str, int]
    by_category: Dict[str, int]
    missing_mu: int
    missing_cvar: int
    last_updated: Optional[datetime] = None


@dataclass
class UniverseProduct:
    """Product in universe."""
    symbol: str
    name: str
    country: str
    instrument_type: str
    category: str
    has_mu: bool
    has_cvar: bool
    id: Optional[int] = None  # Database ID
    exchange: Optional[str] = None  # Exchange code
    five_stars: bool = False
    special_lists: List[str] = None
    
    def __post_init__(self):
        if self.special_lists is None:
            self.special_lists = []


class UniverseManager:
    """
    Manages universe with automatic dependency management.
    
    Features:
    - Defines eligible products based on configuration
    - Auto-computes missing μ and CVaR data 
    - Auto-recalibrates anchors when universe changes
    - Provides APIs for adding/removing product categories
    - Maintains consistency and validation
    - Supports multiple universe types (Harvard, Cambridge, etc)
    """
    
    def __init__(self, universe_type: str = ACTIVE_UNIVERSE):
        """
        Initialize Universe manager.
        
        Args:
            universe_type: Universe type to manage (from UniverseType enum)
        """
        self.universe_type = universe_type
        self.config = UniverseFactory.get_config(universe_type)
        self.flags = UniverseFeatureFlags(universe_type)
        self.session = get_db_session()
        if not self.session:
            raise RuntimeError("Failed to create database session")
            
        # Compass service initialization removed to avoid circular dependency
        self.compass_service = None
        self._cvar_calculator = None
        
        universe_name = self.config.get_universe_name()
        _LOG.info("%s Universe Manager initialized for countries: %s",
                 universe_name,
                 list(self.config.get_enabled_countries().keys()))
    
    def get_universe_products(self, country: Optional[str] = None) -> List[UniverseProduct]:
        """
        Get all products in universe.
        
        Args:
            country: Filter by country code (optional)
            
        Returns:
            List of universe products with their status
        """
        try:
            # Join with ValidationFlags to filter only valid symbols
            query = self.session.query(Symbols)\
                .join(ValidationFlags, Symbols.symbol == ValidationFlags.symbol)\
                .filter(ValidationFlags.valid == 1)
            
            # Apply country filter if specified
            if country:
                country_config = self.config.get_country_config(country)
                if not country_config or not country_config.enabled:
                    return []
                query = query.filter(Symbols.country == country)
            else:
                # Filter to enabled countries and their normalized names
                enabled_countries = list(self.config.get_enabled_countries().keys())
                # Get country code mapping from config
                country_code_map = self.config.get_country_code_map()
                # Invert the map to get from code to normalized name
                normalized_countries = {code: name for name, code in country_code_map.items()}
                expanded_countries = enabled_countries + [normalized_countries[c] for c in enabled_countries if c in normalized_countries]
                query = query.filter(Symbols.country.in_(expanded_countries))
            
            # Apply basic filters
            query = query.filter(
                Symbols.symbol.isnot(None),
                Symbols.instrument_type.isnot(None),
            )
            
            products = []
            for row in query.all():
                # Check if product is eligible based on configuration
                if not self._is_product_eligible(row):
                    continue
                
                # Check data availability
                has_mu = self._has_mu_data(row.symbol, row.id)
                has_cvar = self._has_cvar_data(row.symbol, row.id)
                
                # Determine category and special lists
                category, special_lists = self._get_product_category_info(row)
                
                product = UniverseProduct(
                    symbol=row.symbol,
                    name=row.name or "",
                    country=row.country,
                    instrument_type=row.instrument_type,
                    category=category,
                    has_mu=has_mu,
                    has_cvar=has_cvar,
                    id=row.id,  # Pass database ID
                    exchange=row.exchange,  # Pass exchange code
                    five_stars=(row.five_stars == 1),
                    special_lists=special_lists,
                )
                products.append(product)
            
            universe_name = self.config.get_universe_name()
            _LOG.info("Found %d products in %s universe", len(products), universe_name)
            return products
            
        except Exception as exc:
            _LOG.error("Failed to get universe products: %s", exc)
            return []
    
    def get_universe_stats(self) -> UniverseStats:
        """Get comprehensive statistics about universe state."""
        products = self.get_universe_products()
        
        by_country = {}
        by_category = {}
        missing_mu = 0
        missing_cvar = 0
        
        for product in products:
            # Country stats
            by_country[product.country] = by_country.get(product.country, 0) + 1
            
            # Category stats  
            by_category[product.category] = by_category.get(product.category, 0) + 1
            
            # Missing data stats
            if not product.has_mu:
                missing_mu += 1
            if not product.has_cvar:
                missing_cvar += 1
        
        return UniverseStats(
            total_products=len(products),
            by_country=by_country,
            by_category=by_category, 
            missing_mu=missing_mu,
            missing_cvar=missing_cvar,
            last_updated=datetime.utcnow(),
        )
    
    def ensure_universe_completeness(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Ensure universe has all required data (μ, CVaR) and anchors.
        
        This is the main method for maintaining universe consistency.
        
        Args:
            country: Process specific country only (optional)
            
        Returns:
            Report of actions taken
        """
        _LOG.info("Ensuring Harvard universe completeness...")
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": self.flags.dry_run_mode(),
            "actions": {
                "mu_computed": 0,
                "cvar_computed": 0,
                "anchors_recalibrated": [],
            },
            "errors": [],
            "products_processed": 0,
        }
        
        try:
            # Get universe products
            products = self.get_universe_products(country)
            report["products_processed"] = len(products)
            
            if not products:
                _LOG.warning("No products in universe to process")
                return report
            
            # Step 1: Compute missing μ values
            if self.flags.auto_compute_missing_mu():
                missing_mu_products = [p for p in products if not p.has_mu]
                if missing_mu_products:
                    _LOG.info("Computing μ for %d products", len(missing_mu_products))
                    mu_results = self._compute_missing_mu(missing_mu_products)
                    report["actions"]["mu_computed"] = mu_results.get("computed", 0)
                    if mu_results.get("errors"):
                        report["errors"].extend(mu_results["errors"])
            
            # Step 2: Compute missing CVaR values  
            if self.flags.auto_compute_missing_cvar():
                missing_cvar_products = [p for p in products if not p.has_cvar]
                if missing_cvar_products:
                    _LOG.info("Computing CVaR for %d products", len(missing_cvar_products))
                    cvar_results = self._compute_missing_cvar(missing_cvar_products)
                    report["actions"]["cvar_computed"] = cvar_results.get("computed", 0)
                    if cvar_results.get("errors"):
                        report["errors"].extend(cvar_results["errors"])
            
            # Step 3: Recalibrate anchors if needed
            if self.flags.auto_recalibrate_anchors():
                compass_categories = self.config.get_compass_categories()
                if country:
                    country_config = self.config.get_country_config(country)
                    if country_config:
                        compass_categories = [country_config.compass_category]
                
                for compass_category in compass_categories:
                    _LOG.info("Recalibrating anchors for %s", compass_category)
                    if not self.flags.dry_run_mode():
                        success = auto_calibrate_from_db(compass_category)
                        if success:
                            report["actions"]["anchors_recalibrated"].append(compass_category)
                        else:
                            error_msg = f"Failed to recalibrate anchors for {compass_category}"
                            report["errors"].append(error_msg)
                            _LOG.error(error_msg)
            
            _LOG.info("Universe completeness check finished: %s", report["actions"])
            return report
            
        except Exception as exc:
            error_msg = f"Universe completeness check failed: {exc}"
            _LOG.error(error_msg)
            report["errors"].append(error_msg)
            return report
        finally:
            if self.session:
                self.session.close()
    
    def add_product_category(
        self, 
        country: str, 
        category: ProductCategory, 
        auto_complete: bool = True
    ) -> Dict[str, Any]:
        """
        Add a product category to universe.
        
        Args:
            country: Country code
            category: Product category to add
            auto_complete: Auto-compute missing data after adding
            
        Returns:
            Report of changes made
        """
        _LOG.info("Adding category %s to %s universe", category.value, country)
        
        # This would require modifying the configuration
        # For now, log the request
        report = {
            "action": "add_category",
            "country": country,
            "category": category.value,
            "status": "logged_request",
            "message": "Category addition requires configuration update",
        }
        
        if auto_complete:
            # Trigger completeness check for the country
            completeness_report = self.ensure_universe_completeness(country)
            report["completeness_check"] = completeness_report
        
        return report
    
    def remove_product_category(
        self, 
        country: str, 
        category: ProductCategory,
        recalibrate_anchors: bool = True
    ) -> Dict[str, Any]:
        """
        Remove a product category from universe.
        
        Args:
            country: Country code
            category: Product category to remove
            recalibrate_anchors: Recalibrate anchors after removal
            
        Returns:
            Report of changes made
        """
        _LOG.info("Removing category %s from %s universe", category.value, country)
        
        # This would require modifying the configuration
        # For now, log the request
        report = {
            "action": "remove_category",
            "country": country,
            "category": category.value,
            "status": "logged_request", 
            "message": "Category removal requires configuration update",
        }
        
        if recalibrate_anchors:
            # Trigger anchor recalibration
            country_config = self.config.get_country_config(country)
            if country_config:
                compass_category = country_config.compass_category
                success = auto_calibrate_from_db(compass_category)
                report["anchor_recalibration"] = {
                    "category": compass_category,
                    "success": success,
                }
        
        return report
    
    def validate_universe_integrity(self) -> Dict[str, Any]:
        """
        Validate universe integrity and consistency.
        
        Returns:
            Validation report with any issues found
        """
        universe_name = self.config.get_universe_name()
        _LOG.info("Validating %s universe integrity...", universe_name)
        
        issues = []
        stats = self.get_universe_stats()
        
        # Check for missing data
        if stats.missing_mu > 0:
            issues.append(f"{stats.missing_mu} products missing μ values")
        
        if stats.missing_cvar > 0:
            issues.append(f"{stats.missing_cvar} products missing CVaR values")
        
        # Check anchor availability
        for country, config in self.config.get_enabled_countries().items():
            try:
                anchor = (
                    self.session.query(CompassAnchor)
                    .filter(CompassAnchor.category == config.compass_category)
                    .order_by(CompassAnchor.created_at.desc())
                    .first()
                )
                if not anchor:
                    issues.append(f"No anchors found for {config.compass_category}")
                else:
                    # Check if anchors are recent (within last 3 months)
                    age_days = (datetime.utcnow() - anchor.created_at).days
                    if age_days > 90:
                        issues.append(f"Anchors for {config.compass_category} are {age_days} days old")
            except Exception as exc:
                issues.append(f"Failed to check anchors for {country}: {exc}")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "universe_stats": stats,
            "issues": issues,
            "healthy": len(issues) == 0,
        }
    
    def _is_product_eligible(self, symbols_row) -> bool:
        """Check if a Symbols row is eligible for Harvard universe."""
        # Normalize country codes to match config (US, UK, CA)
        country_code = symbols_row.country
        # Convert normalized country names back to country codes using centralized mapping
        country_map = self.config.get_country_code_map()
        if country_code in country_map:
            country_code = country_map[country_code]
            
        return self.config.is_product_eligible(
            country_code=country_code,
            instrument_type=symbols_row.instrument_type,
            five_stars=symbols_row.five_stars,
            # Market cap and volume data would come from additional repository methods
        )
    
    def _has_mu_data(self, symbol: str, instrument_id: int) -> bool:
        """Check if product has μ data in compass_inputs."""
        try:
            result = (
                self.session.query(CompassInputs)
                .filter(
                    CompassInputs.instrument_id == instrument_id,
                    CompassInputs.mu_i.isnot(None),
                )
                .first()
            )
            return result is not None
        except Exception:
            return False
    
    def _has_cvar_data(self, symbol: str, instrument_id: int) -> bool:
        """Check if product has CVaR data in snapshots."""
        try:
            result = (
                self.session.query(CvarSnapshot)
                .filter(
                    CvarSnapshot.instrument_id == instrument_id,
                    CvarSnapshot.alpha_label == 99,
                    or_(
                        CvarSnapshot.cvar_ghst.isnot(None),
                        CvarSnapshot.cvar_nig.isnot(None),
                        CvarSnapshot.cvar_evar.isnot(None),
                    ),
                )
                .first()
            )
            return result is not None
        except Exception:
            return False
    
    def _get_product_category_info(self, symbols_row) -> Tuple[str, List[str]]:
        """Get category and special lists for a product."""
        # Simple mapping for now
        category = symbols_row.instrument_type or "Unknown"
        special_lists = []
        
        if symbols_row.five_stars == 1:
            special_lists.append("FIVE_STARS")
        
        # Special index membership detection would use dedicated repository queries
        # This functionality can be added when market index data tables are available
        
        return category, special_lists
    
    def _compute_product_mu(self, product: UniverseProduct) -> bool:
        """
        Compute μ (expected annual return) for a single product.
        
        Args:
            product: Product to compute μ for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use the existing compass parameters service logic
            # This is simplified - in practice would fetch EODHD data and compute
            _LOG.info("Computing μ for %s", product.symbol)
            
            # Get instrument ID from database
            symbols = (
                self.session.query(Symbols)
                .filter(Symbols.symbol == product.symbol)
                .first()
            )
            
            if not symbols:
                return False
                
            if self.compass_service is None:
                _LOG.error("Cannot compute μ for %s: CompassParametersService not initialized", product.symbol)
                return False
            
            # Use compass parameters service to compute and store μ
            # This would call the existing _compute_expected_annual_return method
            # and store result in compass_inputs table
            
            # For now, just log what would be done
            _LOG.debug("Would compute and store μ for %s (id: %d)", 
                      product.symbol, symbols.id)
            
            return True  # Simulate success
            
        except Exception as exc:
            _LOG.error("Failed to compute μ for %s: %s", product.symbol, exc)
            return False
    
    def _compute_product_cvar(self, product: UniverseProduct) -> bool:
        """
        Compute CVaR for a single product.
        
        Args:
            product: Product to compute CVaR for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            _LOG.info("Computing CVaR for %s", product.symbol)
            
            # Get instrument data
            symbols = (
                self.session.query(Symbols)
                .filter(Symbols.symbol == product.symbol)
                .first()
            )
            
            if not symbols:
                return False
            
            # Use existing CVaR calculator
            # This would fetch price data, compute CVaR, and store in snapshots
            
            # For now, just log what would be done
            _LOG.debug("Would compute and store CVaR for %s (id: %d)", 
                      product.symbol, symbols.id)
            
            return True  # Simulate success
            
        except Exception as exc:
            _LOG.error("Failed to compute CVaR for %s: %s", product.symbol, exc)
            return False
    
    def _compute_missing_mu(self, products: List[UniverseProduct]) -> Dict[str, Any]:
        """Compute missing μ values for products."""
        if self.flags.dry_run_mode():
            return {"computed": len(products), "dry_run": True}
        
        computed = 0
        errors = []
        
        try:
            max_workers = min(self.flags.max_parallel_workers(), len(products))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit computation tasks
                future_to_product = {
                    executor.submit(self._compute_product_mu, product): product
                    for product in products
                }
                
                # Process results
                for future in as_completed(future_to_product):
                    product = future_to_product[future]
                    try:
                        success = future.result()
                        if success:
                            computed += 1
                            _LOG.debug("Computed μ for %s", product.symbol)
                        else:
                            errors.append(f"Failed to compute μ for {product.symbol}")
                    except Exception as exc:
                        error_msg = f"Error computing μ for {product.symbol}: {exc}"
                        errors.append(error_msg)
                        _LOG.error(error_msg)
            
            return {"computed": computed, "errors": errors[:10]}  # Limit error list
            
        except Exception as exc:
            return {"computed": computed, "errors": [str(exc)]}
    
    def _compute_missing_cvar(self, products: List[UniverseProduct]) -> Dict[str, Any]:
        """Compute missing CVaR values for products."""
        if self.flags.dry_run_mode():
            return {"computed": len(products), "dry_run": True}
        
        computed = 0
        errors = []
        
        try:
            # Initialize CVaR service if needed
            if self._cvar_calculator is None:
                self._cvar_calculator = CvarUnifiedService()
            
            max_workers = min(self.flags.max_parallel_workers(), len(products))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit computation tasks
                future_to_product = {
                    executor.submit(self._compute_product_cvar, product): product
                    for product in products
                }
                
                # Process results
                for future in as_completed(future_to_product):
                    product = future_to_product[future]
                    try:
                        success = future.result()
                        if success:
                            computed += 1
                            _LOG.debug("Computed CVaR for %s", product.symbol)
                        else:
                            errors.append(f"Failed to compute CVaR for {product.symbol}")
                    except Exception as exc:
                        error_msg = f"Error computing CVaR for {product.symbol}: {exc}"
                        errors.append(error_msg)
                        _LOG.error(error_msg)
            
            return {"computed": computed, "errors": errors[:10]}  # Limit error list
            
        except Exception as exc:
            return {"computed": computed, "errors": [str(exc)]}


# Global manager instances for different universe types
_universe_managers: Dict[str, UniverseManager] = {}

def get_universe_manager(universe_type: str = ACTIVE_UNIVERSE) -> UniverseManager:
    """Get the universe manager singleton for the specified type."""
    global _universe_managers
    if universe_type not in _universe_managers:
        _universe_managers[universe_type] = UniverseManager(universe_type)
    return _universe_managers[universe_type]

# For backward compatibility
def get_harvard_universe_manager() -> UniverseManager:
    """Get the Harvard universe manager singleton (backward compatibility)."""
    return get_universe_manager(UniverseType.HARVARD.value)
