"""
Harvard Release Universe Configuration

Defines the frozen universe of products for Nirvana's first release.
Provides flexible configuration for adding/removing product categories.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from enum import Enum


class ProductCategory(Enum):
    """Product categories supported in Harvard release."""
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"  
    FIVE_STARS = "FIVE_STARS"
    INDEX_SP500 = "INDEX_SP500"
    INDEX_DOW_JONES = "INDEX_DOW_JONES"
    INDEX_NASDAQ100 = "INDEX_NASDAQ100"
    COMMON_STOCK = "COMMON_STOCK"
    FTSE100 = "FTSE100"


@dataclass(frozen=True)
class CountryUniverseConfig:
    """Universe configuration for a specific country."""
    country_code: str
    country_name: str
    categories: Set[ProductCategory]
    compass_category: str  # For anchor calibration
    enabled: bool = True
    
    # Filtering criteria
    min_market_cap: Optional[float] = None
    min_volume: Optional[float] = None
    min_history_days: int = 252  # 1 year minimum
    
    # Special lists/indices
    special_lists: List[str] = None  # e.g., ["FTSE100", "SP500"]
    
    def __post_init__(self):
        if self.special_lists is None:
            object.__setattr__(self, 'special_lists', [])


class HarvardUniverseConfig:
    """Harvard Release universe configuration."""
    
    # Core countries and their product categories
    COUNTRIES = {
        "US": CountryUniverseConfig(
            country_code="US",
            country_name="United States", 
            categories={
                ProductCategory.ETF,
                ProductCategory.MUTUAL_FUND,
                ProductCategory.FIVE_STARS,
                ProductCategory.INDEX_SP500,
                ProductCategory.INDEX_DOW_JONES,
                ProductCategory.INDEX_NASDAQ100,
            },
            compass_category="HARVARD-US",
            min_market_cap=100_000_000,  # $100M minimum
            min_volume=1_000_000,  # $1M daily volume
            special_lists=["SP500", "DOW_JONES", "NASDAQ100"],
        ),
        
        "UK": CountryUniverseConfig(
            country_code="UK",
            country_name="United Kingdom",
            categories={
                ProductCategory.ETF,
                ProductCategory.COMMON_STOCK,
                ProductCategory.FTSE100,
            },
            compass_category="HARVARD-UK",
            min_market_cap=50_000_000,  # £50M minimum
            min_volume=500_000,  # £500K daily volume  
            special_lists=["FTSE100"],
        ),
        
        "CA": CountryUniverseConfig(
            country_code="CA", 
            country_name="Canada",
            categories={
                ProductCategory.ETF,
            },
            compass_category="HARVARD-CA",
            min_market_cap=25_000_000,  # CAD $25M minimum
            min_volume=250_000,  # CAD $250K daily volume
            enabled=True,  # Enable when ready
        ),
    }
    
    @classmethod
    def get_enabled_countries(cls) -> Dict[str, CountryUniverseConfig]:
        """Get all enabled country configurations."""
        return {
            code: config 
            for code, config in cls.COUNTRIES.items() 
            if config.enabled
        }
    
    @classmethod
    def get_country_config(cls, country_code: str) -> Optional[CountryUniverseConfig]:
        """Get configuration for specific country."""
        return cls.COUNTRIES.get(country_code.upper())
    
    @classmethod
    def get_compass_categories(cls) -> List[str]:
        """Get all compass categories for anchor calibration."""
        country_categories = [
            config.compass_category 
            for config in cls.COUNTRIES.values() 
            if config.enabled
        ]
        # Add global Harvard Universe category
        return country_categories + ["GLOBAL-HARVARD"]
        
    @classmethod
    def get_country_code_map(cls) -> Dict[str, str]:
        """Get mapping between normalized country names and country codes."""
        return {
            "USA": "US",
            "United Kingdom": "UK", 
            "Canada": "CA"
        }
    
    @classmethod
    def is_product_eligible(
        cls, 
        country_code: str, 
        instrument_type: str, 
        market_cap: Optional[float] = None,
        volume: Optional[float] = None,
        history_days: Optional[int] = None,
        five_stars: Optional[int] = None,
        special_list: Optional[str] = None
    ) -> bool:
        """
        Check if a product is eligible for Harvard release universe.
        
        Args:
            country_code: ISO country code
            instrument_type: Type of instrument
            market_cap: Market capitalization
            volume: Daily trading volume
            history_days: Days of price history available
            five_stars: Five stars rating (0 or 1)
            special_list: Special list membership (SP500, FTSE100, etc.)
        """
        config = cls.get_country_config(country_code)
        if not config or not config.enabled:
            return False
        
        # Map instrument type to category - more flexible matching
        category_mapping = {
            "ETF": ProductCategory.ETF,
            "Mutual Fund": ProductCategory.MUTUAL_FUND,
            "Fund": ProductCategory.MUTUAL_FUND,
            "Common Stock": ProductCategory.COMMON_STOCK,
            "Stock": ProductCategory.COMMON_STOCK,
        }
        
        # First try exact match
        product_category = category_mapping.get(instrument_type)
        
        # If no match, try to match substrings for more flexibility
        if not product_category:
            # Check if the instrument_type contains any of our known types
            for known_type, category in category_mapping.items():
                if known_type.lower() in instrument_type.lower() or instrument_type.lower() in known_type.lower():
                    product_category = category
                    break
        
        if not product_category:
            return False
            
        # Special category checks
        if five_stars == 1 and ProductCategory.FIVE_STARS in config.categories:
            product_category = ProductCategory.FIVE_STARS
            
        if special_list:
            special_categories = {
                "SP500": ProductCategory.INDEX_SP500,
                "DOW_JONES": ProductCategory.INDEX_DOW_JONES,
                "NASDAQ100": ProductCategory.INDEX_NASDAQ100,
                "FTSE100": ProductCategory.FTSE100,
            }
            if special_list in special_categories:
                product_category = special_categories[special_list]
        
        # Check if category is enabled for this country
        if product_category not in config.categories:
            return False
        
        # Apply filters
        if config.min_market_cap and market_cap and market_cap < config.min_market_cap:
            return False
            
        if config.min_volume and volume and volume < config.min_volume:
            return False
            
        if history_days and history_days < config.min_history_days:
            return False
        
        return True


# Environment-controlled feature flags
class UniverseFeatureFlags:
    """Feature flags for universe management."""
    
    @classmethod
    def auto_compute_missing_mu(cls) -> bool:
        """Auto-compute missing μ values."""
        return os.getenv("HARVARD_AUTO_COMPUTE_MU", "true").lower() == "true"
    
    @classmethod
    def auto_compute_missing_cvar(cls) -> bool:
        """Auto-compute missing CVaR values."""
        return os.getenv("HARVARD_AUTO_COMPUTE_CVAR", "false").lower() == "true"
    
    @classmethod  
    def auto_recalibrate_anchors(cls) -> bool:
        """Auto-recalibrate anchors when universe changes."""
        return os.getenv("HARVARD_AUTO_RECALIBRATE", "true").lower() == "true"
    
    @classmethod
    def force_universe_refresh(cls) -> bool:
        """Force complete universe refresh."""
        return os.getenv("HARVARD_FORCE_REFRESH", "false").lower() == "true"
    
    @classmethod
    def dry_run_mode(cls) -> bool:
        """Dry run mode - show changes without applying."""
        return os.getenv("HARVARD_DRY_RUN", "false").lower() == "true"
    
    @classmethod
    def max_parallel_workers(cls) -> int:
        """Maximum parallel workers for computations."""
        return int(os.getenv("HARVARD_MAX_WORKERS", "16"))


# Global configuration instance
def get_harvard_config() -> HarvardUniverseConfig:
    """Get Harvard release configuration."""
    return HarvardUniverseConfig()
