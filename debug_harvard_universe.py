#!/usr/bin/env python3
"""Debug Harvard universe issues."""

import logging
from sqlalchemy import func
from core.db import get_db_session
from core.models import Symbols
from services.universe_manager import get_harvard_universe_manager
from core.universe_config import HarvardUniverseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harvard_debug")

def check_countries():
    """Check country values in the database."""
    session = get_db_session()
    if not session:
        logger.error("Failed to get database session")
        return
        
    try:
        # Check distinct country values
        countries = [r[0] for r in session.query(func.distinct(Symbols.country)).all()]
        logger.info(f"Distinct country values: {countries}")

        # Count symbols by country
        for country in countries:
            if country:
                count = session.query(func.count(Symbols.id)).filter(Symbols.country == country).scalar()
                logger.info(f"Country '{country}': {count} symbols")

        # Check specifically for US, UK, CA which should be in the Harvard universe
        expected_countries = ["US", "UK", "CA", "USA", "United Kingdom", "Canada"]
        for expected in expected_countries:
            count = session.query(func.count(Symbols.id)).filter(Symbols.country == expected).scalar()
            logger.info(f"Expected country '{expected}': {count} symbols")
    finally:
        session.close()

def debug_universe_products():
    """Debug why no products are found in Harvard universe."""
    manager = get_harvard_universe_manager()
    logger.info("Harvard Universe Manager initialized")
    
    # Get config
    config = HarvardUniverseConfig()
    logger.info(f"Enabled countries in config: {list(config.get_enabled_countries().keys())}")
    
    # Try to get products without filters first
    session = get_db_session()
    if not session:
        logger.error("Failed to get database session")
        return
        
    try:
        # Check basic query without filters
        query = session.query(Symbols)
        basic_count = query.count()
        logger.info(f"Total symbols in database: {basic_count}")
        
        # Add country filter
        enabled_countries = list(config.get_enabled_countries().keys())
        query = query.filter(Symbols.country.in_(enabled_countries))
        country_filtered_count = query.count()
        logger.info(f"Symbols with countries {enabled_countries}: {country_filtered_count}")
        
        # Add more filters from the original code
        query = query.filter(
            Symbols.insufficient_history == 0,
            Symbols.symbol.isnot(None),
            Symbols.instrument_type.isnot(None),
        )
        filtered_count = query.count()
        logger.info(f"Symbols after all filters: {filtered_count}")
        
        # Check the first few results
        if filtered_count > 0:
            for row in query.limit(5).all():
                logger.info(f"Sample row: symbol={row.symbol}, country={row.country}, type={row.instrument_type}")
                # Check if product is eligible
                is_eligible = config.is_product_eligible(
                    country_code=row.country,
                    instrument_type=row.instrument_type,
                    five_stars=row.five_stars,
                )
                logger.info(f"  Is eligible: {is_eligible}")
                
    finally:
        session.close()

if __name__ == "__main__":
    logger.info("Starting Harvard universe debug")
    check_countries()
    debug_universe_products()
    logger.info("Debug complete")
