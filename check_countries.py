#!/usr/bin/env python3
"""Utility to check Harvard universe countries in database."""

import logging
import sys
from sqlalchemy import func
from core.db import get_db_session
from core.models import Symbols
from core.universe_config import HarvardUniverseConfig

logger = logging.getLogger(__name__)

def analyze_country_distribution(log_level=logging.INFO):
    """Analyze country distribution in the database for Harvard universe.
    
    Args:
        log_level: Logging level
    
    Returns:
        Dictionary with country statistics
    """
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    
    session = get_db_session()
    if not session:
        logger.error("Failed to get database session")
        return None

    try:
        # Get country mappings from config
        config = HarvardUniverseConfig()
        country_code_map = config.get_country_code_map()
        
        # Check distinct country values
        countries = [r[0] for r in session.query(func.distinct(Symbols.country)).all()]
        logger.info("Distinct country values: %s", countries)
        
        results = {"countries": {}}
        
        # Count symbols by country and check expected mappings
        for country_norm, country_code in country_code_map.items():
            norm_count = session.query(func.count(Symbols.id)).filter(
                Symbols.country == country_norm).scalar()
            code_count = session.query(func.count(Symbols.id)).filter(
                Symbols.country == country_code).scalar()
            
            logger.info("Country '%s': %d symbols", country_norm, norm_count)
            logger.info("Country '%s': %d symbols", country_code, code_count)
            
            results["countries"][country_norm] = norm_count
            results["countries"][country_code] = code_count
            
        return results
    
    finally:
        session.close()

if __name__ == "__main__":
    analyze_country_distribution(log_level=logging.INFO)
    sys.exit(0)
