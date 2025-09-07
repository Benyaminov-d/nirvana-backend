"""
Centralized configuration management for Nirvana App.

This package provides environment-specific configuration classes that consolidate
all application settings in one place.

Usage:
    from config import get_config
    
    config = get_config()  # Auto-detects environment
    # or
    config = get_config('development')  # Explicit environment
    
    print(config.database.host)
    print(config.cvars.min_years)
"""

from config.base import BaseConfig
from config.development import DevelopmentConfig  
from config.production import ProductionConfig
from config.testing import TestingConfig

import os
from typing import Type


def get_config(environment: str = None) -> BaseConfig:
    """
    Get configuration instance based on environment.
    
    Args:
        environment: Environment name ('development', 'production', 'testing')
                    If None, auto-detects from NIRVANA_ENV environment variable
    
    Returns:
        Configuration instance for the specified environment
    """
    if environment is None:
        environment = os.getenv('NIRVANA_ENV', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
        'dev': DevelopmentConfig,
        'prod': ProductionConfig,
        'test': TestingConfig,
    }
    
    config_class = config_map.get(environment, DevelopmentConfig)
    return config_class()


# Convenience function for backward compatibility
def get_settings() -> BaseConfig:
    """Legacy function name - use get_config() instead."""
    return get_config()


__all__ = [
    'BaseConfig',
    'DevelopmentConfig', 
    'ProductionConfig',
    'TestingConfig',
    'get_config',
    'get_settings'  # Legacy
]

