"""
Logging configuration for the Nirvana App.

This module provides a centralized configuration for all loggers in the application.
It allows setting different log levels for different components and configuring
formatters and handlers.
"""

import os
import logging
from typing import Dict, Any


def configure_logging():
    """Configure logging for the application."""
    # Get log level from environment variable
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    
    # Configure specific loggers
    loggers_config = {
        # Service loggers
        "CVAR_SERVICE": os.getenv("LOG_LEVEL_CVAR_SERVICE", "INFO").upper(),
        "COMPASS_RECOMMEND": os.getenv("LOG_LEVEL_COMPASS", "INFO").upper(),
        "sb_consumer": os.getenv("LOG_LEVEL_SB_CONSUMER", "INFO").upper(),
        "symbols_sync": os.getenv("LOG_LEVEL_SYMBOLS_SYNC", "INFO").upper(),
        
        # Azure loggers - always WARNING to reduce noise
        "azure": "WARNING",
        "azure.servicebus": "WARNING",
        "azure.servicebus._pyamqp": "WARNING",
        "uamqp": "WARNING",
        
        # Startup loggers
        "nirvana.compass": os.getenv("LOG_LEVEL_COMPASS_ANCHORS", "INFO").upper(),
        "nirvana.startup": os.getenv("LOG_LEVEL_STARTUP", "INFO").upper(),
    }
    
    # Apply configuration to loggers
    for logger_name, level_name in loggers_config.items():
        logger = logging.getLogger(logger_name)
        level = getattr(logging, level_name, log_level)
        logger.setLevel(level)
        
        # Ensure each logger has a handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        # Set propagate=False for service loggers to avoid duplicate logs
        if logger_name in ["CVAR_SERVICE", "COMPASS_RECOMMEND", "sb_consumer", "symbols_sync"]:
            logger.propagate = False


def get_logger_levels() -> Dict[str, str]:
    """Get current log levels for all configured loggers."""
    result = {}
    
    # Add root logger
    result["root"] = logging.getLevelName(logging.getLogger().level)
    
    # Add specific loggers
    for logger_name in [
        "CVAR_SERVICE", "COMPASS_RECOMMEND", "sb_consumer", "symbols_sync",
        "azure", "azure.servicebus", "nirvana.compass", "nirvana.startup"
    ]:
        logger = logging.getLogger(logger_name)
        result[logger_name] = logging.getLevelName(logger.level)
        
    return result
