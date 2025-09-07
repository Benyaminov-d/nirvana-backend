"""
Application-wide constants and configuration values.

This module centralizes all magic numbers, thresholds, and configuration
constants used throughout the application to eliminate duplication and
provide a single source of truth.
"""

import os
from typing import Dict, List


# =================== TRADING AND FINANCIAL CONSTANTS ===================

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
BUSINESS_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

# CVaR simulation defaults
DEFAULT_SIMS = 20_000
MIN_SIMS = 1_000
MAX_SIMS = 100_000

# Alpha levels for CVaR calculations
CVAR_ALPHA_LEVELS = [50, 95, 99]
DEFAULT_ALPHA_LEVEL = 99

# Risk metrics thresholds
DEFAULT_MIN_YEARS = 10.0
MIN_HISTORY_DAYS = 252  # 1 year minimum
MIN_SAMPLE_SIZE_FOR_CVAR = 100

# Price validation thresholds
ZERO_RETURN_EPSILON = 1e-8
MAX_ZERO_RETURN_SHARE_LAST252 = 0.2  # 20% max zero returns
MAX_DROP_LAST252 = 10  # Max dropped points in last 252 days


# =================== COMPASS SCORING CONSTANTS ===================

# Default Compass parameters
DEFAULT_COMPASS_LAMBDA = 2.25  # Loss-aversion parameter
DEFAULT_COMPASS_MU_LOW = 0.02   # 2% - conservative bonds
DEFAULT_COMPASS_MU_HIGH = 0.18  # 18% - excellent stocks
DEFAULT_LOSS_TOLERANCE = 0.25   # 25% default loss tolerance

# Compass scoring thresholds
MIN_SCORE_THRESHOLD = 3000
MAX_RESULTS_DEFAULT = 10
DEFAULT_MEDIAN_MU = 0.08  # 8% default median return

# Calibration parameters
WINSOR_P_LOW = 0.01   # 1st percentile
WINSOR_P_HIGH = 0.99  # 99th percentile
ANCHOR_HD_LOW = 0.05  # 5th percentile Harrell-Davis
ANCHOR_HD_HIGH = 0.95  # 95th percentile Harrell-Davis
ANCHOR_MIN_SPREAD = 0.02  # 2% minimum spread
ANCHOR_MAX_SPREAD = 0.60  # 60% maximum spread


# =================== VALIDATION AND DATA QUALITY CONSTANTS ===================

# History validation
MIN_YEARS_STRICT = 10.0
MIN_YEARS_LENIENT = 5.0
MIN_DATA_POINTS_AFTER_CLEANUP = 2

# Liquidity validation
MIN_OBSERVATIONS_PER_YEAR = 150
WEAK_YEAR_MIN_OBSERVATIONS = 150
WEAK_YEAR_MAX_OBSERVATIONS = 199
CRITICAL_YEAR_MIN_OBSERVATIONS = 150

# Anomaly detection thresholds
LONG_PLATEAU_MIN_DAYS = 20  # Consecutive equal prices
ROBUST_OUTLIER_Z_THRESHOLD = 3.0
PRICE_JUMP_THRESHOLD_PCT = 0.5  # 50% single-day jump

# Validation batch processing
DEFAULT_VALIDATION_WORKERS = 8
DEFAULT_REPROCESS_WORKERS = 8
DEFAULT_VALIDATION_YEARS = 25


# =================== API AND PERFORMANCE CONSTANTS ===================

# Rate limiting
DEFAULT_RATE_LIMIT_PER_HOUR = 200
ASSISTANT_RATE_LIMIT_PER_HOUR = 300
QUOTE_RATE_LIMIT_PER_HOUR = 200
HISTORY_RATE_LIMIT_PER_HOUR = 200

# Pagination and limits
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50
DEFAULT_SYMBOLS_LIMIT = 100
MAX_SYMBOLS_PER_REQUEST = 1000

# Cache and session management
PREWARM_COOLDOWN_SECONDS = 600
DEFAULT_PREWARM_MAX = 50
DEFAULT_WARM_WORKERS = 4
DB_MAX_AGE_DAYS = 7

# Batch processing
SB_BATCH_SIZE = 100  # Service Bus batch size
SB_SYMBOLS_PER_MESSAGE = 100
DEFAULT_MAX_WORKERS = 16


# =================== FILE AND DATA PROCESSING CONSTANTS ===================

# Price data fields
PREFERRED_PRICE_FIELD = "adjusted_close"
FALLBACK_PRICE_FIELD = "close"

# File extensions and formats
SUPPORTED_DATA_FORMATS = [".csv", ".json", ".parquet"]
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# External API constants
EODHD_API_BASE_URL = "https://eodhd.com/api"
DEFAULT_API_TIMEOUT = 30
DEFAULT_CONNECT_TIMEOUT = 10


# =================== DATABASE AND PERSISTENCE CONSTANTS ===================

# Session management
DEFAULT_SESSION_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

# Snapshot and caching
SNAPSHOT_MAX_AGE_DAYS = 30
CACHE_TTL_SECONDS = 3600  # 1 hour

# Bulk operations
BULK_INSERT_BATCH_SIZE = 1000
BULK_UPDATE_BATCH_SIZE = 500


# =================== ENVIRONMENT VARIABLE KEYS ===================

class EnvKeys:
    """Centralized environment variable keys to avoid typos and duplication."""
    
    # Core configuration
    LICENSE = "NVAR_LICENSE"
    EODHD_API_KEY = "EODHD_API_KEY"
    
    # CVaR computation
    SIMS = "NVAR_SIMS"
    LOG_PHASE = "NVAR_LOG_PHASE"
    TRADING_DAYS = "NVAR_TRADING_DAYS"
    
    # Validation
    MIN_YEARS = "NVAR_MIN_YEARS"
    ENFORCE_MIN_YEARS = "NVAR_ENFORCE_MIN_YEARS"
    EQ_LOOKBACK_DAYS = "NVAR_EQ_LOOKBACK_DAYS"
    PRICE_FIELD = "NVAR_PRICE_FIELD"
    ALLOW_CLOSE_FALLBACK = "NVAR_ALLOW_CLOSE_FALLBACK"
    
    # Liquidity diagnostics
    ZERO_RETURN_EPS = "NVAR_ZERO_RETURN_EPS"
    LAST252_MAX_ZERO_SHARE = "NVAR_LAST252_MAX_ZERO_SHARE"
    MAX_DROP_LAST252 = "NVAR_MAX_DROP_LAST252"
    
    # Workers and concurrency
    REPROCESS_WORKERS = "EXP_REPROCESS_WORKERS"
    VALIDATE_WORKERS = "EXP_VALIDATE_WORKERS"
    VALIDATE_YEARS = "EXP_VALIDATE_YEARS"
    WARM_WORKERS = "NVAR_WARM_WORKERS"
    
    # Authentication
    BASIC_AUTH_USER = "BASIC_AUTH_USER"
    BASIC_AUTH_PASS = "BASIC_AUTH_PASS"
    PUBLIC_TOKEN_SECRET = "NIR_PUBLIC_TOKEN_SECRET"
    ALLOWED_ORIGINS = "NIR_ALLOWED_ORIGINS"
    
    # OpenAI Assistant
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_ASSISTANT_ID = "OPENAI_ASSISTANT_ID"
    OPENAI_MODEL = "OPENAI_MODEL"
    OPENAI_RUN_TIMEOUT_SEC = "OPENAI_RUN_TIMEOUT_SEC"
    ASSISTANT_SYSTEM_PROMPT = "ASSISTANT_SYSTEM_PROMPT"
    
    # Email configuration
    SMTP_SERVER = "SMTP_SERVER"
    SMTP_PORT = "SMTP_PORT"
    
    # Compass configuration
    COMPASS_LAMBDA = "COMPASS_LAMBDA"
    COMPASS_MU_LOW = "COMPASS_MU_LOW"
    COMPASS_MU_HIGH = "COMPASS_MU_HIGH"
    COMPASS_DEFAULT_LOSS_TOLERANCE = "COMPASS_DEFAULT_LOSS_TOLERANCE"
    COMPASS_MIN_SCORE_THRESHOLD = "COMPASS_MIN_SCORE_THRESHOLD"
    COMPASS_MAX_RESULTS = "COMPASS_MAX_RESULTS"
    
    # Service configuration
    CVAR_SERVICE_MODE = "NVAR_CVAR_SERVICE"
    FUNC_URL = "NVAR_FUNC_URL"
    FUNC_TIMEOUT = "NVAR_FUNC_TIMEOUT"
    FUNC_CONNECT_TIMEOUT = "NVAR_FUNC_CONNECT_TIMEOUT"
    FUNC_KEY = "NVAR_FUNC_KEY"
    
    # Service Bus
    SB_BATCH = "SB_BATCH"
    SB_SYMBOLS_PER_MSG = "SB_SYMBOLS_PER_MSG"
    
    # Performance tuning
    PREWARM_COOLDOWN_S = "NVAR_PREWARM_COOLDOWN_S"
    PREWARM_MAX = "NVAR_PREWARM_MAX"
    DB_MAX_AGE_DAYS = "NVAR_DB_MAX_AGE_DAYS"
    
    # Harvard Universe
    HARVARD_MAX_WORKERS = "HARVARD_MAX_WORKERS"


# =================== INSTRUMENT TYPE MAPPINGS ===================

# Canonical instrument types
CANONICAL_INSTRUMENT_TYPES = {
    "COMMON_STOCK": "Common Stock",
    "MUTUAL_FUND": "Mutual Fund", 
    "ETF": "ETF",
    "INDEX": "Index",
    "BOND": "Bond",
    "REIT": "REIT",
    "COMMODITY": "Commodity",
    "CURRENCY": "Currency",
    "CRYPTO": "Cryptocurrency"
}

# Instrument type aliases for normalization
INSTRUMENT_TYPE_ALIASES = {
    "fund": "Mutual Fund",
    "etf": "ETF",
    "stock": "Common Stock",
    "equity": "Common Stock",
    "bond": "Bond",
    "reit": "REIT",
    "index": "Index",
    "commodity": "Commodity",
    "currency": "Currency",
    "crypto": "Cryptocurrency",
    "cryptocurrency": "Cryptocurrency"
}

# Excluded instrument types
EXCLUDED_INSTRUMENT_TYPES = {
    "Warrant", "Right", "Unit", "Depositary Receipt"
}


# =================== COUNTRY AND EXCHANGE MAPPINGS ===================

# Default country assignments
DEFAULT_COUNTRIES = {
    "US": ["NYSE", "NASDAQ", "AMEX", "OTC", "PINK"],
    "CA": ["TSX", "TSXV", "CSE"],
    "GB": ["LSE", "AIM"],
    "DE": ["XETRA", "FSE"],
    "FR": ["EPA"],
    "JP": ["TYO", "OSE"],
    "AU": ["ASX"]
}

# EODHD suffix mappings
EODHD_SUFFIXES = {
    "US": ".US",
    "CA": ".TO", 
    "GB": ".LSE",
    "DE": ".DE",
    "FR": ".PA",
    "JP": ".T",
    "AU": ".AX"
}


# =================== HTTP AND API CONSTANTS ===================

# Standard HTTP status codes
class HttpStatus:
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    RATE_LIMITED = 429
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503

# API response formats
API_SUCCESS_FORMAT = {
    "success": True,
    "data": None,
    "metadata": {}
}

API_ERROR_FORMAT = {
    "success": False,
    "error": None,
    "code": None,
    "metadata": {}
}


# =================== UTILITY FUNCTIONS FOR CONSTANTS ===================

def get_env_int(key: str, default: int) -> int:
    """Safely get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float) -> float:
    """Safely get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Safely get boolean from environment variable."""
    value = os.getenv(key, "").lower().strip()
    if not value:
        return default
    return value in ("true", "1", "yes", "on", "enabled")


def get_env_list(key: str, default: List[str], separator: str = ",") -> List[str]:
    """Get comma-separated list from environment variable."""
    value = os.getenv(key, "").strip()
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


# =================== COMPUTED CONSTANTS ===================

# Environment-driven constants (computed at import time)
SIMS = get_env_int(EnvKeys.SIMS, DEFAULT_SIMS)
TRADING_DAYS = get_env_int(EnvKeys.TRADING_DAYS, TRADING_DAYS_PER_YEAR)
LOG_PHASE = get_env_int(EnvKeys.LOG_PHASE, 1)

MIN_YEARS = get_env_float(EnvKeys.MIN_YEARS, DEFAULT_MIN_YEARS)
ENFORCE_MIN_YEARS = get_env_bool(EnvKeys.ENFORCE_MIN_YEARS, True)

# Worker counts
REPROCESS_WORKERS = get_env_int(EnvKeys.REPROCESS_WORKERS, DEFAULT_REPROCESS_WORKERS)
VALIDATE_WORKERS = get_env_int(EnvKeys.VALIDATE_WORKERS, DEFAULT_VALIDATION_WORKERS)
WARM_WORKERS = get_env_int(EnvKeys.WARM_WORKERS, DEFAULT_WARM_WORKERS)

# Compass parameters from environment
COMPASS_LAMBDA = get_env_float(EnvKeys.COMPASS_LAMBDA, DEFAULT_COMPASS_LAMBDA)
COMPASS_MU_LOW = get_env_float(EnvKeys.COMPASS_MU_LOW, DEFAULT_COMPASS_MU_LOW)
COMPASS_MU_HIGH = get_env_float(EnvKeys.COMPASS_MU_HIGH, DEFAULT_COMPASS_MU_HIGH)

# Performance settings
PREWARM_COOLDOWN_S = get_env_int(EnvKeys.PREWARM_COOLDOWN_S, PREWARM_COOLDOWN_SECONDS)
PREWARM_MAX = get_env_int(EnvKeys.PREWARM_MAX, DEFAULT_PREWARM_MAX)
DB_MAX_AGE_DAYS_ENV = get_env_int(EnvKeys.DB_MAX_AGE_DAYS, DB_MAX_AGE_DAYS)


# =================== VALIDATION CONSTANTS ===================

class ValidationSeverity:
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory:
    """Validation issue categories."""
    DATA_MISSING = "data_missing"
    DATA_STALE = "data_stale"
    CONFIGURATION = "configuration"
    CONSISTENCY = "consistency"
    LIQUIDITY = "liquidity"
    ANOMALY = "anomaly"


# =================== EXPORT ALL CONSTANTS ===================

__all__ = [
    # Trading constants
    "TRADING_DAYS_PER_YEAR",
    "BUSINESS_DAYS_PER_YEAR", 
    "CALENDAR_DAYS_PER_YEAR",
    "DEFAULT_SIMS",
    "CVAR_ALPHA_LEVELS",
    "DEFAULT_ALPHA_LEVEL",
    
    # Environment computed
    "SIMS", 
    "TRADING_DAYS",
    "LOG_PHASE",
    "MIN_YEARS",
    
    # Compass
    "COMPASS_LAMBDA",
    "COMPASS_MU_LOW", 
    "COMPASS_MU_HIGH",
    "MIN_SCORE_THRESHOLD",
    
    # Validation
    "ValidationSeverity",
    "ValidationCategory",
    "MIN_HISTORY_DAYS",
    "ZERO_RETURN_EPSILON",
    
    # API
    "HttpStatus",
    "API_SUCCESS_FORMAT",
    "API_ERROR_FORMAT",
    
    # Mappings
    "CANONICAL_INSTRUMENT_TYPES",
    "INSTRUMENT_TYPE_ALIASES",
    "DEFAULT_COUNTRIES",
    "EODHD_SUFFIXES",
    
    # Environment keys
    "EnvKeys",
    
    # Utility functions
    "get_env_int",
    "get_env_float", 
    "get_env_bool",
    "get_env_list"
]
