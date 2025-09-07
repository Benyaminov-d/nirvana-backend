"""
Common type definitions used throughout the application.

This module centralizes TypedDict definitions, Enums, and other type
definitions to ensure consistency and avoid duplication.
"""

from __future__ import annotations
from typing import TypedDict, Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, date
from enum import Enum
from dataclasses import dataclass


# =================== CVAR AND FINANCIAL TYPES ===================

class CvarResult(TypedDict):
    """CVaR calculation result structure."""
    success: bool
    symbol: str
    as_of_date: str
    start_date: Optional[str]
    data_summary: Dict[str, Any]
    cached: bool
    execution_mode: str
    cvar50: Optional[Dict[str, Any]]
    cvar95: Optional[Dict[str, Any]]
    cvar99: Optional[Dict[str, Any]]
    anomalies_report: Optional[Dict[str, Any]]
    calculated_at: Optional[str]
    service_info: Optional[Dict[str, Any]]


class CvarBlock(TypedDict):
    """Individual CVaR block (50, 95, or 99)."""
    alpha: float
    annual: Dict[str, Optional[float]]  # nig, ghst, evar
    snapshot: Dict[str, Optional[float]]  # nig, ghst, evar


class CvarAnnualValues(TypedDict):
    """Annual CVaR values from different methods."""
    nig: Optional[float]
    ghst: Optional[float] 
    evar: Optional[float]


class PriceData(TypedDict):
    """Price data structure from price loader."""
    success: bool
    symbol: str
    prices: Optional[List[float]]
    returns: Optional[List[float]]
    dates: Optional[List[str]]
    as_of_date: str
    start_date: Optional[str]
    summary: Dict[str, Any]
    error: Optional[str]
    code: Optional[str]


# =================== VALIDATION TYPES ===================

class ValidationResult(TypedDict):
    """Validation result structure."""
    success: bool
    symbol: str
    country: Optional[str]
    validation_flags: Dict[str, bool]
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ValidationIssue:
    """Represents a validation issue found in data."""
    severity: str  # error, warning, info
    category: str  # data_missing, data_stale, configuration, consistency
    description: str
    affected_items: List[str]
    recommended_action: str = ""

    def __post_init__(self):
        if self.affected_items is None:
            self.affected_items = []


@dataclass
class DataQualityReport:
    """Comprehensive data quality analysis."""
    total_symbols: int
    valid_symbols: int
    warning_symbols: int
    error_symbols: int
    issues: List[ValidationIssue]
    categories_affected: Dict[str, int]
    severity_breakdown: Dict[str, int]
    recommendations: List[str]


# =================== COMPASS TYPES ===================

class CompassScore(TypedDict):
    """Compass scoring result."""
    symbol: str
    name: str
    score: float
    rank: int
    mu: Optional[float]
    cvar: Optional[float]
    loss_tolerance: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class CompassAnchor:
    """Compass anchor configuration."""
    category: str
    version: str
    mu_low: float
    mu_high: float
    created_at: Optional[datetime] = None


class CompassConfig(TypedDict):
    """Compass configuration parameters."""
    lambda_param: float  # Loss aversion
    mu_low: float       # Lower anchor
    mu_high: float      # Upper anchor
    min_score_threshold: float
    max_results: int
    default_loss_tolerance: float


# =================== API RESPONSE TYPES ===================

class ApiResponse(TypedDict):
    """Standard API response structure."""
    success: bool
    data: Optional[Any]
    error: Optional[str]
    code: Optional[str]
    metadata: Dict[str, Any]


class PaginatedResponse(TypedDict):
    """Paginated API response."""
    success: bool
    data: List[Any]
    pagination: Dict[str, Any]
    total_count: int
    metadata: Dict[str, Any]


class BatchOperationResult(TypedDict):
    """Result of batch operation."""
    total_requested: int
    successful: int
    failed: int
    skipped: int
    results: List[Dict[str, Any]]
    errors: List[str]
    execution_time_ms: float


# =================== SYMBOL AND MARKET DATA TYPES ===================

class SymbolInfo(TypedDict):
    """Symbol information structure."""
    symbol: str
    name: str
    country: Optional[str]
    exchange: Optional[str]
    instrument_type: Optional[str]
    five_stars: bool
    insufficient_history: Optional[int]
    valid: Optional[int]


class MarketQuote(TypedDict):
    """Market quote data structure."""
    symbol: str
    name: Optional[str]
    price: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    volume: Optional[int]
    timestamp: Optional[str]
    market_cap: Optional[float]
    pe_ratio: Optional[float]


class HistoricalDataPoint(TypedDict):
    """Single historical data point."""
    date: str
    open: Optional[float]
    high: Optional[float] 
    low: Optional[float]
    close: Optional[float]
    adjusted_close: Optional[float]
    volume: Optional[int]


# =================== QUERY FILTER TYPES ===================

class SymbolFilters(TypedDict, total=False):
    """Filters for symbol queries."""
    five_stars: bool
    ready_only: bool
    include_unknown: bool
    country: Optional[str]
    instrument_types: Optional[List[str]]
    exclude_exchanges: Optional[List[str]]
    limit: Optional[int]


class CvarFilters(TypedDict, total=False):
    """Filters for CVaR queries."""
    alpha_labels: Optional[List[int]]
    max_age_days: Optional[int]
    country: Optional[str]
    five_stars: bool
    instrument_types: Optional[List[str]]


# =================== BATCH PROCESSING TYPES ===================

class BatchJob(TypedDict):
    """Batch job configuration."""
    job_id: str
    job_type: str
    symbols: List[str]
    parameters: Dict[str, Any]
    max_workers: int
    created_at: str
    status: str  # pending, running, completed, failed


class JobStatus(TypedDict):
    """Job execution status."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    processed: int
    total: int
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    results: Optional[Dict[str, Any]]


# =================== CONFIGURATION TYPES ===================

class DatabaseConfig(TypedDict):
    """Database configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int
    max_overflow: int


class ApiConfig(TypedDict):
    """API configuration."""
    host: str
    port: int
    debug: bool
    cors_origins: List[str]
    rate_limits: Dict[str, int]


# =================== ENUMS ===================

class ExecutionMode(Enum):
    """CVaR execution modes."""
    LOCAL = "local"
    REMOTE = "remote"
    AUTO = "auto"


class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(Enum):
    """Validation categories."""
    DATA_MISSING = "data_missing"
    DATA_STALE = "data_stale"
    CONFIGURATION = "configuration"
    CONSISTENCY = "consistency"
    LIQUIDITY = "liquidity"
    ANOMALY = "anomaly"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InstrumentType(Enum):
    """Standard instrument types."""
    COMMON_STOCK = "Common Stock"
    MUTUAL_FUND = "Mutual Fund"
    ETF = "ETF"
    INDEX = "Index"
    BOND = "Bond"
    REIT = "REIT"
    COMMODITY = "Commodity"
    CURRENCY = "Currency"
    CRYPTOCURRENCY = "Cryptocurrency"


class Country(Enum):
    """Supported countries."""
    US = "US"
    CA = "CA" 
    GB = "GB"
    DE = "DE"
    FR = "FR"
    JP = "JP"
    AU = "AU"


# =================== UTILITY TYPE FUNCTIONS ===================

def is_valid_cvar_result(data: Any) -> bool:
    """Check if data matches CvarResult structure."""
    if not isinstance(data, dict):
        return False
    
    required_fields = {"success", "symbol"}
    return all(field in data for field in required_fields)


def create_api_response(
    success: bool,
    data: Any = None,
    error: str = None,
    code: str = None,
    **metadata
) -> ApiResponse:
    """Create standardized API response."""
    return {
        "success": success,
        "data": data,
        "error": error,
        "code": code,
        "metadata": metadata
    }


def create_batch_result(
    total_requested: int,
    results: List[Dict[str, Any]],
    errors: List[str] = None,
    execution_time_ms: float = 0.0
) -> BatchOperationResult:
    """Create batch operation result."""
    successful = len([r for r in results if r.get("success", False)])
    failed = len([r for r in results if not r.get("success", True)])
    
    return {
        "total_requested": total_requested,
        "successful": successful,
        "failed": failed,
        "skipped": max(0, total_requested - len(results)),
        "results": results,
        "errors": errors or [],
        "execution_time_ms": execution_time_ms
    }


# =================== TYPE ALIASES ===================

# Common type aliases for convenience
SymbolList = List[str]
CountryCode = str
ExchangeCode = str
AlphaLevel = int  # 50, 95, 99
CvarValue = Optional[float]
Timestamp = Union[str, datetime, date]
JsonDict = Dict[str, Any]
FilterDict = Dict[str, Any]

# Function signature types
CvarCalculationFunction = Callable[[str, bool, Optional[str], bool], CvarResult]
ValidationFunction = Callable[[str, Optional[str]], ValidationResult]
QueryBuilderFunction = Callable[..., List[Dict[str, Any]]]


# =================== EXPORT ALL TYPES ===================

__all__ = [
    # Core financial types
    "CvarResult",
    "CvarBlock", 
    "CvarAnnualValues",
    "PriceData",
    
    # Validation types
    "ValidationResult",
    "ValidationIssue",
    "DataQualityReport",
    
    # Compass types
    "CompassScore",
    "CompassAnchor",
    "CompassConfig",
    
    # API types
    "ApiResponse",
    "PaginatedResponse", 
    "BatchOperationResult",
    
    # Market data types
    "SymbolInfo",
    "MarketQuote",
    "HistoricalDataPoint",
    
    # Filter types
    "SymbolFilters",
    "CvarFilters",
    
    # Job types
    "BatchJob",
    "JobStatus",
    
    # Config types
    "DatabaseConfig",
    "ApiConfig",
    
    # Enums
    "ExecutionMode",
    "ValidationSeverity",
    "ValidationCategory", 
    "InstrumentType",
    "Country",
    
    # Utility functions
    "is_valid_cvar_result",
    "create_api_response",
    "create_batch_result",
    
    # Type aliases
    "SymbolList",
    "CountryCode",
    "ExchangeCode", 
    "AlphaLevel",
    "CvarValue",
    "Timestamp",
    "JsonDict",
    "FilterDict",
    "CvarCalculationFunction",
    "ValidationFunction",
    "QueryBuilderFunction"
]
