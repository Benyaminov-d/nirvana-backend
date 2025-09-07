"""
Common utility functions used throughout the application.

This module provides reusable utility functions for formatting, parsing,
data manipulation, and other common operations to eliminate code duplication.
"""

import os
import re
import json
import time
import hashlib
import logging
import functools
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from shared.constants import (
    LOG_TIMESTAMP_FORMAT,
    EODHD_SUFFIXES,
    DEFAULT_COUNTRIES,
    INSTRUMENT_TYPE_ALIASES,
    CANONICAL_INSTRUMENT_TYPES,
    API_SUCCESS_FORMAT,
    API_ERROR_FORMAT
)
from shared.exceptions import DataValidationError


# =================== STRING UTILITIES ===================

def safe_string(value: Any, default: str = "") -> str:
    """
    Safely convert value to string.
    
    Args:
        value: Value to convert
        default: Default string if value is None/empty
        
    Returns:
        String representation
    """
    if value is None:
        return default
    
    try:
        return str(value).strip()
    except Exception:
        return default


def normalize_string(value: Any, lowercase: bool = False) -> Optional[str]:
    """
    Normalize string by trimming and optionally lowercasing.
    
    Args:
        value: Value to normalize
        lowercase: Whether to convert to lowercase
        
    Returns:
        Normalized string or None if empty
    """
    if not value:
        return None
    
    normalized = safe_string(value).strip()
    if not normalized:
        return None
    
    return normalized.lower() if lowercase else normalized


def truncate_string(value: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        value: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if not value or len(value) <= max_length:
        return value
    
    if max_length <= len(suffix):
        return suffix[:max_length]
    
    return value[:max_length - len(suffix)] + suffix


def clean_symbol(symbol: str) -> str:
    """
    Clean and normalize symbol format.
    
    Args:
        symbol: Symbol to clean
        
    Returns:
        Cleaned symbol
    """
    if not symbol:
        return ""
    
    # Remove extra whitespace and convert to uppercase
    cleaned = re.sub(r'\s+', '', str(symbol).upper())
    
    # Remove any non-alphanumeric except dots, dashes, underscores
    cleaned = re.sub(r'[^A-Z0-9._-]', '', cleaned)
    
    # Remove leading/trailing separators
    cleaned = cleaned.strip('._-')
    
    return cleaned


def pluralize(count: int, singular: str, plural: str = None) -> str:
    """
    Return singular or plural form based on count.
    
    Args:
        count: Number to check
        singular: Singular form
        plural: Plural form (defaults to singular + 's')
        
    Returns:
        Appropriate form with count
    """
    if plural is None:
        plural = singular + 's'
    
    form = singular if count == 1 else plural
    return f"{count} {form}"


# =================== NUMERIC UTILITIES ===================

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in ('n/a', 'null', 'none', ''):
                return default
        
        result = float(value)
        return result if not (result != result) else default  # Check for NaN
    except (ValueError, TypeError, OverflowError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Safely convert value to integer.
    
    Args:
        value: Value to convert
        default: Default if conversion fails
        
    Returns:
        Integer value or default
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in ('n/a', 'null', 'none', ''):
                return default
        
        return int(float(value))  # Convert via float to handle "1.0" strings
    except (ValueError, TypeError, OverflowError):
        return default


def format_number(
    value: Optional[float], 
    decimals: int = 2, 
    percentage: bool = False,
    compact: bool = False
) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        percentage: Whether to format as percentage
        compact: Whether to use compact notation (K, M, B)
        
    Returns:
        Formatted number string
    """
    if value is None:
        return "N/A"
    
    try:
        if percentage:
            value = value * 100
            suffix = "%"
        else:
            suffix = ""
        
        if compact and abs(value) >= 1000:
            if abs(value) >= 1_000_000_000:
                value = value / 1_000_000_000
                suffix = f"B{suffix}"
            elif abs(value) >= 1_000_000:
                value = value / 1_000_000
                suffix = f"M{suffix}"
            elif abs(value) >= 1_000:
                value = value / 1_000
                suffix = f"K{suffix}"
        
        # Use Decimal for precise rounding
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal('0.' + '0' * decimals), 
            rounding=ROUND_HALF_UP
        )
        
        formatted = f"{rounded:.{decimals}f}{suffix}"
        
        # Remove trailing zeros for cleaner display
        if '.' in formatted and not percentage:
            formatted = formatted.rstrip('0').rstrip('.')
        
        return formatted
        
    except Exception:
        return str(value)


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0.05 = 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return format_number(value, decimals, percentage=True)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


# =================== DATE/TIME UTILITIES ===================

def format_timestamp(
    dt: Optional[Union[datetime, date, str]], 
    format_string: str = LOG_TIMESTAMP_FORMAT,
    timezone_aware: bool = False
) -> str:
    """
    Format datetime for display.
    
    Args:
        dt: Datetime to format
        format_string: Format string
        timezone_aware: Whether to include timezone info
        
    Returns:
        Formatted datetime string
    """
    if not dt:
        return "N/A"
    
    try:
        if isinstance(dt, str):
            # Try to parse string datetime
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    dt = datetime.strptime(dt, fmt)
                    break
                except ValueError:
                    continue
            else:
                return dt  # Return original string if parsing fails
        
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())
        
        if timezone_aware and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt.strftime(format_string)
        
    except Exception:
        return str(dt)


def parse_date_string(date_str: str) -> Optional[date]:
    """
    Parse date string in various formats.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed date or None
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Common date formats
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    return None


def days_between(start: Union[date, datetime], end: Union[date, datetime]) -> int:
    """
    Calculate days between two dates.
    
    Args:
        start: Start date
        end: End date
        
    Returns:
        Number of days
    """
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    
    return (end - start).days


def is_recent(dt: Union[datetime, date], days: int = 1) -> bool:
    """
    Check if datetime is within recent days.
    
    Args:
        dt: Datetime to check
        days: Number of recent days
        
    Returns:
        True if recent, False otherwise
    """
    if not dt:
        return False
    
    now = datetime.now().date() if isinstance(dt, date) else datetime.now()
    
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())
    if isinstance(now, date) and not isinstance(now, datetime):
        now = datetime.combine(now, datetime.min.time())
    
    return (now - dt).days <= days


# =================== FINANCIAL UTILITIES ===================

def calculate_worst_cvar(
    nig: Optional[float], 
    ghst: Optional[float], 
    evar: Optional[float]
) -> Optional[float]:
    """
    Calculate worst-case CVaR from three methods.
    
    Args:
        nig: NIG-GARCH CVaR
        ghst: GHST-GARCH CVaR  
        evar: Empirical CVaR
        
    Returns:
        Worst (maximum) CVaR value
    """
    values = []
    
    for val in [nig, ghst, evar]:
        if val is not None:
            try:
                float_val = float(val)
                if float_val == float_val:  # Check for NaN
                    values.append(float_val)
            except (ValueError, TypeError):
                continue
    
    return max(values) if values else None


def get_eodhd_suffix(exchange: str = None, country: str = None) -> str:
    """
    Get EODHD API suffix for exchange/country.
    
    Args:
        exchange: Exchange code
        country: Country code
        
    Returns:
        EODHD suffix string
    """
    if country and country in EODHD_SUFFIXES:
        return EODHD_SUFFIXES[country]
    
    # Fallback based on common exchange patterns
    if exchange:
        exchange = exchange.upper()
        if exchange in ['NYSE', 'NASDAQ', 'AMEX']:
            return '.US'
        elif exchange in ['TSX', 'TSXV']:
            return '.TO'
        elif exchange == 'LSE':
            return '.LSE'
        elif exchange in ['XETRA', 'FSE']:
            return '.DE'
        elif exchange == 'EPA':
            return '.PA'
        elif exchange == 'TYO':
            return '.T'
        elif exchange == 'ASX':
            return '.AX'
    
    return '.US'  # Default to US


def normalize_instrument_type(instrument_type: str) -> Optional[str]:
    """
    Normalize instrument type to canonical form.
    
    Args:
        instrument_type: Raw instrument type
        
    Returns:
        Canonical instrument type or None
    """
    if not instrument_type:
        return None
    
    # Check direct match first
    if instrument_type in CANONICAL_INSTRUMENT_TYPES.values():
        return instrument_type
    
    # Try aliases
    key = instrument_type.lower().strip()
    return INSTRUMENT_TYPE_ALIASES.get(key, instrument_type)


def should_include_instrument_type(instrument_type: str) -> bool:
    """
    Check if instrument type should be included in analysis.
    
    Args:
        instrument_type: Instrument type to check
        
    Returns:
        True if should include, False otherwise
    """
    if not instrument_type:
        return False
    
    # Common exclusions
    excluded_patterns = [
        'warrant', 'right', 'unit', 'depositary receipt',
        'preferred', 'convertible', 'structured product'
    ]
    
    instrument_lower = instrument_type.lower()
    return not any(pattern in instrument_lower for pattern in excluded_patterns)


# =================== DATA STRUCTURE UTILITIES ===================

def deep_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested dictionary value using dot notation.
    
    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "data.cvar.nig")
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    try:
        keys = path.split('.')
        result = data
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
    except Exception:
        return default


def flatten_dict(data: Dict[str, Any], prefix: str = "", separator: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        data: Dictionary to flatten
        prefix: Prefix for keys
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    flattened = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, separator))
        else:
            flattened[new_key] = value
    
    return flattened


def group_by(items: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Group list of dictionaries by key.
    
    Args:
        items: List of dictionaries
        key: Key to group by
        
    Returns:
        Dictionary with grouped items
    """
    groups = {}
    for item in items:
        group_key = item.get(key)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)
    return groups


def deduplicate_list(items: List[Any], key_func: Callable = None) -> List[Any]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        items: List of items
        key_func: Function to extract comparison key
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    
    for item in items:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result


# =================== CACHING UTILITIES ===================

def simple_cache(maxsize: int = 128):
    """
    Simple LRU cache decorator.
    
    Args:
        maxsize: Maximum cache size
        
    Returns:
        Decorator function
    """
    return functools.lru_cache(maxsize=maxsize)


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_string = "|".join(key_parts)
    
    # Use hash for very long keys
    if len(key_string) > 250:
        return hashlib.md5(key_string.encode()).hexdigest()
    
    return key_string


# =================== FILE AND PATH UTILITIES ===================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_filename(filename: str, replacement: str = "_") -> str:
    """
    Make filename safe for filesystem.
    
    Args:
        filename: Original filename
        replacement: Replacement for invalid characters
        
    Returns:
        Safe filename
    """
    # Remove/replace invalid characters
    safe = re.sub(r'[<>:"/\\|?*]', replacement, filename)
    
    # Remove leading/trailing dots and spaces
    safe = safe.strip('. ')
    
    # Limit length
    if len(safe) > 255:
        name, ext = os.path.splitext(safe)
        safe = name[:255-len(ext)] + ext
    
    return safe or "unnamed"


def get_file_size_mb(path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        path: File path
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = Path(path).stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


# =================== API UTILITIES ===================

def create_success_response(
    data: Any = None, 
    message: str = None,
    **metadata
) -> Dict[str, Any]:
    """
    Create standardized success response.
    
    Args:
        data: Response data
        message: Success message
        metadata: Additional metadata
        
    Returns:
        Success response dictionary
    """
    response = API_SUCCESS_FORMAT.copy()
    response["data"] = data
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"].update(metadata)
    
    return response


def create_error_response(
    error: str, 
    code: str = None,
    **metadata
) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error: Error message
        code: Error code
        metadata: Additional metadata
        
    Returns:
        Error response dictionary
    """
    response = API_ERROR_FORMAT.copy()
    response["error"] = error
    response["code"] = code or "error"
    
    if metadata:
        response["metadata"].update(metadata)
    
    return response


def paginate_results(
    items: List[Any],
    page: int = 1,
    per_page: int = 20,
    total_count: int = None
) -> Dict[str, Any]:
    """
    Paginate results.
    
    Args:
        items: Items to paginate
        page: Current page (1-based)
        per_page: Items per page
        total_count: Total count (if known)
        
    Returns:
        Paginated response
    """
    if total_count is None:
        total_count = len(items)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_items = items[start_idx:end_idx]
    
    total_pages = (total_count + per_page - 1) // per_page
    
    return {
        "items": page_items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_count": total_count,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }


# =================== LOGGING UTILITIES ===================

def setup_logger(
    name: str,
    level: str = "INFO",
    format_string: str = None,
    file_path: str = None
) -> logging.Logger:
    """
    Setup logger with standard configuration.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        file_path: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


# =================== PERFORMANCE UTILITIES ===================

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation", logger: logging.Logger = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type:
            self.logger.error(f"{self.name} failed after {duration:.3f}s")
        else:
            self.logger.info(f"{self.name} completed in {duration:.3f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 1.0,
    exceptions: Tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        backoff_factor: Backoff multiplier
        exceptions: Exceptions to retry on
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise e
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
            
            return func(*args, **kwargs)  # Final attempt
        return wrapper
    return decorator


# =================== EXPORT ALL UTILITIES ===================

__all__ = [
    # String utilities
    "safe_string",
    "normalize_string", 
    "truncate_string",
    "clean_symbol",
    "pluralize",
    
    # Numeric utilities
    "safe_float",
    "safe_int",
    "format_number",
    "format_percentage", 
    "clamp",
    
    # Date/time utilities
    "format_timestamp",
    "parse_date_string",
    "days_between",
    "is_recent",
    
    # Financial utilities
    "calculate_worst_cvar",
    "get_eodhd_suffix",
    "normalize_instrument_type",
    "should_include_instrument_type",
    
    # Data structure utilities
    "deep_get",
    "flatten_dict",
    "group_by",
    "deduplicate_list",
    
    # Caching utilities
    "simple_cache",
    "cache_key",
    
    # File utilities
    "ensure_dir",
    "safe_filename",
    "get_file_size_mb",
    
    # API utilities
    "create_success_response",
    "create_error_response",
    "paginate_results",
    
    # Logging utilities
    "setup_logger",
    "log_execution_time",
    
    # Performance utilities
    "Timer",
    "retry_with_backoff"
]
