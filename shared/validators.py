"""
Reusable validators and validation utilities.

This module provides common validation functions used throughout the application
to ensure data consistency and eliminate validation logic duplication.
"""

import re
import math
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

from shared.constants import (
    CVAR_ALPHA_LEVELS,
    CANONICAL_INSTRUMENT_TYPES,
    EXCLUDED_INSTRUMENT_TYPES,
    DEFAULT_COUNTRIES,
    ZERO_RETURN_EPSILON,
    MIN_HISTORY_DAYS
)
from shared.exceptions import (
    DataValidationError,
    InvalidSymbolError,
    BusinessLogicError
)


# =================== SYMBOL VALIDATION ===================

def validate_symbol(symbol: Any) -> str:
    """
    Validate and normalize symbol format.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        Normalized symbol string
        
    Raises:
        InvalidSymbolError: If symbol is invalid
    """
    if not symbol:
        raise InvalidSymbolError("", "Symbol cannot be None or empty")
    
    if not isinstance(symbol, str):
        raise InvalidSymbolError(str(symbol), "Symbol must be a string")
    
    symbol = symbol.strip().upper()
    
    if not symbol:
        raise InvalidSymbolError("", "Symbol cannot be empty after trimming")
    
    if len(symbol) > 32:
        raise InvalidSymbolError(symbol, "Symbol too long (max 32 characters)")
    
    # Check for basic format - alphanumeric plus common separators
    if not re.match(r'^[A-Z0-9._-]+$', symbol):
        raise InvalidSymbolError(symbol, "Symbol contains invalid characters")
    
    # Check for reasonable patterns
    if symbol.startswith('.') or symbol.endswith('.'):
        raise InvalidSymbolError(symbol, "Symbol cannot start or end with '.'")
    
    if '..' in symbol or '--' in symbol:
        raise InvalidSymbolError(symbol, "Symbol cannot contain consecutive separators")
    
    return symbol


def validate_symbol_list(symbols: Any, max_length: int = 1000) -> List[str]:
    """
    Validate and normalize a list of symbols.
    
    Args:
        symbols: Symbols to validate (list, comma-separated string, or single symbol)
        max_length: Maximum number of symbols allowed
        
    Returns:
        List of normalized symbols
        
    Raises:
        DataValidationError: If validation fails
    """
    if not symbols:
        return []
    
    # Handle different input formats
    if isinstance(symbols, str):
        if ',' in symbols:
            symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        else:
            symbol_list = [symbols.strip()]
    elif isinstance(symbols, list):
        symbol_list = [str(s) for s in symbols if s]
    else:
        raise DataValidationError("Symbols must be a string or list")
    
    if len(symbol_list) > max_length:
        raise DataValidationError(
            f"Too many symbols: {len(symbol_list)} > {max_length} maximum"
        )
    
    # Validate each symbol
    validated_symbols = []
    for symbol in symbol_list:
        try:
            validated_symbols.append(validate_symbol(symbol))
        except InvalidSymbolError as e:
            raise DataValidationError(f"Invalid symbol in list: {e.message}")
    
    # Remove duplicates while preserving order
    unique_symbols = []
    seen = set()
    for symbol in validated_symbols:
        if symbol not in seen:
            unique_symbols.append(symbol)
            seen.add(symbol)
    
    return unique_symbols


def is_valid_symbol_format(symbol: str) -> bool:
    """
    Check if symbol has valid format without raising exceptions.
    
    Args:
        symbol: Symbol to check
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        validate_symbol(symbol)
        return True
    except InvalidSymbolError:
        return False


# =================== FINANCIAL DATA VALIDATION ===================

def validate_alpha_level(alpha: Any) -> int:
    """
    Validate CVaR alpha level.
    
    Args:
        alpha: Alpha level to validate
        
    Returns:
        Validated alpha level
        
    Raises:
        BusinessLogicError: If alpha level is invalid
    """
    try:
        alpha_int = int(alpha)
    except (ValueError, TypeError):
        raise BusinessLogicError(
            f"Alpha level must be an integer, got {type(alpha).__name__}: {alpha}"
        )
    
    if alpha_int not in CVAR_ALPHA_LEVELS:
        raise BusinessLogicError(
            f"Invalid alpha level: {alpha_int}. Must be one of {CVAR_ALPHA_LEVELS}"
        )
    
    return alpha_int


def validate_price_value(price: Any, field_name: str = "price") -> Optional[float]:
    """
    Validate price value.
    
    Args:
        price: Price value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Validated price or None if invalid
        
    Raises:
        DataValidationError: If price is invalid
    """
    if price is None:
        return None
    
    try:
        if isinstance(price, str):
            price = price.strip()
            if not price or price.lower() in ('n/a', 'null', 'none', ''):
                return None
        
        price_float = float(price)
        
        if math.isnan(price_float):
            return None
        
        if math.isinf(price_float):
            raise DataValidationError(f"{field_name} cannot be infinite: {price}")
        
        if price_float < 0:
            raise DataValidationError(f"{field_name} cannot be negative: {price_float}")
        
        # Check for reasonable bounds (prices should be > 0 and < $1M)
        if price_float > 0 and (price_float < 0.0001 or price_float > 1_000_000):
            raise DataValidationError(
                f"{field_name} outside reasonable range: {price_float}"
            )
        
        return price_float
        
    except (ValueError, TypeError, OverflowError) as e:
        raise DataValidationError(f"Invalid {field_name} format: {price} ({str(e)})")


def validate_return_value(return_val: Any) -> Optional[float]:
    """
    Validate return value (can be negative).
    
    Args:
        return_val: Return value to validate
        
    Returns:
        Validated return or None if invalid
    """
    if return_val is None:
        return None
    
    try:
        return_float = float(return_val)
        
        if math.isnan(return_float):
            return None
        
        if math.isinf(return_float):
            raise DataValidationError(f"Return cannot be infinite: {return_val}")
        
        # Returns should be reasonable (-100% to +1000%)
        if return_float < -1.0 or return_float > 10.0:
            raise DataValidationError(
                f"Return outside reasonable range: {return_float:.4f}"
            )
        
        return return_float
        
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"Invalid return format: {return_val} ({str(e)})")


def validate_returns_series(returns: List[Any], min_length: int = 2) -> List[float]:
    """
    Validate series of returns.
    
    Args:
        returns: List of return values
        min_length: Minimum required length
        
    Returns:
        List of validated returns
        
    Raises:
        DataValidationError: If returns series is invalid
    """
    if not returns:
        raise DataValidationError("Returns series cannot be empty")
    
    if not isinstance(returns, (list, tuple)):
        raise DataValidationError("Returns must be a list or tuple")
    
    validated_returns = []
    for i, ret in enumerate(returns):
        try:
            validated_ret = validate_return_value(ret)
            if validated_ret is not None:
                validated_returns.append(validated_ret)
        except DataValidationError as e:
            raise DataValidationError(f"Invalid return at index {i}: {e.message}")
    
    if len(validated_returns) < min_length:
        raise DataValidationError(
            f"Insufficient returns after validation: {len(validated_returns)} < {min_length}"
        )
    
    # Check for excessive zero returns
    zero_count = sum(1 for r in validated_returns if abs(r) < ZERO_RETURN_EPSILON)
    zero_ratio = zero_count / len(validated_returns)
    
    if zero_ratio > 0.5:  # More than 50% zero returns
        raise DataValidationError(
            f"Too many zero returns: {zero_count}/{len(validated_returns)} ({zero_ratio:.1%})"
        )
    
    return validated_returns


def validate_years_of_data(years: Any, min_years: float = None) -> float:
    """
    Validate years of historical data.
    
    Args:
        years: Years value to validate
        min_years: Minimum years required
        
    Returns:
        Validated years
        
    Raises:
        DataValidationError: If years value is invalid
    """
    try:
        years_float = float(years)
    except (ValueError, TypeError):
        raise DataValidationError(f"Years must be numeric, got: {years}")
    
    if math.isnan(years_float) or math.isinf(years_float):
        raise DataValidationError(f"Years cannot be NaN or infinite: {years_float}")
    
    if years_float <= 0:
        raise DataValidationError(f"Years must be positive: {years_float}")
    
    if years_float > 100:  # Sanity check
        raise DataValidationError(f"Years seems unreasonable: {years_float}")
    
    if min_years and years_float < min_years:
        raise DataValidationError(
            f"Insufficient years of data: {years_float:.2f} < {min_years:.2f} required"
        )
    
    return years_float


# =================== MARKET DATA VALIDATION ===================

def validate_country_code(country: Any) -> Optional[str]:
    """
    Validate country code.
    
    Args:
        country: Country code to validate
        
    Returns:
        Validated country code or None
        
    Raises:
        DataValidationError: If country code is invalid
    """
    if not country:
        return None
    
    if not isinstance(country, str):
        raise DataValidationError("Country must be a string")
    
    country = country.strip().upper()
    
    if len(country) != 2:
        raise DataValidationError(f"Country code must be 2 characters: {country}")
    
    if not country.isalpha():
        raise DataValidationError(f"Country code must be alphabetic: {country}")
    
    # Check against known countries
    if country not in DEFAULT_COUNTRIES:
        raise DataValidationError(f"Unsupported country code: {country}")
    
    return country


def validate_instrument_type(instrument_type: Any) -> Optional[str]:
    """
    Validate and normalize instrument type.
    
    Args:
        instrument_type: Instrument type to validate
        
    Returns:
        Normalized instrument type or None
    """
    if not instrument_type:
        return None
    
    if not isinstance(instrument_type, str):
        return None
    
    instrument_type = instrument_type.strip()
    
    # Check against canonical types
    if instrument_type in CANONICAL_INSTRUMENT_TYPES.values():
        return instrument_type
    
    # Check against excluded types
    if instrument_type in EXCLUDED_INSTRUMENT_TYPES:
        return None
    
    # Try to normalize common variations
    instrument_lower = instrument_type.lower()
    
    type_mappings = {
        "fund": "Mutual Fund",
        "mutual fund": "Mutual Fund",
        "etf": "ETF",
        "stock": "Common Stock",
        "common stock": "Common Stock",
        "equity": "Common Stock",
        "bond": "Bond",
        "reit": "REIT",
        "real estate investment trust": "REIT",
        "index": "Index",
        "commodity": "Commodity",
        "currency": "Currency",
        "crypto": "Cryptocurrency",
        "cryptocurrency": "Cryptocurrency"
    }
    
    return type_mappings.get(instrument_lower, instrument_type)


def validate_exchange_code(exchange: Any) -> Optional[str]:
    """
    Validate exchange code.
    
    Args:
        exchange: Exchange code to validate
        
    Returns:
        Validated exchange code or None
    """
    if not exchange:
        return None
    
    if not isinstance(exchange, str):
        return None
    
    exchange = exchange.strip().upper()
    
    if not exchange:
        return None
    
    # Basic format validation
    if len(exchange) > 10:  # Reasonable length limit
        return None
    
    if not re.match(r'^[A-Z0-9._-]+$', exchange):
        return None
    
    return exchange


# =================== DATE AND TIME VALIDATION ===================

def validate_date(date_value: Any) -> Optional[date]:
    """
    Validate and parse date value.
    
    Args:
        date_value: Date to validate (string, date, or datetime)
        
    Returns:
        Validated date object or None
        
    Raises:
        DataValidationError: If date is invalid
    """
    if not date_value:
        return None
    
    if isinstance(date_value, date):
        return date_value
    
    if isinstance(date_value, datetime):
        return date_value.date()
    
    if isinstance(date_value, str):
        date_value = date_value.strip()
        if not date_value:
            return None
        
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
            try:
                parsed = datetime.strptime(date_value, fmt)
                return parsed.date()
            except ValueError:
                continue
        
        # Try ISO format parsing
        try:
            parsed = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            return parsed.date()
        except ValueError:
            pass
    
    raise DataValidationError(f"Invalid date format: {date_value}")


def validate_date_range(
    start_date: Any, 
    end_date: Any,
    min_days: int = None,
    max_days: int = None
) -> Tuple[Optional[date], Optional[date]]:
    """
    Validate date range.
    
    Args:
        start_date: Start date
        end_date: End date
        min_days: Minimum days in range
        max_days: Maximum days in range
        
    Returns:
        Tuple of validated start and end dates
        
    Raises:
        DataValidationError: If date range is invalid
    """
    start = validate_date(start_date) if start_date else None
    end = validate_date(end_date) if end_date else None
    
    if start and end:
        if start > end:
            raise DataValidationError(
                f"Start date {start} cannot be after end date {end}"
            )
        
        days_diff = (end - start).days
        
        if min_days and days_diff < min_days:
            raise DataValidationError(
                f"Date range too short: {days_diff} days < {min_days} required"
            )
        
        if max_days and days_diff > max_days:
            raise DataValidationError(
                f"Date range too long: {days_diff} days > {max_days} maximum"
            )
    
    return start, end


# =================== NUMERIC VALIDATION ===================

def validate_positive_number(
    value: Any, 
    field_name: str = "value",
    min_value: float = None,
    max_value: float = None
) -> float:
    """
    Validate positive numeric value.
    
    Args:
        value: Value to validate
        field_name: Field name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated numeric value
        
    Raises:
        DataValidationError: If value is invalid
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise DataValidationError(f"{field_name} must be numeric, got: {value}")
    
    if math.isnan(num_value) or math.isinf(num_value):
        raise DataValidationError(f"{field_name} cannot be NaN or infinite")
    
    if num_value <= 0:
        raise DataValidationError(f"{field_name} must be positive: {num_value}")
    
    if min_value and num_value < min_value:
        raise DataValidationError(
            f"{field_name} too small: {num_value} < {min_value}"
        )
    
    if max_value and num_value > max_value:
        raise DataValidationError(
            f"{field_name} too large: {num_value} > {max_value}"
        )
    
    return num_value


def validate_percentage(
    value: Any, 
    field_name: str = "percentage",
    as_decimal: bool = False
) -> float:
    """
    Validate percentage value.
    
    Args:
        value: Percentage to validate
        field_name: Field name for error messages
        as_decimal: If True, expect decimal form (0.0-1.0), otherwise 0-100
        
    Returns:
        Validated percentage
        
    Raises:
        DataValidationError: If percentage is invalid
    """
    try:
        pct_value = float(value)
    except (ValueError, TypeError):
        raise DataValidationError(f"{field_name} must be numeric, got: {value}")
    
    if math.isnan(pct_value) or math.isinf(pct_value):
        raise DataValidationError(f"{field_name} cannot be NaN or infinite")
    
    if as_decimal:
        if pct_value < 0.0 or pct_value > 1.0:
            raise DataValidationError(
                f"{field_name} must be between 0.0 and 1.0: {pct_value}"
            )
    else:
        if pct_value < 0.0 or pct_value > 100.0:
            raise DataValidationError(
                f"{field_name} must be between 0 and 100: {pct_value}"
            )
    
    return pct_value


def validate_integer_range(
    value: Any,
    field_name: str = "value",
    min_value: int = None,
    max_value: int = None
) -> int:
    """
    Validate integer within specified range.
    
    Args:
        value: Value to validate
        field_name: Field name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated integer
        
    Raises:
        DataValidationError: If value is invalid
    """
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise DataValidationError(f"{field_name} must be an integer, got: {value}")
    
    if min_value is not None and int_value < min_value:
        raise DataValidationError(
            f"{field_name} too small: {int_value} < {min_value}"
        )
    
    if max_value is not None and int_value > max_value:
        raise DataValidationError(
            f"{field_name} too large: {int_value} > {max_value}"
        )
    
    return int_value


# =================== COLLECTION VALIDATION ===================

def validate_non_empty_list(
    value: Any, 
    field_name: str = "list",
    max_length: int = None
) -> List[Any]:
    """
    Validate non-empty list.
    
    Args:
        value: Value to validate
        field_name: Field name for error messages
        max_length: Maximum allowed length
        
    Returns:
        Validated list
        
    Raises:
        DataValidationError: If value is invalid
    """
    if not isinstance(value, (list, tuple)):
        raise DataValidationError(f"{field_name} must be a list, got: {type(value)}")
    
    if len(value) == 0:
        raise DataValidationError(f"{field_name} cannot be empty")
    
    if max_length and len(value) > max_length:
        raise DataValidationError(
            f"{field_name} too long: {len(value)} > {max_length}"
        )
    
    return list(value)


# =================== COMPOSITE VALIDATORS ===================

def validate_cvar_request(
    symbol: str,
    alpha: int = None,
    force_recalculate: bool = False,
    to_date: str = None
) -> Dict[str, Any]:
    """
    Validate complete CVaR calculation request.
    
    Args:
        symbol: Symbol to validate
        alpha: Alpha level to validate
        force_recalculate: Force recalculation flag
        to_date: Historical date
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        DataValidationError: If any parameter is invalid
    """
    validated = {
        "symbol": validate_symbol(symbol),
        "force_recalculate": bool(force_recalculate)
    }
    
    if alpha is not None:
        validated["alpha"] = validate_alpha_level(alpha)
    
    if to_date:
        validated["to_date"] = validate_date(to_date)
    
    return validated


# =================== EXPORT ALL VALIDATORS ===================

__all__ = [
    # Symbol validation
    "validate_symbol",
    "validate_symbol_list",
    "is_valid_symbol_format",
    
    # Financial data validation
    "validate_alpha_level",
    "validate_price_value",
    "validate_return_value",
    "validate_returns_series",
    "validate_years_of_data",
    
    # Market data validation
    "validate_country_code",
    "validate_instrument_type",
    "validate_exchange_code",
    
    # Date validation
    "validate_date",
    "validate_date_range",
    
    # Numeric validation
    "validate_positive_number",
    "validate_percentage",
    "validate_integer_range",
    
    # Collection validation
    "validate_non_empty_list",
    
    # Composite validators
    "validate_cvar_request"
]
