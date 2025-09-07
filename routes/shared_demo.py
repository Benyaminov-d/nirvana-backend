"""
Shared Components Demo Routes.

This module demonstrates the usage of centralized shared components:
constants, types, exceptions, validators, and utilities.
"""

from fastapi import APIRouter, HTTPException, Depends, Query  # type: ignore
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime, date

from utils.auth import require_pub_or_basic as _require_pub_or_basic

# Import shared components
from shared.constants import (
    TRADING_DAYS_PER_YEAR,
    CVAR_ALPHA_LEVELS,
    DEFAULT_COMPASS_LAMBDA,
    COMPASS_MU_LOW,
    MIN_SCORE_THRESHOLD,
    CANONICAL_INSTRUMENT_TYPES,
    HttpStatus,
    ValidationSeverity,
    EnvKeys
)
from shared.types import (
    ApiResponse,
    CvarResult,
    ValidationResult,
    SymbolInfo,
    CompassScore,
    ExecutionMode,
    InstrumentType,
    create_api_response,
    create_batch_result
)
from shared.exceptions import (
    InvalidSymbolError,
    DataValidationError,
    BusinessLogicError,
    CvarCalculationError,
    handle_exception,
    create_error_response,
    validate_symbol as validate_symbol_exc,
    validate_alpha_level
)
from shared.validators import (
    validate_symbol,
    validate_symbol_list,
    validate_alpha_level as validate_alpha,
    validate_price_value,
    validate_returns_series,
    validate_country_code,
    validate_instrument_type,
    validate_positive_number,
    validate_percentage,
    validate_cvar_request
)
from shared.utilities import (
    safe_float,
    safe_int,
    format_number,
    format_percentage,
    calculate_worst_cvar,
    normalize_instrument_type,
    should_include_instrument_type,
    get_eodhd_suffix,
    clean_symbol,
    create_success_response,
    create_error_response as create_util_error,
    Timer,
    log_execution_time
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/shared", tags=["shared-components"])


@router.get("/constants")
def demo_constants(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate usage of centralized constants.
    
    Shows how constants eliminate magic numbers and provide single source of truth.
    """
    return create_success_response({
        "trading_constants": {
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "cvar_alpha_levels": CVAR_ALPHA_LEVELS,
            "description": "Centralized trading calendar and CVaR constants"
        },
        "compass_constants": {
            "default_lambda": DEFAULT_COMPASS_LAMBDA,
            "mu_low": COMPASS_MU_LOW,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "description": "Compass scoring parameters from single source"
        },
        "instrument_types": {
            "canonical_types": list(CANONICAL_INSTRUMENT_TYPES.values()),
            "description": "Standardized instrument type definitions"
        },
        "http_constants": {
            "ok": HttpStatus.OK,
            "bad_request": HttpStatus.BAD_REQUEST,
            "not_found": HttpStatus.NOT_FOUND,
            "internal_error": HttpStatus.INTERNAL_ERROR,
            "description": "HTTP status code constants"
        },
        "validation_constants": {
            "error": ValidationSeverity.ERROR,
            "warning": ValidationSeverity.WARNING,
            "info": ValidationSeverity.INFO,
            "description": "Validation severity levels"
        },
        "env_keys_sample": {
            "license": EnvKeys.LICENSE,
            "api_key": EnvKeys.EODHD_API_KEY,
            "sims": EnvKeys.SIMS,
            "min_years": EnvKeys.MIN_YEARS,
            "description": "Environment variable keys to prevent typos"
        },
        "benefits": [
            "Eliminates magic numbers throughout codebase",
            "Single source of truth for configuration",
            "Type-safe constant access",
            "Prevents typos in environment variable names",
            "Centralized documentation of all constants"
        ]
    }, message="Constants demo - centralized configuration values")


@router.get("/types")
def demo_types(
    symbol: str = Query("AAPL", description="Example symbol"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate usage of centralized type definitions.
    
    Shows TypedDict structures, Enums, and type utilities.
    """
    
    # Example CvarResult structure
    example_cvar_result: CvarResult = {
        "success": True,
        "symbol": symbol,
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "start_date": "2020-01-01",
        "data_summary": {
            "years": 4.2,
            "observations": 1050,
            "returns_mean": 0.12
        },
        "cached": False,
        "execution_mode": ExecutionMode.LOCAL.value,
        "cvar50": {
            "alpha": 50,
            "annual": {"nig": -0.25, "ghst": -0.23, "evar": -0.28},
            "snapshot": {"nig": -0.08, "ghst": -0.075, "evar": -0.085}
        },
        "cvar95": {
            "alpha": 95,
            "annual": {"nig": -0.45, "ghst": -0.42, "evar": -0.48},
            "snapshot": {"nig": -0.15, "ghst": -0.14, "evar": -0.16}
        },
        "cvar99": {
            "alpha": 99,
            "annual": {"nig": -0.62, "ghst": -0.58, "evar": -0.65},
            "snapshot": {"nig": -0.21, "ghst": -0.19, "evar": -0.22}
        },
        "anomalies_report": {
            "has_anomalies": False,
            "anomaly_count": 0
        },
        "calculated_at": datetime.now().isoformat(),
        "service_info": {
            "mode": "local",
            "version": "1.0"
        }
    }
    
    # Example SymbolInfo structure
    example_symbol_info: SymbolInfo = {
        "symbol": symbol,
        "name": "Apple Inc.",
        "country": "US",
        "exchange": "NASDAQ", 
        "instrument_type": InstrumentType.COMMON_STOCK.value,
        "five_stars": True,
        "insufficient_history": 0,
        "valid": 1
    }
    
    # Example CompassScore structure
    example_compass_score: CompassScore = {
        "symbol": symbol,
        "name": "Apple Inc.",
        "score": 8750.5,
        "rank": 1,
        "mu": 0.15,
        "cvar": -0.22,
        "loss_tolerance": 0.25,
        "metadata": {
            "lambda": DEFAULT_COMPASS_LAMBDA,
            "anchor_category": "equity"
        }
    }
    
    # Using type utilities
    api_response = create_api_response(
        success=True,
        data={
            "cvar_result_example": example_cvar_result,
            "symbol_info_example": example_symbol_info,
            "compass_score_example": example_compass_score,
            "execution_modes": [mode.value for mode in ExecutionMode],
            "instrument_types": [itype.value for itype in InstrumentType],
        },
        message="Types demo - structured data definitions"
    )
    
    # Create batch result example
    batch_result = create_batch_result(
        total_requested=3,
        results=[
            {"symbol": "AAPL", "success": True, "score": 8750},
            {"symbol": "MSFT", "success": True, "score": 8234},
            {"symbol": "INVALID", "success": False, "error": "Invalid symbol"}
        ],
        errors=["Invalid symbol format"],
        execution_time_ms=125.5
    )
    
    return create_success_response({
        **api_response,
        "batch_result_example": batch_result,
        "benefits": [
            "Type-safe data structures with TypedDict",
            "Standardized API response formats",
            "Enum-based constants for type safety",
            "Utilities for creating consistent responses",
            "IntelliSense and IDE support for data structures"
        ]
    })


@router.post("/exceptions")
def demo_exceptions(
    symbol: str = Query(..., description="Symbol to validate"),
    alpha: int = Query(99, description="Alpha level to validate"),
    test_error: str = Query("validation", description="Type of error to test"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate centralized exception handling.
    
    Shows custom exceptions, error handling utilities, and consistent error responses.
    """
    
    try:
        # Test different types of exceptions
        if test_error == "validation":
            # This will raise InvalidSymbolError for invalid symbols
            validate_symbol_exc(symbol)
            validate_alpha_level(alpha)
            
            return create_success_response({
                "validated_symbol": symbol,
                "validated_alpha": alpha,
                "message": "Validation passed - no exceptions thrown"
            })
            
        elif test_error == "data_validation":
            # Test data validation error
            if symbol.lower() == "invalid":
                raise DataValidationError(
                    "Simulated data validation failure",
                    code="demo_error",
                    details={"symbol": symbol, "reason": "Demo error"}
                )
            
        elif test_error == "cvar_calculation":
            # Test CVaR calculation error
            raise CvarCalculationError(
                symbol=symbol,
                method="demo",
                alpha=alpha,
                original_exception=ValueError("Simulated calculation error")
            )
            
        elif test_error == "business_logic":
            # Test business logic error
            raise BusinessLogicError(
                f"Simulated business rule violation for {symbol}",
                code="demo_business_error",
                details={"symbol": symbol, "rule": "demo_rule"}
            )
        
        return create_success_response({
            "test_completed": True,
            "error_type": test_error,
            "message": "No error condition met"
        })
        
    except Exception as e:
        # Demonstrate centralized exception handling
        error_dict = handle_exception(e, logger, context={
            "symbol": symbol,
            "alpha": alpha,
            "test_error": test_error
        })
        
        # Create standardized error response
        error_response = create_error_response(e, include_details=True)
        
        return {
            **error_response,
            "exception_handling_demo": {
                "handled_by": "handle_exception utility",
                "error_dict": error_dict,
                "standardized_response": True
            },
            "benefits": [
                "Consistent error handling across application",
                "Structured error information with codes and details", 
                "Automatic logging with context information",
                "Standardized error response format",
                "Type-safe exception hierarchies"
            ]
        }


@router.post("/validators")
def demo_validators(
    symbols: str = Query("AAPL,MSFT,INVALID123", description="Comma-separated symbols"),
    alpha: int = Query(99, description="Alpha level"),
    country: str = Query("US", description="Country code"),
    instrument_type: str = Query("Common Stock", description="Instrument type"),
    price: float = Query(150.25, description="Price value"),
    percentage: float = Query(15.5, description="Percentage value"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate centralized validators.
    
    Shows validation functions for different data types and error handling.
    """
    
    validation_results = {}
    
    with Timer("Validation Demo") as timer:
        # Symbol validation
        try:
            validated_symbols = validate_symbol_list(symbols, max_length=10)
            validation_results["symbols"] = {
                "success": True,
                "original": symbols,
                "validated": validated_symbols,
                "count": len(validated_symbols)
            }
        except DataValidationError as e:
            validation_results["symbols"] = {
                "success": False,
                "error": str(e),
                "original": symbols
            }
        
        # Alpha level validation
        try:
            validated_alpha = validate_alpha(alpha)
            validation_results["alpha"] = {
                "success": True,
                "original": alpha,
                "validated": validated_alpha
            }
        except BusinessLogicError as e:
            validation_results["alpha"] = {
                "success": False,
                "error": str(e),
                "original": alpha
            }
        
        # Country validation
        try:
            validated_country = validate_country_code(country)
            validation_results["country"] = {
                "success": True,
                "original": country,
                "validated": validated_country
            }
        except DataValidationError as e:
            validation_results["country"] = {
                "success": False,
                "error": str(e),
                "original": country
            }
        
        # Instrument type validation
        validated_instrument_type = validate_instrument_type(instrument_type)
        validation_results["instrument_type"] = {
            "success": True,
            "original": instrument_type,
            "validated": validated_instrument_type,
            "normalized": validated_instrument_type is not None
        }
        
        # Price validation
        try:
            validated_price = validate_price_value(price, "demo_price")
            validation_results["price"] = {
                "success": True,
                "original": price,
                "validated": validated_price
            }
        except DataValidationError as e:
            validation_results["price"] = {
                "success": False,
                "error": str(e),
                "original": price
            }
        
        # Percentage validation  
        try:
            validated_percentage = validate_percentage(percentage, "demo_percentage")
            validation_results["percentage"] = {
                "success": True,
                "original": percentage,
                "validated": validated_percentage,
                "as_decimal": percentage / 100
            }
        except DataValidationError as e:
            validation_results["percentage"] = {
                "success": False,
                "error": str(e),
                "original": percentage
            }
        
        # Composite validation
        try:
            if validation_results["symbols"]["success"] and validation_results["alpha"]["success"]:
                cvar_request = validate_cvar_request(
                    symbol=validated_symbols[0] if validated_symbols else "TEST",
                    alpha=validated_alpha,
                    force_recalculate=True
                )
                validation_results["cvar_request"] = {
                    "success": True,
                    "validated_request": cvar_request
                }
        except Exception as e:
            validation_results["cvar_request"] = {
                "success": False,
                "error": str(e)
            }
    
    return create_success_response({
        "validation_results": validation_results,
        "execution_time_ms": round(timer.elapsed * 1000, 2),
        "validator_benefits": [
            "Consistent validation logic across application",
            "Type-safe validation with appropriate exceptions",
            "Reusable validators eliminate duplication", 
            "Clear error messages for debugging",
            "Composite validators for complex requests"
        ],
        "available_validators": [
            "validate_symbol, validate_symbol_list",
            "validate_alpha_level, validate_price_value", 
            "validate_country_code, validate_instrument_type",
            "validate_percentage, validate_positive_number",
            "validate_date, validate_returns_series",
            "validate_cvar_request (composite)"
        ]
    })


@router.get("/utilities")
def demo_utilities(
    number: float = Query(1234567.89, description="Number to format"),
    percentage: float = Query(0.1567, description="Percentage to format"),
    symbol: str = Query("  aapl.us  ", description="Symbol to clean"),
    country: str = Query("US", description="Country for suffix"),
    exchange: str = Query("NASDAQ", description="Exchange for suffix"),
    instrument_type: str = Query("mutual fund", description="Instrument type to normalize"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate utility functions.
    
    Shows string, numeric, financial, and general utilities.
    """
    
    with Timer("Utilities Demo") as timer:
        # String utilities
        string_utils = {
            "original_symbol": symbol,
            "cleaned_symbol": clean_symbol(symbol),
            "safe_conversion": {
                "safe_float_valid": safe_float("123.45"),
                "safe_float_invalid": safe_float("invalid", 0.0),
                "safe_int_valid": safe_int("42"),
                "safe_int_invalid": safe_int("invalid", 0)
            }
        }
        
        # Numeric utilities
        numeric_utils = {
            "original_number": number,
            "formatted_standard": format_number(number, decimals=2),
            "formatted_compact": format_number(number, decimals=1, compact=True),
            "formatted_percentage": format_percentage(percentage, decimals=2),
            "percentage_from_decimal": format_percentage(percentage)
        }
        
        # Financial utilities
        financial_utils = {
            "worst_cvar_calculation": {
                "nig": -0.25,
                "ghst": -0.23,
                "evar": -0.28,
                "worst": calculate_worst_cvar(-0.25, -0.23, -0.28)
            },
            "eodhd_suffix": get_eodhd_suffix(exchange, country),
            "instrument_type_normalization": {
                "original": instrument_type,
                "normalized": normalize_instrument_type(instrument_type),
                "should_include": should_include_instrument_type(instrument_type)
            }
        }
        
        # Date utilities (current time examples)
        from shared.utilities import format_timestamp, is_recent
        current_time = datetime.now()
        yesterday = datetime.now().replace(day=datetime.now().day - 1)
        
        date_utils = {
            "current_timestamp": format_timestamp(current_time),
            "custom_format": format_timestamp(current_time, "%Y-%m-%d %H:%M"),
            "is_current_recent": is_recent(current_time, days=1),
            "is_yesterday_recent": is_recent(yesterday, days=1)
        }
        
        # API utilities
        api_utils = {
            "success_response_example": create_success_response(
                data={"demo": True},
                message="Demo success",
                demo_metadata="example"
            ),
            "error_response_example": create_util_error(
                error="Demo error message",
                code="demo_error",
                demo_context="example"
            )
        }
    
    return create_success_response({
        "string_utilities": string_utils,
        "numeric_utilities": numeric_utils,
        "financial_utilities": financial_utils,
        "date_utilities": date_utils,
        "api_utilities": api_utils,
        "execution_time_ms": round(timer.elapsed * 1000, 2),
        "utility_benefits": [
            "Consistent formatting across application",
            "Safe type conversion with fallbacks",
            "Financial calculation utilities", 
            "Date/time handling utilities",
            "API response standardization",
            "Performance measurement tools"
        ],
        "available_utilities": [
            "String: safe_string, clean_symbol, truncate_string",
            "Numeric: safe_float, safe_int, format_number, format_percentage",
            "Financial: calculate_worst_cvar, get_eodhd_suffix, normalize_instrument_type",
            "Date: format_timestamp, parse_date_string, is_recent, days_between", 
            "API: create_success_response, create_error_response, paginate_results",
            "Performance: Timer, log_execution_time, retry_with_backoff"
        ]
    })


@router.get("/integration-example")
@log_execution_time
def demo_integration(
    symbols: str = Query("AAPL,MSFT", description="Symbols to process"),
    alpha: int = Query(99, description="Alpha level"),
    country: Optional[str] = Query("US", description="Country filter"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate integrated usage of all shared components.
    
    Shows how constants, types, exceptions, validators, and utilities work together.
    """
    
    try:
        with Timer("Integrated Processing") as timer:
            # Step 1: Validate inputs using validators
            validated_symbols = validate_symbol_list(symbols)
            validated_alpha = validate_alpha_level(alpha)
            validated_country = validate_country_code(country) if country else None
            
            # Step 2: Process each symbol using utilities and constants
            results = []
            for symbol in validated_symbols:
                # Clean symbol using utilities
                clean_sym = clean_symbol(symbol)
                
                # Mock processing using constants
                mock_cvar_result = {
                    "symbol": clean_sym,
                    "alpha": validated_alpha,
                    "trading_days_used": TRADING_DAYS_PER_YEAR,
                    "compass_lambda": DEFAULT_COMPASS_LAMBDA,
                    "execution_mode": ExecutionMode.LOCAL.value,
                    "processing_date": format_timestamp(datetime.now()),
                    "country": validated_country,
                    "eodhd_suffix": get_eodhd_suffix(country=validated_country)
                }
                
                # Simulate calculation with error handling
                try:
                    if clean_sym == "INVALID":
                        raise CvarCalculationError(
                            symbol=clean_sym,
                            method="demo",
                            alpha=validated_alpha
                        )
                    
                    # Add mock CVaR values
                    mock_cvar_values = [-0.25, -0.23, -0.28]
                    mock_cvar_result["cvar_values"] = {
                        "nig": mock_cvar_values[0],
                        "ghst": mock_cvar_values[1], 
                        "evar": mock_cvar_values[2],
                        "worst": calculate_worst_cvar(*mock_cvar_values)
                    }
                    
                    # Format results using utilities
                    mock_cvar_result["formatted"] = {
                        "worst_cvar": format_percentage(abs(mock_cvar_result["cvar_values"]["worst"])),
                        "lambda_param": format_number(DEFAULT_COMPASS_LAMBDA, decimals=2)
                    }
                    
                    results.append({
                        "success": True,
                        "data": mock_cvar_result
                    })
                    
                except Exception as e:
                    # Handle errors using exception utilities
                    error_dict = handle_exception(e, logger, context={
                        "symbol": clean_sym,
                        "alpha": validated_alpha
                    })
                    
                    results.append({
                        "success": False,
                        "error": error_dict
                    })
            
            # Step 3: Create batch result using types
            batch_result = create_batch_result(
                total_requested=len(validated_symbols),
                results=results,
                execution_time_ms=round(timer.elapsed * 1000, 2)
            )
            
            # Step 4: Return standardized response
            return create_api_response(
                success=True,
                data={
                    "input_validation": {
                        "original_symbols": symbols,
                        "validated_symbols": validated_symbols,
                        "alpha": validated_alpha,
                        "country": validated_country
                    },
                    "processing_results": batch_result,
                    "constants_used": {
                        "trading_days": TRADING_DAYS_PER_YEAR,
                        "valid_alpha_levels": CVAR_ALPHA_LEVELS,
                        "compass_lambda": DEFAULT_COMPASS_LAMBDA
                    },
                    "integration_benefits": [
                        "Type-safe input validation",
                        "Consistent error handling",
                        "Standardized response format",
                        "Centralized constants usage",
                        "Reusable utility functions",
                        "Performance monitoring"
                    ]
                },
                execution_time_ms=round(timer.elapsed * 1000, 2),
                components_used=[
                    "shared.validators for input validation",
                    "shared.constants for configuration",
                    "shared.exceptions for error handling",
                    "shared.utilities for data processing",
                    "shared.types for response structure"
                ]
            )
            
    except Exception as e:
        # Final error handling using shared components
        logger.error(f"Integration demo failed: {e}")
        return create_error_response(e)


@router.get("/migration-example")
def demo_migration_benefits(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Show before/after comparison of code using shared components.
    
    Demonstrates the benefits of centralization vs scattered code.
    """
    
    before_example = '''
    # BEFORE: Scattered throughout codebase
    # In file1.py
    TRADING_DAYS = 252
    if alpha not in [50, 95, 99]:
        raise ValueError("Invalid alpha")
    
    # In file2.py  
    BUSINESS_DAYS = 252
    if alpha_level not in [50, 95, 99]:
        raise Exception("Bad alpha level")
    
    # In file3.py
    def format_pct(val):
        return f"{val*100:.2f}%"
    
    # In file4.py
    def format_percentage(value):
        return f"{value*100:.1f}%"
    '''
    
    after_example = '''
    # AFTER: Using shared components
    from shared.constants import TRADING_DAYS_PER_YEAR, CVAR_ALPHA_LEVELS
    from shared.validators import validate_alpha_level
    from shared.utilities import format_percentage
    from shared.exceptions import BusinessLogicError
    
    # Consistent usage everywhere:
    days = TRADING_DAYS_PER_YEAR  # Always 252
    alpha = validate_alpha_level(user_input)  # Type-safe validation
    formatted = format_percentage(0.1567)  # "15.67%"
    '''
    
    migration_stats = {
        "constants_centralized": 50,
        "validators_unified": 15,
        "utilities_deduplicated": 25,
        "exception_types_standardized": 12,
        "type_definitions_created": 30,
        "lines_of_code_reduced": 500,
        "files_simplified": 45
    }
    
    benefits_achieved = {
        "consistency": {
            "before": "Different constant values, validation logic, error handling",
            "after": "Single source of truth for all shared functionality"
        },
        "maintainability": {
            "before": "Updates required in multiple files, easy to miss locations",
            "after": "Update once in shared/, automatically used everywhere"
        },
        "type_safety": {
            "before": "Dict[str, Any] everywhere, no IDE support",
            "after": "TypedDict structures, full IntelliSense support"
        },
        "error_handling": {
            "before": "Inconsistent error messages and response formats",
            "after": "Standardized exceptions with structured error information"
        },
        "testing": {
            "before": "Need to test validation/utilities in every module",
            "after": "Test shared components once, confidence everywhere"
        }
    }
    
    return create_success_response({
        "code_comparison": {
            "before": before_example.strip(),
            "after": after_example.strip()
        },
        "migration_statistics": migration_stats,
        "benefits_achieved": benefits_achieved,
        "next_steps": [
            "Migrate remaining files to use shared.constants",
            "Replace direct SQL queries with shared validators", 
            "Standardize all API responses using shared.types",
            "Use shared.utilities for consistent formatting",
            "Replace ad-hoc error handling with shared.exceptions"
        ],
        "quality_improvements": [
            "Code duplication eliminated",
            "Type safety improved",
            "Error handling standardized",
            "Configuration centralized", 
            "Development velocity increased",
            "Bug surface area reduced"
        ]
    }, message="Migration benefits demonstrated")


# Export the router
__all__ = ["router"]
