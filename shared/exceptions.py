"""
Custom exceptions and error handling utilities.

This module centralizes all custom exceptions used throughout the application
and provides utilities for consistent error handling and reporting.
"""

from typing import Dict, Any, Optional, List
import logging


# =================== BASE EXCEPTIONS ===================

class NirvanaError(Exception):
    """Base exception for all Nirvana application errors."""
    
    def __init__(
        self, 
        message: str, 
        code: str = None, 
        details: Dict[str, Any] = None,
        original_exception: Exception = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__.replace("Error", "").lower()
        self.details = details or {}
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": self.message,
            "code": self.code,
            "type": self.__class__.__name__
        }
        
        if self.details:
            result["details"] = self.details
            
        if self.original_exception:
            result["original_error"] = str(self.original_exception)
            
        return result


# =================== DATA AND VALIDATION EXCEPTIONS ===================

class DataValidationError(NirvanaError):
    """Raised when data validation fails."""
    pass


class InsufficientHistoryError(DataValidationError):
    """Raised when insufficient historical data is available."""
    
    def __init__(
        self, 
        symbol: str, 
        actual_years: float, 
        required_years: float,
        details: Dict[str, Any] = None
    ):
        message = f"Insufficient history for {symbol}: {actual_years:.2f} years < {required_years:.2f} required"
        super().__init__(
            message=message,
            code="insufficient_history",
            details={
                "symbol": symbol,
                "actual_years": actual_years,
                "required_years": required_years,
                **(details or {})
            }
        )
        self.symbol = symbol
        self.actual_years = actual_years
        self.required_years = required_years


class InsufficientDataError(DataValidationError):
    """Raised when insufficient data points remain after cleanup."""
    
    def __init__(
        self, 
        symbol: str, 
        data_points: int, 
        min_required: int = 2,
        details: Dict[str, Any] = None
    ):
        message = f"Insufficient data for {symbol}: {data_points} points < {min_required} required"
        super().__init__(
            message=message,
            code="insufficient_data",
            details={
                "symbol": symbol,
                "data_points": data_points,
                "min_required": min_required,
                **(details or {})
            }
        )
        self.symbol = symbol
        self.data_points = data_points
        self.min_required = min_required


class DataQualityError(DataValidationError):
    """Raised when data quality issues are detected."""
    
    def __init__(
        self, 
        symbol: str, 
        issues: List[str],
        severity: str = "error",
        details: Dict[str, Any] = None
    ):
        message = f"Data quality issues for {symbol}: {', '.join(issues)}"
        super().__init__(
            message=message,
            code="data_quality",
            details={
                "symbol": symbol,
                "issues": issues,
                "severity": severity,
                **(details or {})
            }
        )
        self.symbol = symbol
        self.issues = issues
        self.severity = severity


# =================== CVAR CALCULATION EXCEPTIONS ===================

class CvarCalculationError(NirvanaError):
    """Base class for CVaR calculation errors."""
    pass


class CvarComputationError(CvarCalculationError):
    """Raised when CVaR computation fails."""
    
    def __init__(
        self, 
        symbol: str, 
        method: str = None,
        alpha: int = None,
        original_exception: Exception = None,
        details: Dict[str, Any] = None
    ):
        message = f"CVaR computation failed for {symbol}"
        if method:
            message += f" using {method}"
        if alpha:
            message += f" at {alpha}% level"
            
        super().__init__(
            message=message,
            code="cvar_computation_failed",
            details={
                "symbol": symbol,
                "method": method,
                "alpha": alpha,
                **(details or {})
            },
            original_exception=original_exception
        )
        self.symbol = symbol
        self.method = method
        self.alpha = alpha


class CvarServiceError(CvarCalculationError):
    """Raised when CVaR service encounters an error."""
    
    def __init__(
        self, 
        message: str, 
        service_mode: str = None,
        symbol: str = None,
        details: Dict[str, Any] = None,
        original_exception: Exception = None
    ):
        super().__init__(
            message=message,
            code="cvar_service_error",
            details={
                "service_mode": service_mode,
                "symbol": symbol,
                **(details or {})
            },
            original_exception=original_exception
        )
        self.service_mode = service_mode
        self.symbol = symbol


class RemoteCalculationError(CvarServiceError):
    """Raised when remote CVaR calculation fails."""
    
    def __init__(
        self, 
        symbol: str,
        endpoint_url: str = None, 
        http_status: int = None,
        response_error: str = None,
        details: Dict[str, Any] = None
    ):
        message = f"Remote CVaR calculation failed for {symbol}"
        if http_status:
            message += f" (HTTP {http_status})"
            
        super().__init__(
            message=message,
            service_mode="remote",
            symbol=symbol,
            details={
                "endpoint_url": endpoint_url,
                "http_status": http_status,
                "response_error": response_error,
                **(details or {})
            }
        )
        self.endpoint_url = endpoint_url
        self.http_status = http_status
        self.response_error = response_error


# =================== CONFIGURATION EXCEPTIONS ===================

class ConfigurationError(NirvanaError):
    """Raised when configuration is invalid or missing."""
    pass


class DatabaseConfigurationError(ConfigurationError):
    """Raised when database configuration is invalid."""
    
    def __init__(self, message: str, config_keys: List[str] = None):
        super().__init__(
            message=message,
            code="database_config_error",
            details={"missing_config_keys": config_keys} if config_keys else None
        )
        self.config_keys = config_keys


class ApiConfigurationError(ConfigurationError):
    """Raised when API configuration is invalid."""
    pass


class ServiceConfigurationError(ConfigurationError):
    """Raised when service configuration is invalid."""
    
    def __init__(self, service_name: str, message: str, config: Dict[str, Any] = None):
        super().__init__(
            message=f"{service_name}: {message}",
            code="service_config_error",
            details={
                "service_name": service_name,
                "config": config
            }
        )
        self.service_name = service_name


# =================== EXTERNAL SERVICE EXCEPTIONS ===================

class ExternalServiceError(NirvanaError):
    """Base class for external service errors."""
    pass


class EODHDApiError(ExternalServiceError):
    """Raised when EODHD API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        symbol: str = None,
        endpoint: str = None,
        http_status: int = None,
        response_data: Any = None
    ):
        super().__init__(
            message=message,
            code="eodhd_api_error",
            details={
                "symbol": symbol,
                "endpoint": endpoint,
                "http_status": http_status,
                "response_data": response_data
            }
        )
        self.symbol = symbol
        self.endpoint = endpoint
        self.http_status = http_status
        self.response_data = response_data


class AzureServiceBusError(ExternalServiceError):
    """Raised when Azure Service Bus operations fail."""
    
    def __init__(
        self, 
        message: str,
        operation: str = None, 
        queue_name: str = None,
        correlation_id: str = None,
        original_exception: Exception = None
    ):
        super().__init__(
            message=message,
            code="azure_service_bus_error",
            details={
                "operation": operation,
                "queue_name": queue_name,
                "correlation_id": correlation_id
            },
            original_exception=original_exception
        )
        self.operation = operation
        self.queue_name = queue_name
        self.correlation_id = correlation_id


class OpenAIServiceError(ExternalServiceError):
    """Raised when OpenAI API calls fail."""
    
    def __init__(
        self, 
        message: str,
        model: str = None,
        thread_id: str = None,
        run_id: str = None,
        original_exception: Exception = None
    ):
        super().__init__(
            message=message,
            code="openai_service_error",
            details={
                "model": model,
                "thread_id": thread_id,
                "run_id": run_id
            },
            original_exception=original_exception
        )
        self.model = model
        self.thread_id = thread_id
        self.run_id = run_id


# =================== REPOSITORY EXCEPTIONS ===================

class RepositoryError(NirvanaError):
    """Base class for repository errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when requested entity is not found."""
    
    def __init__(self, entity_type: str, identifier: str, details: Dict[str, Any] = None):
        message = f"{entity_type} not found: {identifier}"
        super().__init__(
            message=message,
            code="entity_not_found",
            details={
                "entity_type": entity_type,
                "identifier": identifier,
                **(details or {})
            }
        )
        self.entity_type = entity_type
        self.identifier = identifier


class RepositoryConnectionError(RepositoryError):
    """Raised when repository cannot connect to data store."""
    
    def __init__(self, repository_name: str, original_exception: Exception = None):
        message = f"Cannot connect to {repository_name}"
        super().__init__(
            message=message,
            code="repository_connection_error",
            details={"repository_name": repository_name},
            original_exception=original_exception
        )
        self.repository_name = repository_name


# =================== BUSINESS LOGIC EXCEPTIONS ===================

class BusinessLogicError(NirvanaError):
    """Base class for business logic violations."""
    pass


class InvalidSymbolError(BusinessLogicError):
    """Raised when symbol is invalid or unsupported."""
    
    def __init__(self, symbol: str, reason: str = None):
        message = f"Invalid symbol: {symbol}"
        if reason:
            message += f" ({reason})"
            
        super().__init__(
            message=message,
            code="invalid_symbol",
            details={"symbol": symbol, "reason": reason}
        )
        self.symbol = symbol
        self.reason = reason


class UnsupportedOperationError(BusinessLogicError):
    """Raised when requested operation is not supported."""
    
    def __init__(self, operation: str, context: str = None):
        message = f"Unsupported operation: {operation}"
        if context:
            message += f" in context: {context}"
            
        super().__init__(
            message=message,
            code="unsupported_operation",
            details={"operation": operation, "context": context}
        )
        self.operation = operation
        self.context = context


class RateLimitExceededError(BusinessLogicError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self, 
        resource: str, 
        limit: int, 
        period: str,
        retry_after: int = None
    ):
        message = f"Rate limit exceeded for {resource}: {limit} requests per {period}"
        super().__init__(
            message=message,
            code="rate_limit_exceeded",
            details={
                "resource": resource,
                "limit": limit,
                "period": period,
                "retry_after": retry_after
            }
        )
        self.resource = resource
        self.limit = limit
        self.period = period
        self.retry_after = retry_after


# =================== BATCH PROCESSING EXCEPTIONS ===================

class BatchProcessingError(NirvanaError):
    """Base class for batch processing errors."""
    pass


class BatchJobError(BatchProcessingError):
    """Raised when batch job fails."""
    
    def __init__(
        self, 
        job_id: str, 
        job_type: str,
        processed_count: int, 
        total_count: int,
        errors: List[str] = None
    ):
        message = f"Batch job {job_id} failed: {processed_count}/{total_count} processed"
        super().__init__(
            message=message,
            code="batch_job_failed",
            details={
                "job_id": job_id,
                "job_type": job_type,
                "processed_count": processed_count,
                "total_count": total_count,
                "errors": errors or []
            }
        )
        self.job_id = job_id
        self.job_type = job_type
        self.processed_count = processed_count
        self.total_count = total_count
        self.errors = errors or []


# =================== ERROR HANDLING UTILITIES ===================

def handle_exception(
    exception: Exception, 
    logger: logging.Logger,
    context: Dict[str, Any] = None,
    reraise: bool = False
) -> Dict[str, Any]:
    """
    Centralized exception handling utility.
    
    Args:
        exception: The exception to handle
        logger: Logger instance for error reporting
        context: Additional context information
        reraise: Whether to re-raise the exception
        
    Returns:
        Dictionary representation of the error
    """
    error_dict = {}
    
    if isinstance(exception, NirvanaError):
        error_dict = exception.to_dict()
        logger.error(f"NirvanaError: {exception.message}", extra={
            "error_code": exception.code,
            "details": exception.details,
            "context": context
        })
    else:
        error_dict = {
            "error": str(exception),
            "code": "unexpected_error",
            "type": exception.__class__.__name__
        }
        logger.error(f"Unexpected error: {str(exception)}", extra={
            "exception_type": exception.__class__.__name__,
            "context": context
        }, exc_info=True)
    
    if context:
        error_dict["context"] = context
    
    if reraise:
        raise exception
        
    return error_dict


def create_error_response(
    exception: Exception,
    include_details: bool = True
) -> Dict[str, Any]:
    """
    Create standardized error response from exception.
    
    Args:
        exception: The exception to convert
        include_details: Whether to include detailed error information
        
    Returns:
        Standardized error response dictionary
    """
    if isinstance(exception, NirvanaError):
        response = {
            "success": False,
            "error": exception.message,
            "code": exception.code
        }
        
        if include_details and exception.details:
            response["details"] = exception.details
            
        return response
    else:
        return {
            "success": False,
            "error": str(exception),
            "code": "unexpected_error"
        }


def validate_symbol(symbol: str) -> None:
    """
    Validate symbol format and raise appropriate exception.
    
    Args:
        symbol: Symbol to validate
        
    Raises:
        InvalidSymbolError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise InvalidSymbolError(symbol or "None", "Symbol must be a non-empty string")
    
    symbol = symbol.strip().upper()
    
    if len(symbol) == 0:
        raise InvalidSymbolError(symbol, "Symbol cannot be empty")
    
    if len(symbol) > 32:
        raise InvalidSymbolError(symbol, "Symbol too long (max 32 characters)")
    
    # Basic format validation
    if not symbol.replace(".", "").replace("-", "").isalnum():
        raise InvalidSymbolError(symbol, "Symbol contains invalid characters")


def validate_alpha_level(alpha: int) -> None:
    """
    Validate alpha level for CVaR calculations.
    
    Args:
        alpha: Alpha level to validate
        
    Raises:
        BusinessLogicError: If alpha level is invalid
    """
    valid_alphas = [50, 95, 99]
    if alpha not in valid_alphas:
        raise BusinessLogicError(
            f"Invalid alpha level: {alpha}. Must be one of {valid_alphas}",
            code="invalid_alpha_level",
            details={"alpha": alpha, "valid_alphas": valid_alphas}
        )


# =================== EXPORT ALL EXCEPTIONS ===================

__all__ = [
    # Base exceptions
    "NirvanaError",
    
    # Data validation exceptions
    "DataValidationError",
    "InsufficientHistoryError",
    "InsufficientDataError", 
    "DataQualityError",
    
    # CVaR calculation exceptions
    "CvarCalculationError",
    "CvarComputationError",
    "CvarServiceError",
    "RemoteCalculationError",
    
    # Configuration exceptions
    "ConfigurationError",
    "DatabaseConfigurationError",
    "ApiConfigurationError",
    "ServiceConfigurationError",
    
    # External service exceptions
    "ExternalServiceError",
    "EODHDApiError",
    "AzureServiceBusError",
    "OpenAIServiceError",
    
    # Repository exceptions
    "RepositoryError",
    "EntityNotFoundError",
    "RepositoryConnectionError",
    
    # Business logic exceptions
    "BusinessLogicError",
    "InvalidSymbolError",
    "UnsupportedOperationError",
    "RateLimitExceededError",
    
    # Batch processing exceptions
    "BatchProcessingError",
    "BatchJobError",
    
    # Utility functions
    "handle_exception",
    "create_error_response",
    "validate_symbol",
    "validate_alpha_level"
]
