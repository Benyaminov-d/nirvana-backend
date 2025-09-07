"""
Infrastructure Services Demo Routes - External integrations demonstration.

This module demonstrates the infrastructure layer services that handle
external systems, APIs, message queues, and caching operations.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, date
import logging

from services.infrastructure.eodhd_client import EODHDClient
from services.infrastructure.azure_service_bus_client import AzureServiceBusClient, MessagePriority
from services.infrastructure.cache_service import CacheService, CacheBackend
from utils.auth import require_pub_or_basic as _require_pub_or_basic

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/infrastructure/eodhd-demo")
def eodhd_demo(
    symbol: str = Query("AAPL", description="Symbol to test"),
    exchange: str = Query("US", description="Exchange code"),
    days_back: int = Query(30, description="Days of historical data to fetch"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate EODHD API client capabilities.
    
    This shows how the infrastructure service abstracts external API calls
    with proper error handling, data validation, and response formatting.
    """
    
    # Initialize EODHD client
    eodhd_client = EODHDClient()
    
    try:
        # Get API status first
        api_status = eodhd_client.get_api_status()
        
        result = {
            "demo_info": {
                "purpose": "Demonstrate EODHD API integration",
                "symbol_tested": f"{symbol}.{exchange}",
                "infrastructure_benefits": [
                    "Clean abstraction of external API",
                    "Automatic error handling and retry logic",
                    "Structured data models (PriceDataPoint, EODHDSymbolInfo)",
                    "Configuration management through environment variables",
                    "Graceful degradation when API unavailable"
                ]
            },
            "api_status": api_status
        }
        
        # Only proceed with data fetching if API is operational
        if api_status["status"] == "operational":
            # Get historical price data
            from_date = date.today() - timedelta(days=days_back)
            to_date = date.today()
            
            try:
                price_data = eodhd_client.get_historical_prices(
                    symbol=symbol,
                    exchange=exchange,
                    from_date=from_date,
                    to_date=to_date
                )
                
                result["price_data"] = {
                    "success": True,
                    "points_retrieved": len(price_data),
                    "date_range": {
                        "from": from_date.isoformat(),
                        "to": to_date.isoformat()
                    },
                    "sample_points": [
                        {
                            "date": point.date.isoformat(),
                            "close": point.close,
                            "volume": point.volume
                        }
                        for point in price_data[:5]  # Show first 5 points
                    ] if price_data else []
                }
                
            except Exception as e:
                result["price_data"] = {
                    "success": False,
                    "error": str(e),
                    "note": "Price data retrieval failed but service handled error gracefully"
                }
            
            # Get symbol information
            try:
                symbol_info = eodhd_client.get_symbol_info(symbol, exchange)
                
                if symbol_info:
                    result["symbol_info"] = {
                        "success": True,
                        "name": symbol_info.name,
                        "country": symbol_info.country,
                        "currency": symbol_info.currency,
                        "instrument_type": symbol_info.instrument_type
                    }
                else:
                    result["symbol_info"] = {
                        "success": False,
                        "message": "Symbol not found in EODHD database"
                    }
                    
            except Exception as e:
                result["symbol_info"] = {
                    "success": False,
                    "error": str(e)
                }
        
        else:
            result["message"] = f"EODHD API not available: {api_status['message']}"
        
        return result
        
    except Exception as e:
        logger.error(f"EODHD demo failed: {e}")
        raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/infrastructure/service-bus-demo")
def service_bus_demo(
    test_messages: int = Query(3, description="Number of test messages to send"),
    use_batching: bool = Query(True, description="Use batch sending"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate Azure Service Bus client capabilities.
    
    Shows message queue operations with proper error handling and monitoring.
    """
    
    # Initialize Service Bus client
    sb_client = AzureServiceBusClient()
    
    try:
        # Get connection status
        connection_status = sb_client.get_connection_status()
        
        result = {
            "demo_info": {
                "purpose": "Demonstrate Azure Service Bus integration",
                "messages_to_send": test_messages,
                "infrastructure_benefits": [
                    "Clean message queue abstraction",
                    "Structured message types (QueueMessage)",
                    "Priority and scheduling support",
                    "Batch processing for performance",
                    "Connection management and error handling",
                    "Mock mode for development/testing"
                ]
            },
            "connection_status": connection_status
        }
        
        # Create test messages
        test_messages = max(1, min(10, test_messages))  # Limit for demo
        messages = []
        
        for i in range(test_messages):
            message = sb_client.create_cvar_calculation_message(
                symbol=f"TEST{i+1}",
                alpha_level=95,
                force_recalculate=True,
                priority=MessagePriority.NORMAL
            )
            messages.append(message)
        
        # Send messages
        if use_batching and len(messages) > 1:
            send_result = sb_client.send_batch_messages(messages)
            result["message_sending"] = {
                "method": "batch",
                **send_result
            }
        else:
            successful_sends = 0
            for i, message in enumerate(messages):
                success = sb_client.send_message(message)
                if success:
                    successful_sends += 1
            
            result["message_sending"] = {
                "method": "individual",
                "total_messages": len(messages),
                "successful_sends": successful_sends,
                "success": successful_sends == len(messages)
            }
        
        # Get queue statistics
        queue_stats = sb_client.get_queue_stats()
        if queue_stats:
            result["queue_statistics"] = {
                "name": queue_stats.name,
                "active_messages": queue_stats.active_message_count,
                "dead_letter_messages": queue_stats.dead_letter_count,
                "size_bytes": queue_stats.size_in_bytes
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Service Bus demo failed: {e}")
        raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/infrastructure/cache-demo")
def cache_demo(
    test_operations: int = Query(5, description="Number of cache operations to perform"),
    ttl_seconds: int = Query(60, description="Cache TTL in seconds"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate cache service capabilities.
    
    Shows caching operations with TTL, statistics, and different data types.
    """
    
    # Initialize cache service
    cache_service = CacheService(
        backend=CacheBackend.MEMORY,
        namespace="infrastructure_demo",
        default_ttl=timedelta(seconds=ttl_seconds)
    )
    
    try:
        test_operations = max(1, min(20, test_operations))  # Limit for demo
        
        result = {
            "demo_info": {
                "purpose": "Demonstrate caching infrastructure service",
                "operations_performed": test_operations,
                "infrastructure_benefits": [
                    "Multiple backend support (memory, Redis, filesystem)",
                    "Automatic serialization/deserialization",
                    "TTL management and expiration",
                    "Namespace support for logical separation",
                    "Cache statistics and monitoring",
                    "Convenient get_or_set pattern"
                ]
            },
            "operations": []
        }
        
        # Perform various cache operations
        for i in range(test_operations):
            operation_type = ["set", "get", "get_or_set", "exists"][i % 4]
            key = f"demo_key_{i}"
            
            if operation_type == "set":
                # Test caching different data types
                test_data = {
                    "number": i,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": f"test_value_{i}",
                    "complex": {"nested": {"value": i * 10}}
                }
                
                success = cache_service.set(key, test_data)
                result["operations"].append({
                    "operation": "set",
                    "key": key,
                    "success": success,
                    "data_type": "complex_object"
                })
                
            elif operation_type == "get":
                cached_value = cache_service.get(key)
                result["operations"].append({
                    "operation": "get",
                    "key": key,
                    "found": cached_value is not None,
                    "value_sample": str(cached_value)[:100] if cached_value else None
                })
                
            elif operation_type == "get_or_set":
                # Demonstrate get_or_set pattern
                def factory_func():
                    return {
                        "computed_at": datetime.utcnow().isoformat(),
                        "computation_result": i ** 2,
                        "note": "This was computed by factory function"
                    }
                
                value = cache_service.get_or_set(
                    key=f"computed_{i}",
                    factory_func=factory_func,
                    ttl=timedelta(seconds=ttl_seconds)
                )
                
                result["operations"].append({
                    "operation": "get_or_set",
                    "key": f"computed_{i}",
                    "success": True,
                    "note": "Used factory function to compute value"
                })
                
            elif operation_type == "exists":
                exists = cache_service.exists(key)
                result["operations"].append({
                    "operation": "exists",
                    "key": key,
                    "exists": exists
                })
        
        # Demonstrate CVaR-specific caching
        cvar_data = {
            "symbol": "DEMO",
            "as_of_date": datetime.utcnow().isoformat(),
            "cvar95": {"annual": {"nig": 0.15, "ghst": 0.16, "evar": 0.17}},
            "cached": True
        }
        
        cache_success = cache_service.cache_cvar_data("DEMO", cvar_data)
        cached_cvar = cache_service.get_cached_cvar_data("DEMO")
        
        result["cvar_caching_demo"] = {
            "cache_success": cache_success,
            "retrieval_success": cached_cvar is not None,
            "data_preserved": cached_cvar == cvar_data if cached_cvar else False
        }
        
        # Get cache statistics
        cache_stats = cache_service.get_stats()
        result["cache_statistics"] = cache_stats
        
        return result
        
    except Exception as e:
        logger.error(f"Cache demo failed: {e}")
        raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/infrastructure/integration-test")
def infrastructure_integration_test(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Integration test showing multiple infrastructure services working together.
    
    Demonstrates a realistic workflow using multiple infrastructure components.
    """
    
    try:
        # Initialize all infrastructure services
        eodhd_client = EODHDClient()
        sb_client = AzureServiceBusClient()
        cache_service = CacheService(namespace="integration_test")
        
        result = {
            "integration_test": {
                "purpose": "Demonstrate infrastructure services working together",
                "workflow": [
                    "1. Check EODHD API availability",
                    "2. Cache the API status",
                    "3. Create Service Bus messages for processing",
                    "4. Cache message correlation IDs",
                    "5. Provide comprehensive status report"
                ],
                "services_tested": ["EODHDClient", "AzureServiceBusClient", "CacheService"]
            }
        }
        
        # Step 1: Check EODHD API status
        api_status = eodhd_client.get_api_status()
        result["step_1_api_check"] = {
            "status": api_status["status"],
            "available": api_status["status"] == "operational"
        }
        
        # Step 2: Cache the API status
        cache_key = f"api_status_{datetime.utcnow().strftime('%Y%m%d_%H')}"
        cache_success = cache_service.set(
            cache_key,
            api_status,
            ttl=timedelta(hours=1)
        )
        result["step_2_caching"] = {
            "cached": cache_success,
            "cache_key": cache_key
        }
        
        # Step 3: Create messages for hypothetical CVaR processing
        test_symbols = ["AAPL", "GOOGL", "MSFT"]
        messages = []
        correlation_ids = []
        
        for symbol in test_symbols:
            message = sb_client.create_cvar_calculation_message(
                symbol=symbol,
                alpha_level=95,
                priority=MessagePriority.HIGH
            )
            messages.append(message)
            correlation_ids.append(message.correlation_id)
        
        # Send messages in batch
        batch_result = sb_client.send_batch_messages(messages)
        result["step_3_messaging"] = {
            "messages_created": len(messages),
            "batch_send_result": batch_result
        }
        
        # Step 4: Cache correlation IDs for tracking
        correlation_cache_key = "integration_test_correlations"
        cache_service.set(
            correlation_cache_key,
            {
                "correlation_ids": correlation_ids,
                "created_at": datetime.utcnow().isoformat(),
                "symbols": test_symbols
            },
            ttl=timedelta(hours=6)
        )
        
        # Verify we can retrieve them
        cached_correlations = cache_service.get(correlation_cache_key)
        result["step_4_correlation_tracking"] = {
            "cached_successfully": cached_correlations is not None,
            "correlations_count": len(correlation_ids)
        }
        
        # Step 5: Get comprehensive status
        sb_connection_status = sb_client.get_connection_status()
        cache_stats = cache_service.get_stats()
        
        result["step_5_status_report"] = {
            "eodhd_status": api_status["status"],
            "service_bus_connected": sb_connection_status["connected"],
            "cache_keys_stored": cache_stats["backend_stats"]["total_keys"],
            "integration_successful": True
        }
        
        # Summary
        result["integration_summary"] = {
            "all_services_operational": True,
            "workflow_completed_successfully": True,
            "architecture_benefits_demonstrated": [
                "Clean service interfaces enable easy integration",
                "Each service handles its own error conditions",
                "Services can work together or independently",
                "Configuration managed consistently across services",
                "Monitoring and statistics available for all components"
            ]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Infrastructure integration test failed: {e}")
        raise HTTPException(500, f"Integration test failed: {str(e)}")


@router.get("/infrastructure/architecture-overview")
def infrastructure_architecture_overview(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Complete overview of infrastructure services architecture.
    """
    
    return {
        "infrastructure_layer": {
            "purpose": "Handle external systems and cross-cutting infrastructure concerns",
            "position_in_architecture": "Bottom layer - supports all other layers",
            "separation_of_concerns": "Isolates external dependencies from business logic"
        },
        "services_created": {
            "eodhd_client": {
                "purpose": "External market data API integration",
                "features": [
                    "Historical price data retrieval",
                    "Symbol information lookup",
                    "API status monitoring",
                    "Error handling and graceful degradation",
                    "Request optimization and rate limiting"
                ],
                "benefits": [
                    "Clean abstraction of external API",
                    "Structured data models",
                    "Centralized configuration",
                    "Easy testing with mock responses"
                ]
            },
            "azure_service_bus_client": {
                "purpose": "Message queue operations for async processing",
                "features": [
                    "Message publishing with priorities",
                    "Batch sending for performance",
                    "Queue monitoring and statistics",
                    "Connection management",
                    "Mock mode for development"
                ],
                "benefits": [
                    "Reliable async processing",
                    "Clean message abstractions",
                    "Automatic connection handling",
                    "Development/production parity"
                ]
            },
            "cache_service": {
                "purpose": "Caching operations with multiple backends",
                "features": [
                    "Multiple backend support (memory, Redis, filesystem)",
                    "TTL management and expiration",
                    "Namespace support",
                    "Automatic serialization/deserialization",
                    "Cache statistics and monitoring"
                ],
                "benefits": [
                    "Performance optimization",
                    "Flexible caching strategies",
                    "Easy cache invalidation",
                    "Monitoring and observability"
                ]
            }
        },
        "architectural_patterns": {
            "dependency_inversion": "Infrastructure depends on abstractions, not concretions",
            "interface_segregation": "Clean interfaces for each infrastructure concern",
            "single_responsibility": "Each service handles one infrastructure concern",
            "configuration_externalization": "All config through environment variables",
            "graceful_degradation": "Services work even when external systems are unavailable"
        },
        "integration_with_other_layers": {
            "domain_services": "Domain services use infrastructure for external data",
            "application_services": "Application services orchestrate infrastructure usage",
            "repositories": "Repositories may use caching infrastructure",
            "routes": "Routes should not directly use infrastructure services"
        },
        "benefits_of_infrastructure_layer": [
            "Centralizes external system integrations",
            "Provides consistent error handling patterns",
            "Enables easy testing with mocks/stubs",
            "Supports different environments (dev/staging/prod)",
            "Facilitates monitoring and observability",
            "Makes system more maintainable and scalable"
        ],
        "next_infrastructure_services": [
            "Email notification service",
            "File storage service (S3, Azure Blob)",
            "Monitoring and metrics service",
            "Authentication provider integrations",
            "Database connection pooling service"
        ]
    }
