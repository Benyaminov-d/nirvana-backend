"""
Application Services Demo Routes - Orchestration layer demonstration.

This module demonstrates how application services orchestrate between
multiple domain services and repositories to handle complex workflows.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
import logging

from services.application.cvar_orchestration_service import CvarOrchestrationService
from services.application.batch_processing_service import BatchProcessingService, BatchTask
from utils.auth import require_pub_or_basic as _require_pub_or_basic

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/orchestration/batch-recalc")
@router.get("/orchestration/batch-recalc") 
def orchestrated_batch_recalculation(
    symbols: Optional[str] = Query(None, description="Comma-separated symbols, or use filters"),
    five_stars: bool = Query(False, description="Process only five-star symbols"),
    country: Optional[str] = Query(None, description="Country filter"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated instrument types"),
    limit: int = Query(0, description="0=all, >0 to limit symbols"),
    max_workers: int = Query(4, description="Parallel workers (1-8)"),
    verbose: bool = Query(False, description="Include detailed results"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Orchestrated batch CVaR recalculation using application services.
    
    This endpoint demonstrates:
    - Application service orchestration between domain services and repositories
    - Parallel processing with configurable workers
    - Cross-cutting concerns like validation and monitoring
    - Complex workflow coordination
    """
    
    # Initialize application service
    orchestration_service = CvarOrchestrationService()
    
    try:
        # Parse input parameters
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        
        # Prepare filters if no specific symbols provided
        filters = None if symbol_list else {
            "five_stars": five_stars,
            "country": country,
            "instrument_types": [t.strip() for t in instrument_types.split(",") if t.strip()] if instrument_types else None,
            "ready_only": True,
            "limit": limit if limit > 0 else None
        }
        
        # Validate max_workers parameter
        max_workers = max(1, min(8, max_workers))
        
        # Execute orchestrated batch recalculation
        result = orchestration_service.batch_recalculate_cvar(
            symbols=symbol_list,
            filters=filters,
            max_workers=max_workers,
            verbose=verbose
        )
        
        # Add orchestration metadata
        result["orchestration_info"] = {
            "service_type": "application_orchestration",
            "layers_involved": [
                "application_service (orchestration)",
                "domain_service (business_logic)", 
                "repository (data_access)"
            ],
            "parallel_workers": max_workers,
            "cross_cutting_concerns": [
                "validation_updates",
                "error_handling",
                "performance_monitoring"
            ]
        }
        
        logger.info(
            f"Orchestrated batch recalc: {result.get('successful_calculations', 0)}"
            f"/{result.get('symbols_processed', 0)} symbols processed"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Orchestrated batch recalculation failed: {e}")
        raise HTTPException(500, f"Orchestration failed: {str(e)}")


@router.get("/orchestration/system-health")
def system_health_report(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Comprehensive system health report using orchestration service.
    
    Demonstrates cross-repository data aggregation and analysis.
    """
    
    orchestration_service = CvarOrchestrationService()
    
    try:
        health_report = orchestration_service.get_system_health_report()
        
        # Add architectural context
        health_report["architecture_info"] = {
            "data_sources": [
                "ValidationRepository (data quality metrics)",
                "CvarRepository (freshness and completeness)",
                "PriceSeriesRepository (symbol distribution)"
            ],
            "analysis_layers": [
                "Repository aggregation",
                "Application service orchestration", 
                "Cross-domain correlation"
            ],
            "benefits": [
                "Single API for complex analysis",
                "Consistent data aggregation patterns",
                "Reusable health monitoring"
            ]
        }
        
        return health_report
        
    except Exception as e:
        logger.error(f"System health report failed: {e}")
        raise HTTPException(500, f"Health report generation failed: {str(e)}")


@router.get("/orchestration/stale-data-refresh")
def refresh_stale_data(
    max_age_days: int = Query(7, description="Consider data stale after this many days"),
    max_symbols: int = Query(100, description="Maximum symbols to refresh in one batch"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Intelligent stale data refresh using orchestration workflow.
    
    This demonstrates complex workflow orchestration:
    1. Identify stale data using repository queries
    2. Prioritize refresh candidates  
    3. Execute parallel recalculation
    4. Update validation status
    5. Generate completion report
    """
    
    orchestration_service = CvarOrchestrationService()
    
    try:
        # Validate parameters
        max_age_days = max(1, min(30, max_age_days))
        max_symbols = max(1, min(500, max_symbols))
        
        # Execute orchestrated refresh workflow
        refresh_result = orchestration_service.validate_and_refresh_stale_data(
            max_age_days=max_age_days,
            max_symbols=max_symbols
        )
        
        # Add workflow metadata
        refresh_result["workflow_info"] = {
            "orchestration_steps": [
                "1. Query stale data using CvarRepository",
                "2. Prioritize refresh candidates",
                "3. Execute parallel CVaR recalculation", 
                "4. Update validation flags via ValidationRepository",
                "5. Generate comprehensive report"
            ],
            "services_coordinated": [
                "CvarOrchestrationService (workflow)",
                "CvarUnifiedService (domain logic)",
                "CvarRepository, ValidationRepository (data access)"
            ],
            "parameters_applied": {
                "max_age_days": max_age_days,
                "max_symbols": max_symbols
            }
        }
        
        return refresh_result
        
    except Exception as e:
        logger.error(f"Stale data refresh failed: {e}")
        raise HTTPException(500, f"Refresh workflow failed: {str(e)}")


@router.get("/orchestration/five-stars-analysis")
def orchestrated_five_stars_analysis(
    country: Optional[str] = Query(None, description="Country filter (US, Canada, etc.)"),
    alpha_level: int = Query(99, description="Alpha level: 50, 95, or 99"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Comprehensive five-star analysis using orchestration patterns.
    
    Demonstrates rich analytical workflows that combine:
    - Repository queries for data retrieval
    - Statistical analysis and ranking
    - Multi-dimensional data correlation
    """
    
    orchestration_service = CvarOrchestrationService()
    
    try:
        # Validate alpha level
        if alpha_level not in (50, 95, 99):
            alpha_level = 99
        
        # Execute orchestrated analysis
        analysis_result = orchestration_service.orchestrate_five_stars_analysis(
            country=country,
            alpha_level=alpha_level
        )
        
        # Add analytical methodology info
        analysis_result["methodology"] = {
            "data_integration": [
                "Five-star flags from PriceSeries",
                "Latest CVaR snapshots by alpha level",
                "Statistical distribution analysis"
            ],
            "analytical_steps": [
                "1. Filter five-star symbols by country/alpha",
                "2. Retrieve latest CVaR data for each symbol", 
                "3. Calculate statistical measures (mean, median, percentiles)",
                "4. Rank symbols by risk (ascending CVaR)",
                "5. Assess data completeness and quality"
            ],
            "orchestration_benefits": [
                "Complex analysis in single API call",
                "Consistent data correlation patterns",
                "Reusable analytical components"
            ]
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Five stars analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.get("/orchestration/batch-processing-demo")
def batch_processing_demo(
    task_count: int = Query(10, description="Number of demo tasks to create"),
    max_workers: int = Query(3, description="Parallel workers"),
    include_failures: bool = Query(True, description="Include some failing tasks for demo"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Generic batch processing service demonstration.
    
    Shows how the BatchProcessingService can handle any type of 
    parallel processing with monitoring and error handling.
    """
    
    batch_service = BatchProcessingService()
    
    try:
        # Validate parameters
        task_count = max(1, min(50, task_count))
        max_workers = max(1, min(8, max_workers))
        
        # Create demo tasks with varying complexity
        demo_tasks = []
        for i in range(task_count):
            # Create some tasks that will fail for demonstration
            will_fail = include_failures and i % 7 == 0
            
            demo_tasks.append(BatchTask(
                id=f"demo_task_{i}",
                data={
                    "task_number": i,
                    "complexity": i % 3 + 1,  # 1-3 complexity levels
                    "should_fail": will_fail
                },
                priority=i % 5  # Vary priority for demonstration
            ))
        
        # Define a demo processor function
        def demo_processor(task_data: Dict[str, Any]) -> Dict[str, Any]:
            import time
            import random
            
            # Simulate work based on complexity
            complexity = task_data.get("complexity", 1)
            work_time = complexity * 0.1 + random.uniform(0.05, 0.15)
            time.sleep(work_time)
            
            # Simulate failures
            if task_data.get("should_fail", False):
                raise Exception(f"Simulated failure for task {task_data['task_number']}")
            
            return {
                "task_number": task_data["task_number"],
                "processing_time": work_time,
                "result_value": task_data["task_number"] * complexity,
                "status": "completed"
            }
        
        # Execute batch with comprehensive monitoring
        batch_result = batch_service.execute_batch(
            tasks=demo_tasks,
            processor_func=demo_processor,
            batch_id="orchestration_demo",
            max_workers=max_workers,
            retry_failed=True,
            max_retries=1
        )
        
        # Add demonstration context
        batch_result["demo_info"] = {
            "purpose": "Demonstrate generic batch processing capabilities",
            "features_shown": [
                "Parallel task execution",
                "Priority-based task ordering",
                "Comprehensive statistics tracking",
                "Error handling and retry logic",
                "Progress monitoring",
                "Resource management"
            ],
            "real_world_applications": [
                "CVaR batch recalculations",
                "Data validation workflows", 
                "Report generation",
                "ETL processing",
                "Notification delivery"
            ],
            "parameters_used": {
                "task_count": task_count,
                "max_workers": max_workers,
                "failures_included": include_failures
            }
        }
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch processing demo failed: {e}")
        raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/orchestration/architecture-overview")
def architecture_overview(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Complete overview of the new layered architecture.
    """
    
    return {
        "architecture_layers": {
            "presentation": {
                "layer": "Routes/Controllers",
                "responsibility": "HTTP request/response handling",
                "examples": [
                    "application_services_demo.py - Orchestration demos",
                    "cvar_refactored_demo.py - Domain service demos", 
                    "ticker_refactored.py - Repository demos"
                ],
                "characteristics": [
                    "Thin layer - minimal business logic",
                    "Parameter validation and serialization",
                    "Authentication and authorization",
                    "Error handling and HTTP status codes"
                ]
            },
            "application": {
                "layer": "Application Services",
                "responsibility": "Workflow orchestration and cross-cutting concerns",
                "examples": [
                    "CvarOrchestrationService - Complex CVaR workflows",
                    "BatchProcessingService - Generic parallel processing"
                ],
                "characteristics": [
                    "Coordinates between multiple domain services",
                    "Handles cross-cutting concerns (logging, monitoring)",
                    "Manages complex workflows and transactions",
                    "Provides coarse-grained business operations"
                ]
            },
            "domain": {
                "layer": "Domain Services",
                "responsibility": "Pure business logic and domain rules",
                "examples": [
                    "CvarUnifiedService - CVaR computation logic"
                ],
                "characteristics": [
                    "Encapsulates core business rules",
                    "Independent of external dependencies",
                    "Highly testable and reusable",
                    "Domain-specific operations"
                ]
            },
            "data_access": {
                "layer": "Repositories",
                "responsibility": "Data access abstraction",
                "examples": [
                    "CvarRepository - CVaR data operations",
                    "PriceSeriesRepository - Price/symbol data",
                    "UserRepository, ValidationRepository, etc."
                ],
                "characteristics": [
                    "Abstract database operations",
                    "Centralized query logic",
                    "Automatic session management",
                    "Type-safe data access"
                ]
            }
        },
        "benefits_realized": {
            "separation_of_concerns": "Each layer has single responsibility",
            "testability": "Each layer can be tested independently",
            "maintainability": "Changes isolated to appropriate layers",
            "reusability": "Components can be reused across contexts",
            "scalability": "Easy to optimize individual layers",
            "flexibility": "Easy to swap implementations"
        },
        "migration_progress": {
            "repositories": "✅ Complete - 6 repositories with 60+ methods",
            "domain_services": "IN_PROGRESS - 1 created, more planned",
            "application_services": "✅ Created - 2 services with orchestration",
            "route_migration": "IN_PROGRESS - 5 endpoints migrated",
            "legacy_elimination": "PLANNED - Remove old patterns"
        },
        "next_steps": [
            "Complete route migration from legacy patterns",
            "Create infrastructure services for external integrations",
            "Add comprehensive monitoring and metrics",
            "Implement caching strategies at appropriate layers",
            "Add integration tests for complete workflows"
        ]
    }
