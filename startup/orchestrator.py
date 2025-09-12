"""Startup orchestrator - coordinates all bootstrap phases."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Any, Union

from startup.infrastructure_bootstrap import run_infrastructure_bootstrap
from startup.data_bootstrap import run_data_bootstrap
from startup.business_bootstrap import run_business_bootstrap
from startup.servicebus import start_servicebus_consumer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure startup logs are visible


def run_startup_tasks() -> Dict[str, Union[str, Dict[str, Any], int]]:
    """
    Run all startup tasks in the correct order:
    1. Infrastructure (DB, RAG)
    2. Data (exchanges, symbols, market data)
    3. Business logic (CVaR, caching, Compass)
    
    Returns:
        Summary of all bootstrap results
    """
    logger.info("Starting application bootstrap...")
    all_results = {}
    
    # Phase 1: Infrastructure Bootstrap
    logger.info("Phase 1: Infrastructure bootstrap (DB, RAG)")
    try:
        infrastructure_results = run_infrastructure_bootstrap()
        all_results['infrastructure'] = infrastructure_results
        db_ready = infrastructure_results.get('database', False)
        
        if not db_ready:
            logger.error("Database not ready - aborting startup")
            all_results['status'] = 'failed'
            all_results['error'] = 'Database initialization failed'
            return all_results
        
        infra_count = len([k for k, v in infrastructure_results.items() if v not in ['error', 'skipped']])
        logger.info("Infrastructure bootstrap completed: %s", infrastructure_results)
    except Exception:
        logger.exception("Infrastructure bootstrap failed")
        all_results['infrastructure'] = {'error': 'Infrastructure bootstrap failed'}
        all_results['status'] = 'failed'
        return all_results
    
    # Phase 2: Data Bootstrap
    logger.info("Phase 2: Data bootstrap (exchanges, symbols, market data)")
    try:
        data_results = run_data_bootstrap(db_ready)
        all_results['data'] = data_results
        data_count = len([k for k, v in data_results.items() if v not in ['error', 'skipped']])
        logger.info("Data bootstrap completed: %d tasks", len(data_results))
    except Exception:
        logger.exception("Data bootstrap failed")
        all_results['data'] = {'error': 'Data bootstrap failed'}
        # Continue with business bootstrap even if data bootstrap fails
    
    # Initialize Service Bus Consumer before business logic
    logger.info("Initializing Service Bus Consumer for CVaR results...")
    try:
        # Start Service Bus consumer in a separate thread
        sb_thread = threading.Thread(
            target=start_servicebus_consumer,
            daemon=True,
            name="ServiceBusConsumer"
        )
        sb_thread.start()
        logger.info("Service Bus Consumer thread started")
        all_results['service_bus'] = True
    except Exception:
        logger.exception("Service Bus Consumer initialization failed")
        all_results['service_bus'] = 'error'
        # Continue even if Service Bus fails
    
    # Phase 3: Business Logic Bootstrap
    logger.info("Phase 3: Business logic bootstrap (CVaR, caching, Compass)")
    try:
        business_results = run_business_bootstrap(db_ready)
        all_results['business'] = business_results
        business_count = len([k for k, v in business_results.items() if v not in ['error', 'skipped']])
        logger.info("Business bootstrap completed: %d tasks", len(business_results))
    except Exception:
        logger.exception("Business bootstrap failed")
        all_results['business'] = {'error': 'Business bootstrap failed'}
        # This is non-fatal, application can still start
    
    # Summary
    all_results['status'] = 'completed'
    # Calculate detailed task statistics
    successful_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    total_tasks = 0
    
    for phase_name, phase_results in [
        ('infrastructure', all_results.get('infrastructure', {})),
        ('data', all_results.get('data', {})),
        ('business', all_results.get('business', {}))
    ]:
        if isinstance(phase_results, dict):
            for k, v in phase_results.items():
                if k != 'error':
                    total_tasks += 1
                    if v == 'skipped':
                        skipped_tasks += 1
                    elif v == 'error':
                        failed_tasks += 1
                    else:
                        successful_tasks += 1
    
    completion_msg = f"Application bootstrap completed: {successful_tasks} successful, {skipped_tasks} skipped, {failed_tasks} failed out of {total_tasks} total tasks"
    logger.info("COMPLETED: %s", completion_msg)
    
    return all_results


# Legacy compatibility function
def run_startup_tasks_legacy() -> None:
    """Legacy function for backward compatibility."""
    results = run_startup_tasks()
    # Log any critical errors for backward compatibility
    if results.get('status') == 'failed':
        logger.error("Startup failed: %s", results.get('error', 'Unknown error'))
    elif any('error' in phase for phase in [
        results.get('infrastructure', {}),
        results.get('data', {}),
        results.get('business', {})
    ]):
        logger.warning(
            "Startup completed with some errors - check logs for details"
        )
