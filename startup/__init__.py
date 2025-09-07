"""Startup orchestration package.

Provides small, testable units for app initialization.
Consolidated from 12 files into 4 focused modules:
- infrastructure_bootstrap: DB, RAG, external services
- data_bootstrap: exchanges, symbols, market data  
- business_bootstrap: CVaR, caching, Compass
- orchestrator: coordinates all phases
"""


def run_startup_tasks() -> None:
    """Run all startup tasks using the new consolidated architecture."""
    # Lazy import to avoid import-time graph issues
    from startup.orchestrator import run_startup_tasks_legacy
    
    run_startup_tasks_legacy()
