"""
CVaR Orchestration Service - Complex CVaR workflows and cross-cutting concerns.

This application service orchestrates between multiple domain services and repositories
to handle complex CVaR operations like batch recalculations, data validation,
and performance monitoring.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.domain.cvar_unified_service import CvarUnifiedService
from repositories import (
    CvarRepository, 
    PriceSeriesRepository, 
    ValidationRepository
)

logger = logging.getLogger(__name__)


class CvarOrchestrationService:
    """
    Application service for orchestrating complex CVaR operations.
    
    This service coordinates between domain services and repositories
    to handle batch processing, validation workflows, and reporting.
    """
    
    def __init__(self):
        self.cvar_domain = CvarUnifiedService()
        self.cvar_repo = CvarRepository()
        self.price_repo = PriceSeriesRepository()
        self.validation_repo = ValidationRepository()
    
    def batch_recalculate_cvar(
        self,
        symbols: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Orchestrate batch CVaR recalculation with parallel processing.
        
        Args:
            symbols: Specific symbols to process, or None to use filters
            filters: Symbol filtering criteria (five_stars, country, etc.)
            max_workers: Maximum parallel workers for processing
            verbose: Include detailed results in response
            
        Returns:
            Batch processing results with statistics and details
        """
        
        try:
            # Determine symbols to process
            if symbols:
                target_symbols = symbols
            else:
                # Use filters to get symbols from repository
                filter_params = filters or {}
                target_symbols = self.price_repo.get_symbols_by_filters(
                    five_stars=filter_params.get("five_stars", False),
                    ready_only=filter_params.get("ready_only", True),
                    include_unknown=filter_params.get("include_unknown", False),
                    country=filter_params.get("country"),
                    instrument_types=filter_params.get("instrument_types"),
                    exclude_exchanges=filter_params.get("exclude_exchanges"),
                    limit=filter_params.get("limit")
                )
            
            if not target_symbols:
                return {
                    "success": False,
                    "error": "No symbols found matching criteria",
                    "symbols_processed": 0
                }
            
            # Initialize processing statistics
            stats = {
                "total_symbols": len(target_symbols),
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "start_time": datetime.utcnow(),
                "details": [] if verbose else None
            }
            
            # Process symbols in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(
                        self._process_single_symbol,
                        symbol
                    ): symbol
                    for symbol in target_symbols
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        result = future.result()
                        
                        if result["status"] == "success":
                            stats["successful"] += 1
                        elif result["status"] == "failed":
                            stats["failed"] += 1
                        else:
                            stats["skipped"] += 1
                        
                        # Add detailed results if verbose
                        if verbose and stats["details"] is not None:
                            stats["details"].append(result)
                            
                    except Exception as e:
                        logger.error(f"Future processing failed for {symbol}: {e}")
                        stats["failed"] += 1
                        
                        if verbose and stats["details"] is not None:
                            stats["details"].append({
                                "symbol": symbol,
                                "status": "error",
                                "error": str(e)
                            })
            
            # Calculate final statistics
            stats["end_time"] = datetime.utcnow()
            stats["duration_seconds"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()
            stats["success_rate"] = (
                (stats["successful"] / stats["total_symbols"]) * 100
                if stats["total_symbols"] > 0 else 0.0
            )
            
            # Log summary
            logger.info(
                f"Batch CVaR recalculation completed: {stats['successful']}/{stats['total_symbols']} "
                f"successful in {stats['duration_seconds']:.1f}s"
            )
            
            return {
                "success": True,
                "statistics": stats,
                "symbols_processed": stats["total_symbols"],
                "successful_calculations": stats["successful"],
                "failed_calculations": stats["failed"],
                "filters_applied": filters,
                "processing_mode": "parallel_orchestrated"
            }
            
        except Exception as e:
            logger.error(f"Batch recalculation orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbols_processed": 0
            }
    
    def validate_and_refresh_stale_data(
        self,
        max_age_days: int = 7,
        max_symbols: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Identify and refresh stale CVaR data.
        
        This workflow finds symbols with outdated CVaR data and triggers
        recalculation while updating validation status.
        """
        
        try:
            # Get symbols needing updates using repository
            stale_symbols = self.cvar_repo.get_symbols_needing_update(
                max_age_days=max_age_days
            )
            
            if max_symbols and len(stale_symbols) > max_symbols:
                stale_symbols = stale_symbols[:max_symbols]
            
            logger.info(f"Found {len(stale_symbols)} symbols with stale CVaR data")
            
            if not stale_symbols:
                return {
                    "success": True,
                    "message": "All CVaR data is fresh",
                    "stale_symbols_found": 0,
                    "symbols_refreshed": 0
                }
            
            # Process stale symbols
            refresh_result = self.batch_recalculate_cvar(
                symbols=stale_symbols,
                max_workers=6,  # Higher parallelism for refresh operations
                verbose=False
            )
            
            # Update validation status for processed symbols
            self._update_validation_status_batch(stale_symbols)
            
            return {
                "success": refresh_result["success"],
                "stale_symbols_found": len(stale_symbols),
                "symbols_refreshed": refresh_result.get("successful_calculations", 0),
                "refresh_statistics": refresh_result.get("statistics"),
                "max_age_days": max_age_days
            }
            
        except Exception as e:
            logger.error(f"Stale data refresh failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stale_symbols_found": 0
            }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive CVaR system health report.
        
        This orchestrates data collection from multiple repositories
        to provide system-wide health metrics.
        """
        
        try:
            # Get validation summary
            validation_summary = self.validation_repo.get_validation_summary()
            
            # Get data freshness metrics
            fresh_symbols = self.cvar_repo.get_symbols_with_fresh_data(max_age_days=1)
            week_fresh_symbols = self.cvar_repo.get_symbols_with_fresh_data(max_age_days=7)
            
            # Get country distribution analysis
            all_symbols = self.price_repo.get_symbols_by_filters(ready_only=False)
            country_analysis = self.price_repo.get_country_mix_analysis(all_symbols)
            
            # Calculate health scores
            total_symbols = validation_summary.get("total_symbols", 1)
            data_quality_score = validation_summary.get("data_quality_percentage", 0.0)
            
            freshness_score = (len(week_fresh_symbols) / total_symbols) * 100 if total_symbols > 0 else 0.0
            
            overall_health = (data_quality_score + freshness_score) / 2
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health_score": round(overall_health, 2),
                "data_quality": {
                    "score": data_quality_score,
                    "total_symbols": total_symbols,
                    "valid_symbols": validation_summary.get("sufficient_data", 0),
                    "invalid_symbols": validation_summary.get("insufficient_data", 0)
                },
                "data_freshness": {
                    "score": round(freshness_score, 2),
                    "fresh_today": len(fresh_symbols),
                    "fresh_this_week": len(week_fresh_symbols),
                    "total_symbols": total_symbols
                },
                "geographic_distribution": country_analysis,
                "recent_anomalies": validation_summary.get("recent_anomalies", 0),
                "recommendations": self._generate_health_recommendations(
                    data_quality_score, freshness_score
                )
            }
            
        except Exception as e:
            logger.error(f"Health report generation failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "overall_health_score": 0.0
            }
    
    def orchestrate_five_stars_analysis(
        self,
        country: Optional[str] = None,
        alpha_level: int = 99
    ) -> Dict[str, Any]:
        """
        Comprehensive five-star symbols analysis with CVaR data.
        
        This orchestrates data collection from multiple repositories
        and domain services to provide rich analytical insights.
        """
        
        try:
            # Get five-star symbols with CVaR data
            five_star_batch = self.cvar_repo.get_five_stars_batch(
                alpha_label=alpha_level,
                country=country
            )
            
            if not five_star_batch:
                return {
                    "success": True,
                    "message": "No five-star symbols found with specified criteria",
                    "total_symbols": 0,
                    "alpha_level": alpha_level,
                    "country_filter": country
                }
            
            # Analyze CVaR distribution
            cvar_values = [item["value"] for item in five_star_batch if item["value"] is not None]
            
            analysis = {
                "success": True,
                "total_symbols": len(five_star_batch),
                "symbols_with_cvar": len(cvar_values),
                "alpha_level": alpha_level,
                "country_filter": country,
                "cvar_statistics": self._calculate_cvar_statistics(cvar_values) if cvar_values else None,
                "top_performers": sorted(
                    [item for item in five_star_batch if item["value"] is not None],
                    key=lambda x: x["value"] or float('inf')
                )[:10],  # Top 10 lowest risk
                "data_completeness": {
                    "complete_records": len(cvar_values),
                    "incomplete_records": len(five_star_batch) - len(cvar_values),
                    "completeness_percentage": (
                        (len(cvar_values) / len(five_star_batch)) * 100
                        if five_star_batch else 0.0
                    )
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Five stars analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_symbols": 0
            }
    
    # Private helper methods
    
    def _process_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process CVaR calculation for a single symbol."""
        try:
            result = self.cvar_domain.process_symbol_calculation(
                symbol=symbol,
                force_recalculate=True
            )
            return result
        except Exception as e:
            logger.error(f"Symbol processing failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            }
    
    def _update_validation_status_batch(self, symbols: List[str]) -> None:
        """Update validation status for multiple symbols."""
        try:
            for symbol in symbols:
                # Simple validation update - in real implementation,
                # this would check actual calculation success
                self.validation_repo.upsert_validation_flags(
                    symbol=symbol,
                    validation_data={
                        "success": True,
                        "message": "Processed in batch refresh",
                        "code": "batch_refresh"
                    }
                )
        except Exception as e:
            logger.warning(f"Batch validation update failed: {e}")
    
    def _calculate_cvar_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for CVaR values."""
        if not values:
            return {}
        
        import numpy as np
        
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "percentile_25": float(np.percentile(values, 25)),
            "percentile_75": float(np.percentile(values, 75)),
            "count": len(values)
        }
    
    def _generate_health_recommendations(
        self, 
        data_quality_score: float, 
        freshness_score: float
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if data_quality_score < 80:
            recommendations.append(
                "Data quality is below 80%. Consider running validation cleanup."
            )
        
        if freshness_score < 70:
            recommendations.append(
                "Data freshness is below 70%. Schedule more frequent recalculations."
            )
        
        if data_quality_score > 95 and freshness_score > 90:
            recommendations.append(
                "System health is excellent. Consider optimizing performance further."
            )
        
        return recommendations
