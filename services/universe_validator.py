"""
Harvard Universe Validator

Advanced validation and consistency checking for Harvard universe.
Provides detailed analysis of universe health and data quality.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from core.db import get_db_session
from core.models import PriceSeries, CompassInputs, CvarSnapshot, CompassAnchor
from core.universe_config import HarvardUniverseConfig, UniverseFeatureFlags
from services.universe_manager import get_harvard_universe_manager


_LOG = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the universe."""
    severity: str  # "error", "warning", "info"
    category: str  # "data_missing", "data_stale", "configuration", "consistency"
    description: str
    affected_items: List[str] = None
    recommended_action: str = ""
    
    def __post_init__(self):
        if self.affected_items is None:
            self.affected_items = []


@dataclass  
class DataQualityReport:
    """Detailed data quality analysis for universe."""
    total_products: int
    data_coverage: Dict[str, float]  # percentage coverage by data type
    freshness_stats: Dict[str, Dict[str, Any]]  # age statistics by data type
    outlier_analysis: Dict[str, List[str]]  # products with outlier values
    consistency_checks: Dict[str, bool]  # various consistency test results


class HarvardUniverseValidator:
    """
    Advanced validator for Harvard universe integrity and quality.
    
    Performs comprehensive validation including:
    - Data completeness and coverage
    - Data quality and outlier detection  
    - Temporal consistency and freshness
    - Cross-system consistency checks
    - Configuration validation
    """
    
    def __init__(self):
        self.config = HarvardUniverseConfig()
        self.session = get_db_session()
        if not self.session:
            raise RuntimeError("Failed to create database session")
    
    def validate_full_universe(self) -> Dict[str, Any]:
        """
        Perform comprehensive universe validation.
        
        Returns:
            Complete validation report with issues and recommendations
        """
        _LOG.info("Starting comprehensive Harvard universe validation")
        
        validation_start = datetime.utcnow()
        issues: List[ValidationIssue] = []
        
        try:
            # Get universe manager for basic stats
            manager = get_harvard_universe_manager()
            universe_stats = manager.get_universe_stats()
            
            # 1. Configuration validation
            config_issues = self._validate_configuration()
            issues.extend(config_issues)
            
            # 2. Data completeness validation
            completeness_issues = self._validate_data_completeness()
            issues.extend(completeness_issues)
            
            # 3. Data quality validation
            quality_issues = self._validate_data_quality()
            issues.extend(quality_issues)
            
            # 4. Temporal consistency validation
            temporal_issues = self._validate_temporal_consistency()
            issues.extend(temporal_issues)
            
            # 5. Cross-system consistency validation
            consistency_issues = self._validate_cross_system_consistency()
            issues.extend(consistency_issues)
            
            # 6. Performance and capacity validation
            capacity_issues = self._validate_capacity_metrics()
            issues.extend(capacity_issues)
            
            # Categorize issues by severity
            errors = [i for i in issues if i.severity == "error"]
            warnings = [i for i in issues if i.severity == "warning"]
            info = [i for i in issues if i.severity == "info"]
            
            # Generate data quality report
            quality_report = self._generate_data_quality_report()
            
            validation_duration = (datetime.utcnow() - validation_start).total_seconds()
            
            return {
                "timestamp": validation_start.isoformat(),
                "validation_duration_seconds": validation_duration,
                "universe_stats": universe_stats,
                "summary": {
                    "total_issues": len(issues),
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "info": len(info),
                    "overall_health": "healthy" if len(errors) == 0 else "issues_found",
                },
                "issues": {
                    "errors": [self._issue_to_dict(i) for i in errors],
                    "warnings": [self._issue_to_dict(i) for i in warnings],
                    "info": [self._issue_to_dict(i) for i in info],
                },
                "data_quality": quality_report,
                "recommendations": self._generate_recommendations(issues),
            }
            
        except Exception as exc:
            _LOG.error("Universe validation failed: %s", exc)
            return {
                "timestamp": validation_start.isoformat(),
                "error": str(exc),
                "summary": {"overall_health": "validation_failed"},
            }
        finally:
            if self.session:
                self.session.close()
    
    def _validate_configuration(self) -> List[ValidationIssue]:
        """Validate universe configuration settings."""
        issues = []
        
        try:
            enabled_countries = self.config.get_enabled_countries()
            
            if not enabled_countries:
                issues.append(ValidationIssue(
                    severity="error",
                    category="configuration",
                    description="No countries enabled in Harvard universe",
                    recommended_action="Enable at least one country in configuration"
                ))
            
            for country_code, country_config in enabled_countries.items():
                # Check compass categories are defined
                if not country_config.compass_category:
                    issues.append(ValidationIssue(
                        severity="error", 
                        category="configuration",
                        description=f"No compass category defined for {country_code}",
                        affected_items=[country_code],
                        recommended_action="Define compass_category in country config"
                    ))
                
                # Check minimum thresholds make sense
                if country_config.min_market_cap and country_config.min_market_cap < 1_000_000:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="configuration", 
                        description=f"Very low market cap threshold for {country_code}",
                        affected_items=[country_code],
                        recommended_action="Review market cap threshold"
                    ))
                
        except Exception as exc:
            issues.append(ValidationIssue(
                severity="error",
                category="configuration",
                description=f"Configuration validation failed: {exc}"
            ))
        
        return issues
    
    def _validate_data_completeness(self) -> List[ValidationIssue]:
        """Validate data completeness across the universe."""
        issues = []
        
        try:
            manager = get_harvard_universe_manager()
            products = manager.get_universe_products()
            
            if not products:
                issues.append(ValidationIssue(
                    severity="error",
                    category="data_missing",
                    description="No products found in Harvard universe",
                    recommended_action="Check universe configuration and database"
                ))
                return issues
            
            # Calculate completion rates
            total_products = len(products)
            missing_mu = len([p for p in products if not p.has_mu])
            missing_cvar = len([p for p in products if not p.has_cvar])
            
            mu_completion_rate = (total_products - missing_mu) / total_products
            cvar_completion_rate = (total_products - missing_cvar) / total_products
            
            # Check completion thresholds
            if mu_completion_rate < 0.9:  # Less than 90% complete
                issues.append(ValidationIssue(
                    severity="error" if mu_completion_rate < 0.5 else "warning",
                    category="data_missing",
                    description=f"Low μ data completion: {mu_completion_rate:.1%} ({missing_mu}/{total_products} missing)",
                    recommended_action="Run universe completeness with auto_compute_mu=true"
                ))
            
            if cvar_completion_rate < 0.8:  # Less than 80% complete
                issues.append(ValidationIssue(
                    severity="warning",
                    category="data_missing", 
                    description=f"Low CVaR data completion: {cvar_completion_rate:.1%} ({missing_cvar}/{total_products} missing)",
                    recommended_action="Run universe completeness with auto_compute_cvar=true"
                ))
            
            # Check for anchor availability
            for country_code, country_config in self.config.get_enabled_countries().items():
                anchor = (
                    self.session.query(CompassAnchor)
                    .filter(CompassAnchor.category == country_config.compass_category)
                    .order_by(CompassAnchor.created_at.desc())
                    .first()
                )
                
                if not anchor:
                    issues.append(ValidationIssue(
                        severity="error",
                        category="data_missing",
                        description=f"No anchors found for {country_config.compass_category}",
                        affected_items=[country_code],
                        recommended_action="Recalibrate anchors for this category"
                    ))
                    
        except Exception as exc:
            issues.append(ValidationIssue(
                severity="error",
                category="data_missing",
                description=f"Completeness validation failed: {exc}"
            ))
        
        return issues
    
    def _validate_data_quality(self) -> List[ValidationIssue]:
        """Validate data quality and detect outliers."""
        issues = []
        
        try:
            # Check for outlier μ values
            mu_outliers = self._detect_mu_outliers()
            if mu_outliers:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="data_quality",
                    description=f"Found {len(mu_outliers)} products with outlier μ values",
                    affected_items=mu_outliers[:10],  # Limit to first 10
                    recommended_action="Review μ calculation for these products"
                ))
            
            # Check for outlier CVaR values  
            cvar_outliers = self._detect_cvar_outliers()
            if cvar_outliers:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="data_quality",
                    description=f"Found {len(cvar_outliers)} products with outlier CVaR values", 
                    affected_items=cvar_outliers[:10],  # Limit to first 10
                    recommended_action="Review CVaR calculation for these products"
                ))
            
        except Exception as exc:
            issues.append(ValidationIssue(
                severity="error",
                category="data_quality",
                description=f"Quality validation failed: {exc}"
            ))
        
        return issues
    
    def _validate_temporal_consistency(self) -> List[ValidationIssue]:
        """Validate temporal consistency and data freshness."""
        issues = []
        
        try:
            # Check anchor age
            cutoff_date = datetime.utcnow() - timedelta(days=120)  # 4 months
            
            for country_code, country_config in self.config.get_enabled_countries().items():
                anchor = (
                    self.session.query(CompassAnchor)
                    .filter(CompassAnchor.category == country_config.compass_category)
                    .order_by(CompassAnchor.created_at.desc())
                    .first()
                )
                
                if anchor and anchor.created_at < cutoff_date:
                    age_days = (datetime.utcnow() - anchor.created_at).days
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="data_stale",
                        description=f"Anchors for {country_config.compass_category} are {age_days} days old",
                        affected_items=[country_code],
                        recommended_action="Recalibrate anchors (should be quarterly)"
                    ))
            
            # Check for very old CVaR snapshots
            old_snapshots = self._count_stale_snapshots(days=90)
            if old_snapshots > 0:
                issues.append(ValidationIssue(
                    severity="info",
                    category="data_stale",
                    description=f"{old_snapshots} CVaR snapshots are older than 90 days",
                    recommended_action="Consider refreshing stale CVaR calculations"
                ))
                
        except Exception as exc:
            issues.append(ValidationIssue(
                severity="error",
                category="temporal",
                description=f"Temporal validation failed: {exc}"
            ))
        
        return issues
    
    def _validate_cross_system_consistency(self) -> List[ValidationIssue]:
        """Validate consistency across different data systems."""
        issues = []
        
        try:
            # Check for compass_inputs without corresponding snapshots
            orphaned_inputs = self._count_orphaned_compass_inputs()
            if orphaned_inputs > 0:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="consistency",
                    description=f"{orphaned_inputs} compass_inputs have no corresponding CVaR snapshots",
                    recommended_action="Ensure CVaR computation for all products with μ data"
                ))
            
            # Check for products in universe but missing from price_series
            # (This shouldn't happen but good to verify)
            
        except Exception as exc:
            issues.append(ValidationIssue(
                severity="error", 
                category="consistency",
                description=f"Consistency validation failed: {exc}"
            ))
        
        return issues
    
    def _validate_capacity_metrics(self) -> List[ValidationIssue]:
        """Validate system capacity and performance metrics."""
        issues = []
        
        try:
            # Check universe size vs configured limits
            manager = get_harvard_universe_manager()
            stats = manager.get_universe_stats()
            
            if stats.total_products > 50000:  # Arbitrary large threshold
                issues.append(ValidationIssue(
                    severity="info",
                    category="capacity",
                    description=f"Large universe size: {stats.total_products} products",
                    recommended_action="Monitor processing times and consider optimization"
                ))
            
        except Exception as exc:
            issues.append(ValidationIssue(
                severity="warning",
                category="capacity", 
                description=f"Capacity validation failed: {exc}"
            ))
        
        return issues
    
    def _generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate detailed data quality report."""
        try:
            manager = get_harvard_universe_manager()
            products = manager.get_universe_products()
            
            total = len(products)
            if total == 0:
                return {"error": "No products in universe"}
            
            # Calculate coverage percentages
            mu_coverage = len([p for p in products if p.has_mu]) / total
            cvar_coverage = len([p for p in products if p.has_cvar]) / total
            
            return {
                "total_products": total,
                "data_coverage": {
                    "mu": mu_coverage,
                    "cvar": cvar_coverage,
                    "both": len([p for p in products if p.has_mu and p.has_cvar]) / total,
                },
                "by_country": {
                    country: {
                        "total": len([p for p in products if p.country == country]),
                        "mu_coverage": len([p for p in products if p.country == country and p.has_mu]) / max(1, len([p for p in products if p.country == country])),
                        "cvar_coverage": len([p for p in products if p.country == country and p.has_cvar]) / max(1, len([p for p in products if p.country == country])),
                    }
                    for country in set(p.country for p in products)
                },
            }
            
        except Exception as exc:
            return {"error": f"Failed to generate quality report: {exc}"}
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on validation issues."""
        recommendations = []
        
        # Group issues by category
        by_category = defaultdict(list)
        for issue in issues:
            by_category[issue.category].append(issue)
        
        # Generate category-specific recommendations
        if "data_missing" in by_category:
            recommendations.append(
                "Run 'python commands/harvard_universe.py ensure-completeness' to compute missing data"
            )
        
        if "data_stale" in by_category:
            recommendations.append(
                "Run 'python commands/harvard_universe.py recalibrate-anchors' to update stale anchors"
            )
        
        if "configuration" in by_category:
            recommendations.append(
                "Review universe configuration in core/universe_config.py"
            )
        
        if "data_quality" in by_category:
            recommendations.append(
                "Investigate outlier products and consider data validation rules"
            )
        
        # Add general recommendations
        if any(issue.severity == "error" for issue in issues):
            recommendations.append(
                "Address ERROR level issues before deploying to production"
            )
        
        return recommendations
    
    def _detect_mu_outliers(self) -> List[str]:
        """Detect products with outlier μ values."""
        try:
            # Query μ values from compass_inputs
            results = (
                self.session.query(CompassInputs.instrument_id, CompassInputs.mu_i, PriceSeries.symbol)
                .join(PriceSeries, PriceSeries.id == CompassInputs.instrument_id)
                .filter(CompassInputs.mu_i.isnot(None))
                .all()
            )
            
            if len(results) < 10:  # Need reasonable sample size
                return []
            
            mu_values = [r.mu_i for r in results]
            
            # Simple outlier detection (3 standard deviations)
            import numpy as np
            mean = np.mean(mu_values)
            std = np.std(mu_values)
            threshold = 3 * std
            
            outliers = []
            for r in results:
                if abs(r.mu_i - mean) > threshold:
                    outliers.append(r.symbol)
            
            return outliers
            
        except Exception:
            return []
    
    def _detect_cvar_outliers(self) -> List[str]:
        """Detect products with outlier CVaR values."""
        try:
            # Query CVaR values from snapshots  
            results = (
                self.session.query(CvarSnapshot.symbol, CvarSnapshot.cvar_evar)
                .filter(
                    CvarSnapshot.alpha_label == 99,
                    CvarSnapshot.cvar_evar.isnot(None)
                )
                .all()
            )
            
            if len(results) < 10:
                return []
            
            cvar_values = [abs(r.cvar_evar) for r in results]
            
            # Simple outlier detection
            import numpy as np
            mean = np.mean(cvar_values)
            std = np.std(cvar_values)
            threshold = 3 * std
            
            outliers = []
            for r in results:
                if abs(abs(r.cvar_evar) - mean) > threshold:
                    outliers.append(r.symbol)
            
            return outliers
            
        except Exception:
            return []
    
    def _count_stale_snapshots(self, days: int) -> int:
        """Count CVaR snapshots older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            count = (
                self.session.query(CvarSnapshot)
                .filter(CvarSnapshot.as_of_date < cutoff_date)
                .count()
            )
            return count
        except Exception:
            return 0
    
    def _count_orphaned_compass_inputs(self) -> int:
        """Count compass_inputs without corresponding CVaR snapshots."""
        try:
            # This is a simplified check - in practice would need more sophisticated joins
            inputs_count = self.session.query(CompassInputs).count()
            snapshots_count = (
                self.session.query(CvarSnapshot)
                .filter(CvarSnapshot.alpha_label == 99)
                .count()
            )
            
            # Rough estimate of orphaned inputs
            return max(0, inputs_count - snapshots_count)
        except Exception:
            return 0
    
    def _issue_to_dict(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Convert ValidationIssue to dictionary."""
        return {
            "severity": issue.severity,
            "category": issue.category,
            "description": issue.description,
            "affected_items": issue.affected_items,
            "recommended_action": issue.recommended_action,
        }


# Global validator instance
_validator: Optional[HarvardUniverseValidator] = None

def get_harvard_validator() -> HarvardUniverseValidator:
    """Get the Harvard universe validator singleton."""
    global _validator
    if _validator is None:
        _validator = HarvardUniverseValidator()
    return _validator
