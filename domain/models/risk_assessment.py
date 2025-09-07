"""
Risk Assessment domain models.

Models for CVaR calculations and risk assessments
using type-safe value objects and business rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import date, datetime
from uuid import UUID, uuid4
from enum import Enum

from domain.value_objects.symbol import Symbol
from domain.value_objects.risk_metrics import CVaRValue, CVaRTriple, AlphaLevel
from domain.value_objects.percentage import Percentage
from domain.value_objects.date_range import DateRange
from shared.exceptions import DataValidationError, BusinessLogicError


class CalculationMethod(Enum):
    """CVaR calculation methods."""
    NIG_GARCH = "nig"
    GHST_GARCH = "ghst" 
    EMPIRICAL = "evar"
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"


class ExecutionMode(Enum):
    """CVaR calculation execution modes."""
    LOCAL = "local"
    REMOTE = "remote"
    AUTO = "auto"


@dataclass
class CVaRCalculation:
    """
    Single CVaR calculation result.
    
    Represents one CVaR calculation with method, parameters, and metadata.
    """
    
    id: UUID
    symbol: Symbol
    cvar_value: CVaRValue
    calculation_date: date
    data_period: DateRange
    
    # Calculation parameters
    method: CalculationMethod
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    
    # Data quality metrics
    observations_used: int = 0
    data_completeness: Percentage = field(default_factory=Percentage.zero)
    outliers_detected: int = 0
    
    # Metadata
    calculation_time_ms: Optional[float] = None
    library_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Generate UUID if not provided
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())
        
        # Validate business rules
        self._validate()
    
    def _validate(self):
        """Validate business rules."""
        # Check date consistency
        if self.calculation_date < self.data_period.start_date:
            raise DataValidationError(
                f"Calculation date {self.calculation_date} cannot be before data period start {self.data_period.start_date}"
            )
        
        # Data quality validation
        if self.observations_used < 0:
            raise DataValidationError("Observations used cannot be negative")
        
        if self.outliers_detected < 0:
            raise DataValidationError("Outliers detected cannot be negative")
        
        if self.calculation_time_ms is not None and self.calculation_time_ms < 0:
            raise DataValidationError("Calculation time cannot be negative")
        
        # Method consistency
        if self.method.value != self.cvar_value.method and self.cvar_value.method != "unknown":
            raise DataValidationError(
                f"Calculation method {self.method.value} does not match CVaR method {self.cvar_value.method}"
            )
    
    @classmethod
    def create(
        cls,
        symbol: str,
        cvar_value: float,
        alpha: int,
        method: str,
        calculation_date: date = None,
        data_start: date = None,
        data_end: date = None,
        **kwargs
    ) -> CVaRCalculation:
        """
        Factory method to create CVaRCalculation from basic parameters.
        
        Args:
            symbol: Symbol string
            cvar_value: CVaR value as decimal
            alpha: Alpha level (50, 95, 99)
            method: Calculation method string
            calculation_date: Date of calculation
            data_start: Data period start
            data_end: Data period end
            **kwargs: Additional parameters
            
        Returns:
            CVaRCalculation instance
        """
        symbol_obj = Symbol(symbol)
        cvar_obj = CVaRValue.from_decimal(cvar_value, alpha, method)
        method_enum = CalculationMethod(method)
        
        if calculation_date is None:
            calculation_date = date.today()
        
        if data_end is None:
            data_end = calculation_date
        
        if data_start is None:
            # Default to 5 years of data
            data_period = DateRange.from_years(data_end, 5.0)
        else:
            data_period = DateRange(data_start, data_end)
        
        return cls(
            id=uuid4(),
            symbol=symbol_obj,
            cvar_value=cvar_obj,
            calculation_date=calculation_date,
            data_period=data_period,
            method=method_enum,
            **kwargs
        )
    
    @property
    def is_recent(self, days: int = 7) -> bool:
        """Check if calculation is recent."""
        return (date.today() - self.calculation_date).days <= days
    
    @property
    def data_years(self) -> float:
        """Get years of data used."""
        return self.data_period.years
    
    @property
    def is_high_quality(self) -> bool:
        """Check if calculation is high quality."""
        return (
            self.data_completeness.to_decimal() >= 0.95 and  # 95% data completeness
            self.observations_used >= 500 and  # At least 500 observations
            self.outliers_detected / max(1, self.observations_used) <= 0.05  # Max 5% outliers
        )
    
    @property
    def quality_score(self) -> Percentage:
        """Calculate overall quality score."""
        score = 0.0
        
        # Data completeness (40% weight)
        score += float(self.data_completeness.to_decimal()) * 0.4
        
        # Sample size (30% weight)
        obs_score = min(1.0, self.observations_used / 1000)  # 1000 obs = full score
        score += obs_score * 0.3
        
        # Outlier rate (20% weight) - fewer outliers is better
        if self.observations_used > 0:
            outlier_rate = self.outliers_detected / self.observations_used
            outlier_score = max(0.0, 1.0 - outlier_rate * 10)  # 10% outliers = 0 score
        else:
            outlier_score = 0.0
        score += outlier_score * 0.2
        
        # Recency (10% weight)
        days_old = (date.today() - self.calculation_date).days
        recency_score = max(0.0, 1.0 - days_old / 30)  # 30 days = 0 score
        score += recency_score * 0.1
        
        return Percentage.from_decimal(min(1.0, score))
    
    def is_comparable_to(self, other: CVaRCalculation) -> bool:
        """
        Check if this calculation is comparable to another.
        
        Args:
            other: Other CVaRCalculation
            
        Returns:
            True if calculations are comparable
        """
        if not isinstance(other, CVaRCalculation):
            return False
        
        return (
            self.cvar_value.alpha == other.cvar_value.alpha and
            abs(self.data_years - other.data_years) <= 1.0 and  # Within 1 year of data
            abs((self.calculation_date - other.calculation_date).days) <= 30  # Within 30 days
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "symbol": self.symbol.value,
            "cvar": self.cvar_value.to_dict(),
            "calculation_date": self.calculation_date.isoformat(),
            "data_period": self.data_period.to_dict(),
            "method": self.method.value,
            "execution_mode": self.execution_mode.value,
            "observations_used": self.observations_used,
            "data_completeness": float(self.data_completeness.to_decimal()),
            "outliers_detected": self.outliers_detected,
            "quality_score": float(self.quality_score.to_decimal()),
            "is_high_quality": self.is_high_quality,
            "calculation_time_ms": self.calculation_time_ms,
            "library_version": self.library_version,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class RiskAssessment:
    """
    Comprehensive risk assessment for a financial instrument.
    
    Contains multiple CVaR calculations at different alpha levels
    and provides risk analysis methods.
    """
    
    id: UUID
    symbol: Symbol
    assessment_date: date
    
    # CVaR calculations by alpha level
    cvar_calculations: Dict[AlphaLevel, CVaRTriple] = field(default_factory=dict)
    
    # Overall assessment metrics
    overall_risk_rating: str = "UNKNOWN"  # LOW, MEDIUM, HIGH, VERY_HIGH
    confidence_level: Percentage = field(default_factory=lambda: Percentage.from_decimal(0.5))
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())
        
        self._validate()
    
    def _validate(self):
        """Validate business rules."""
        # Check that we have at least one CVaR calculation
        if not self.cvar_calculations:
            raise DataValidationError("Risk assessment must have at least one CVaR calculation")
        
        # Validate risk rating
        valid_ratings = {"LOW", "MEDIUM", "HIGH", "VERY_HIGH", "UNKNOWN"}
        if self.overall_risk_rating not in valid_ratings:
            raise DataValidationError(f"Invalid risk rating: {self.overall_risk_rating}")
    
    @classmethod
    def create(
        cls,
        symbol: str,
        assessment_date: date = None,
        **kwargs
    ) -> RiskAssessment:
        """
        Factory method to create RiskAssessment.
        
        Args:
            symbol: Symbol string
            assessment_date: Date of assessment
            **kwargs: Additional parameters
            
        Returns:
            RiskAssessment instance
        """
        if assessment_date is None:
            assessment_date = date.today()
        
        return cls(
            id=uuid4(),
            symbol=Symbol(symbol),
            assessment_date=assessment_date,
            **kwargs
        )
    
    def add_cvar_calculation(
        self,
        alpha_level: AlphaLevel,
        nig: Optional[CVaRValue] = None,
        ghst: Optional[CVaRValue] = None,
        evar: Optional[CVaRValue] = None
    ) -> None:
        """
        Add CVaR calculations for an alpha level.
        
        Args:
            alpha_level: Alpha level
            nig: NIG-GARCH CVaR
            ghst: GHST-GARCH CVaR
            evar: Empirical CVaR
        """
        triple = CVaRTriple(
            nig=nig,
            ghst=ghst,
            evar=evar,
            alpha=alpha_level
        )
        
        self.cvar_calculations[alpha_level] = triple
        self._update_overall_assessment()
        self._touch()
    
    def get_cvar_at_level(self, alpha: int) -> Optional[CVaRTriple]:
        """Get CVaR triple for specific alpha level."""
        alpha_level = AlphaLevel.from_int(alpha)
        return self.cvar_calculations.get(alpha_level)
    
    def get_worst_cvar_at_level(self, alpha: int) -> Optional[CVaRValue]:
        """Get worst CVaR at specific alpha level."""
        triple = self.get_cvar_at_level(alpha)
        return triple.worst if triple else None
    
    def get_all_worst_cvars(self) -> Dict[AlphaLevel, CVaRValue]:
        """Get worst CVaR for each alpha level."""
        worst_cvars = {}
        for alpha, triple in self.cvar_calculations.items():
            if triple.worst:
                worst_cvars[alpha] = triple.worst
        return worst_cvars
    
    def get_primary_cvar(self) -> Optional[CVaRValue]:
        """Get primary CVaR (worst at 99% level, fallback to 95%)."""
        # Try 99% first
        cvar_99 = self.get_worst_cvar_at_level(99)
        if cvar_99:
            return cvar_99
        
        # Fallback to 95%
        cvar_95 = self.get_worst_cvar_at_level(95)
        if cvar_95:
            return cvar_95
        
        # Last resort - 50%
        return self.get_worst_cvar_at_level(50)
    
    def _update_overall_assessment(self) -> None:
        """Update overall risk rating based on CVaR values."""
        primary_cvar = self.get_primary_cvar()
        if not primary_cvar:
            self.overall_risk_rating = "UNKNOWN"
            self.confidence_level = Percentage.from_decimal(0.0)
            return
        
        # Risk rating based on CVaR magnitude
        loss_magnitude = primary_cvar.loss_magnitude
        loss_percent = loss_magnitude.to_percent()
        
        if loss_percent <= 10:  # <= 10% loss
            self.overall_risk_rating = "LOW"
        elif loss_percent <= 25:  # <= 25% loss
            self.overall_risk_rating = "MEDIUM"
        elif loss_percent <= 50:  # <= 50% loss
            self.overall_risk_rating = "HIGH"
        else:  # > 50% loss
            self.overall_risk_rating = "VERY_HIGH"
        
        # Confidence based on data quality and consistency
        self._update_confidence_level()
    
    def _update_confidence_level(self) -> None:
        """Update confidence level based on calculation quality and consistency."""
        if not self.cvar_calculations:
            self.confidence_level = Percentage.from_decimal(0.0)
            return
        
        confidence_factors = []
        
        # Factor 1: Number of alpha levels
        num_levels = len(self.cvar_calculations)
        level_factor = min(1.0, num_levels / 3)  # Full confidence with 3+ levels
        confidence_factors.append(level_factor)
        
        # Factor 2: Method diversity
        all_methods = set()
        for triple in self.cvar_calculations.values():
            all_methods.update(triple.available_methods)
        
        method_factor = min(1.0, len(all_methods) / 3)  # Full confidence with 3 methods
        confidence_factors.append(method_factor)
        
        # Factor 3: Consistency between methods
        consistency_scores = []
        for triple in self.cvar_calculations.values():
            if triple.is_complete:
                # Calculate coefficient of variation
                values = [abs(v.value.to_decimal()) for v in [triple.nig, triple.ghst, triple.evar] if v]
                if len(values) >= 2:
                    mean_val = sum(values) / len(values)
                    if mean_val > 0:
                        std_dev = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                        cv = std_dev / mean_val
                        consistency_score = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
                        consistency_scores.append(consistency_score)
        
        if consistency_scores:
            consistency_factor = sum(consistency_scores) / len(consistency_scores)
        else:
            consistency_factor = 0.5  # Neutral if can't calculate
        
        confidence_factors.append(consistency_factor)
        
        # Overall confidence is average of factors
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        self.confidence_level = Percentage.from_decimal(overall_confidence)
    
    def is_recent(self, days: int = 30) -> bool:
        """Check if assessment is recent."""
        return (date.today() - self.assessment_date).days <= days
    
    def compare_risk_to(self, other: RiskAssessment) -> Dict[str, Any]:
        """
        Compare risk to another assessment.
        
        Args:
            other: Other RiskAssessment
            
        Returns:
            Comparison results
        """
        if not isinstance(other, RiskAssessment):
            raise TypeError("Can only compare to another RiskAssessment")
        
        my_cvar = self.get_primary_cvar()
        other_cvar = other.get_primary_cvar()
        
        if not my_cvar or not other_cvar:
            return {
                "comparable": False,
                "reason": "Missing primary CVaR values"
            }
        
        if my_cvar.alpha != other_cvar.alpha:
            return {
                "comparable": False,
                "reason": f"Different alpha levels: {my_cvar.alpha.value}% vs {other_cvar.alpha.value}%"
            }
        
        # Calculate relative risk
        my_loss = abs(my_cvar.value.to_decimal())
        other_loss = abs(other_cvar.value.to_decimal())
        
        if other_loss == 0:
            relative_risk = float('inf') if my_loss > 0 else 1.0
        else:
            relative_risk = float(my_loss / other_loss)
        
        return {
            "comparable": True,
            "my_cvar": my_cvar.to_dict(),
            "other_cvar": other_cvar.to_dict(),
            "relative_risk": relative_risk,
            "is_riskier": my_loss > other_loss,
            "risk_difference_percent": float((my_loss - other_loss) * 100),
            "my_rating": self.overall_risk_rating,
            "other_rating": other.overall_risk_rating
        }
    
    def _touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        cvar_dict = {}
        for alpha, triple in self.cvar_calculations.items():
            cvar_dict[str(alpha.value)] = triple.to_dict()
        
        return {
            "id": str(self.id),
            "symbol": self.symbol.value,
            "assessment_date": self.assessment_date.isoformat(),
            "cvar_calculations": cvar_dict,
            "overall_risk_rating": self.overall_risk_rating,
            "confidence_level": float(self.confidence_level.to_decimal()),
            "primary_cvar": self.get_primary_cvar().to_dict() if self.get_primary_cvar() else None,
            "is_recent": self.is_recent(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation."""
        primary_cvar = self.get_primary_cvar()
        cvar_str = f"{primary_cvar.value}" if primary_cvar else "N/A"
        return f"Risk Assessment for {self.symbol.value}: {cvar_str} ({self.overall_risk_rating})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"RiskAssessment(symbol='{self.symbol.value}', rating='{self.overall_risk_rating}', levels={len(self.cvar_calculations)})"
