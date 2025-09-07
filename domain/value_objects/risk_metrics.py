"""
Risk metrics value objects for financial calculations.

Contains CVaR values, alpha levels, and other risk-related measurements
with proper validation and type safety.
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Union, Optional, Dict, Any
import math

from shared.exceptions import DataValidationError
from domain.value_objects.percentage import Percentage


class AlphaLevel(Enum):
    """Standard alpha levels for CVaR calculations."""
    ALPHA_50 = 50
    ALPHA_95 = 95  
    ALPHA_99 = 99
    
    @classmethod
    def from_int(cls, value: int) -> AlphaLevel:
        """Create AlphaLevel from integer."""
        for level in cls:
            if level.value == value:
                return level
        raise DataValidationError(f"Invalid alpha level: {value}. Must be one of {[l.value for l in cls]}")
    
    @classmethod
    def all_values(cls) -> list[int]:
        """Get all alpha level values."""
        return [level.value for level in cls]
    
    def to_confidence_level(self) -> Percentage:
        """Convert to confidence level percentage (95 -> 0.95)."""
        return Percentage.from_percent(self.value)
    
    def to_tail_probability(self) -> Percentage:
        """Convert to tail probability (95 -> 0.05)."""
        return Percentage.from_percent(100 - self.value)


@dataclass(frozen=True)
class CVaRValue:
    """
    Conditional Value at Risk (CVaR) measurement.
    
    Represents CVaR as a percentage loss with proper validation
    and methods for comparison and calculation.
    """
    
    value: Percentage  # CVaR as percentage (negative for losses)
    alpha: AlphaLevel  # Confidence level
    method: str        # Calculation method (nig, ghst, evar)
    
    def __init__(
        self, 
        value: Union[float, int, str, Decimal, Percentage], 
        alpha: Union[int, AlphaLevel],
        method: str = "unknown"
    ):
        """
        Create CVaRValue with validation.
        
        Args:
            value: CVaR value (typically negative percentage)
            alpha: Alpha level (50, 95, 99)
            method: Calculation method
        """
        # Validate and convert percentage
        if isinstance(value, Percentage):
            pct_value = value
        else:
            pct_value = Percentage.from_decimal(value)
        
        # Validate alpha level
        if isinstance(alpha, int):
            alpha_level = AlphaLevel.from_int(alpha)
        elif isinstance(alpha, AlphaLevel):
            alpha_level = alpha
        else:
            raise DataValidationError(f"Invalid alpha type: {type(alpha)}")
        
        # Validate method
        method = method.strip().lower()
        valid_methods = {'nig', 'ghst', 'evar', 'historical', 'parametric', 'unknown'}
        if method not in valid_methods:
            raise DataValidationError(f"Invalid CVaR method: {method}. Must be one of {valid_methods}")
        
        # CVaR validation - should typically be negative (losses)
        if pct_value.is_positive and pct_value.to_percent() > Decimal('1'):  # Allow small positive values due to rounding
            raise DataValidationError(f"CVaR value seems unusually positive: {pct_value}. CVaR typically represents losses (negative values).")
        
        # Reasonable bounds check
        if pct_value.to_percent() < Decimal('-200') or pct_value.to_percent() > Decimal('100'):
            raise DataValidationError(f"CVaR value outside reasonable range: {pct_value}")
        
        object.__setattr__(self, 'value', pct_value)
        object.__setattr__(self, 'alpha', alpha_level)
        object.__setattr__(self, 'method', method)
    
    @classmethod
    def from_decimal(cls, decimal_value: Union[float, str, Decimal], alpha: Union[int, AlphaLevel], method: str = "unknown") -> CVaRValue:
        """Create CVaRValue from decimal form (-0.25 for -25%)."""
        return cls(Percentage.from_decimal(decimal_value), alpha, method)
    
    @classmethod
    def from_percent(cls, percent_value: Union[float, str, Decimal], alpha: Union[int, AlphaLevel], method: str = "unknown") -> CVaRValue:
        """Create CVaRValue from percentage form (-25 for -25%)."""
        return cls(Percentage.from_percent(percent_value), alpha, method)
    
    @property
    def is_loss(self) -> bool:
        """Check if CVaR represents a loss (negative value)."""
        return self.value.is_negative
    
    @property
    def is_gain(self) -> bool:
        """Check if CVaR represents a gain (positive value)."""
        return self.value.is_positive
    
    @property
    def loss_magnitude(self) -> Percentage:
        """Get loss magnitude as positive percentage."""
        return abs(self.value)
    
    def to_decimal(self) -> Decimal:
        """Get CVaR as decimal (-0.25 for -25%)."""
        return self.value.to_decimal()
    
    def to_percent(self) -> Decimal:
        """Get CVaR as percentage (-25.0 for -25%)."""
        return self.value.to_percent()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": float(self.value.to_decimal()),
            "alpha": self.alpha.value,
            "method": self.method,
            "percentage": float(self.value.to_percent())
        }
    
    def format(self, precision: int = 2, show_method: bool = False) -> str:
        """
        Format CVaR for display.
        
        Args:
            precision: Decimal precision
            show_method: Include calculation method
            
        Returns:
            Formatted string
        """
        formatted = self.value.format(precision, include_sign=True)
        if show_method and self.method != "unknown":
            return f"{formatted} ({self.method.upper()})"
        return formatted
    
    def is_worse_than(self, other: CVaRValue) -> bool:
        """
        Check if this CVaR is worse (higher loss) than another.
        
        Args:
            other: CVaRValue to compare against
            
        Returns:
            True if this CVaR represents higher loss
        """
        if not isinstance(other, CVaRValue):
            raise TypeError(f"Cannot compare CVaRValue with {type(other)}")
        
        # For CVaR (losses), more negative is worse
        return self.value < other.value
    
    def is_better_than(self, other: CVaRValue) -> bool:
        """Check if this CVaR is better (lower loss) than another."""
        return not self.is_worse_than(other) and self != other
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, CVaRValue):
            return False
        return (self.value == other.value and 
                self.alpha == other.alpha and 
                self.method == other.method)
    
    def __lt__(self, other: CVaRValue) -> bool:
        """Less than (worse CVaR) comparison."""
        if not isinstance(other, CVaRValue):
            raise TypeError(f"Cannot compare CVaRValue with {type(other)}")
        return self.value < other.value
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value} (CVaR{self.alpha.value}%)"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"CVaRValue(value={self.value}, alpha={self.alpha.value}, method='{self.method}')"
    
    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        return hash((self.value, self.alpha, self.method))


@dataclass(frozen=True)
class CVaRTriple:
    """
    Triple of CVaR values from different calculation methods.
    
    Represents NIG-GARCH, GHST-GARCH, and EVaR calculations
    for the same instrument and alpha level.
    """
    
    nig: Optional[CVaRValue]
    ghst: Optional[CVaRValue] 
    evar: Optional[CVaRValue]
    alpha: AlphaLevel
    
    def __init__(
        self,
        nig: Optional[Union[float, Decimal, Percentage, CVaRValue]] = None,
        ghst: Optional[Union[float, Decimal, Percentage, CVaRValue]] = None,
        evar: Optional[Union[float, Decimal, Percentage, CVaRValue]] = None,
        alpha: Union[int, AlphaLevel] = AlphaLevel.ALPHA_99
    ):
        """
        Create CVaRTriple with validation.
        
        Args:
            nig: NIG-GARCH CVaR value
            ghst: GHST-GARCH CVaR value
            evar: Empirical CVaR value
            alpha: Alpha level (must be same for all)
        """
        # Validate alpha level
        if isinstance(alpha, int):
            alpha_level = AlphaLevel.from_int(alpha)
        elif isinstance(alpha, AlphaLevel):
            alpha_level = alpha
        else:
            raise DataValidationError(f"Invalid alpha type: {type(alpha)}")
        
        # Convert values to CVaRValue objects
        def convert_value(value: Optional[Union[float, Decimal, Percentage, CVaRValue]], method: str) -> Optional[CVaRValue]:
            if value is None:
                return None
            if isinstance(value, CVaRValue):
                # Validate alpha matches
                if value.alpha != alpha_level:
                    raise DataValidationError(f"CVaRValue alpha {value.alpha.value} does not match triple alpha {alpha_level.value}")
                return value
            else:
                return CVaRValue(value, alpha_level, method)
        
        nig_cvar = convert_value(nig, "nig")
        ghst_cvar = convert_value(ghst, "ghst") 
        evar_cvar = convert_value(evar, "evar")
        
        # Validate at least one value is present
        if all(v is None for v in [nig_cvar, ghst_cvar, evar_cvar]):
            raise DataValidationError("CVaRTriple must have at least one non-None value")
        
        object.__setattr__(self, 'nig', nig_cvar)
        object.__setattr__(self, 'ghst', ghst_cvar)
        object.__setattr__(self, 'evar', evar_cvar)
        object.__setattr__(self, 'alpha', alpha_level)
    
    @property
    def worst(self) -> Optional[CVaRValue]:
        """Get worst (highest loss) CVaR from available values."""
        available = [v for v in [self.nig, self.ghst, self.evar] if v is not None]
        if not available:
            return None
        return min(available)  # Most negative (worst) CVaR
    
    @property
    def best(self) -> Optional[CVaRValue]:
        """Get best (lowest loss) CVaR from available values."""
        available = [v for v in [self.nig, self.ghst, self.evar] if v is not None]
        if not available:
            return None
        return max(available)  # Least negative (best) CVaR
    
    @property
    def available_methods(self) -> list[str]:
        """Get list of available calculation methods."""
        methods = []
        if self.nig is not None:
            methods.append("nig")
        if self.ghst is not None:
            methods.append("ghst")
        if self.evar is not None:
            methods.append("evar")
        return methods
    
    @property
    def is_complete(self) -> bool:
        """Check if all three methods have values."""
        return all(v is not None for v in [self.nig, self.ghst, self.evar])
    
    def get_method(self, method: str) -> Optional[CVaRValue]:
        """Get CVaR value for specific method."""
        method = method.lower().strip()
        if method == "nig":
            return self.nig
        elif method == "ghst":
            return self.ghst
        elif method == "evar":
            return self.evar
        else:
            raise DataValidationError(f"Invalid method: {method}. Must be one of: nig, ghst, evar")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nig": self.nig.to_dict() if self.nig else None,
            "ghst": self.ghst.to_dict() if self.ghst else None,
            "evar": self.evar.to_dict() if self.evar else None,
            "alpha": self.alpha.value,
            "worst": self.worst.to_dict() if self.worst else None,
            "available_methods": self.available_methods
        }
    
    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.nig:
            parts.append(f"NIG: {self.nig.value}")
        if self.ghst:
            parts.append(f"GHST: {self.ghst.value}")
        if self.evar:
            parts.append(f"EVaR: {self.evar.value}")
        return f"CVaRTriple({', '.join(parts)}) @ {self.alpha.value}%"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"CVaRTriple(nig={self.nig}, ghst={self.ghst}, evar={self.evar}, alpha={self.alpha.value})"


# Utility functions for risk metrics
def worst_cvar(cvar_values: list[CVaRValue]) -> Optional[CVaRValue]:
    """
    Find worst (highest loss) CVaR from list.
    
    Args:
        cvar_values: List of CVaRValue objects
        
    Returns:
        CVaRValue with highest loss, or None if empty list
    """
    if not cvar_values:
        return None
    
    return min(cvar_values)  # Most negative is worst


def best_cvar(cvar_values: list[CVaRValue]) -> Optional[CVaRValue]:
    """
    Find best (lowest loss) CVaR from list.
    
    Args:
        cvar_values: List of CVaRValue objects
        
    Returns:
        CVaRValue with lowest loss, or None if empty list
    """
    if not cvar_values:
        return None
    
    return max(cvar_values)  # Least negative is best


def average_cvar(cvar_values: list[CVaRValue]) -> Optional[CVaRValue]:
    """
    Calculate average CVaR from list (same alpha level required).
    
    Args:
        cvar_values: List of CVaRValue objects with same alpha
        
    Returns:
        Average CVaRValue, or None if empty list
    """
    if not cvar_values:
        return None
    
    # Validate all have same alpha level
    first_alpha = cvar_values[0].alpha
    for cvar in cvar_values[1:]:
        if cvar.alpha != first_alpha:
            raise DataValidationError("All CVaR values must have same alpha level for averaging")
    
    # Calculate average percentage
    total_percentage = Percentage.zero()
    for cvar in cvar_values:
        total_percentage = total_percentage + cvar.value
    
    average_percentage = total_percentage / len(cvar_values)
    
    return CVaRValue(average_percentage, first_alpha, "average")
