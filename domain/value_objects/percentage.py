"""
Percentage value object for financial calculations.

Handles percentage values with proper precision, validation,
and conversion between different representations (0.15 vs 15%).
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union
import math

from shared.exceptions import DataValidationError


@dataclass(frozen=True)  
class Percentage:
    """
    Immutable percentage value object.
    
    Stores percentage as decimal (0.15 for 15%) for precision in calculations.
    Provides methods for display formatting and validation.
    """
    
    value: Decimal  # Stored as decimal (0.15 for 15%)
    
    def __init__(self, value: Union[float, int, str, Decimal], as_decimal: bool = True):
        """
        Create Percentage instance.
        
        Args:
            value: Percentage value
            as_decimal: If True, value is 0.15 for 15%. If False, value is 15 for 15%
        """
        try:
            if isinstance(value, Decimal):
                decimal_value = value
            else:
                decimal_value = Decimal(str(value))
            
            # Check for invalid values
            if not decimal_value.is_finite():
                raise DataValidationError(f"Percentage cannot be infinite or NaN: {value}")
            
            # Convert percentage form (15) to decimal form (0.15) if needed
            if not as_decimal:
                decimal_value = decimal_value / Decimal('100')
            
            # Reasonable bounds check (-10.0 to 10.0, i.e., -1000% to 1000%)
            if decimal_value < Decimal('-10.0') or decimal_value > Decimal('10.0'):
                raise DataValidationError(f"Percentage outside reasonable range: {decimal_value}")
            
            # Set precision to 6 decimal places for accurate percentage calculations
            decimal_value = decimal_value.quantize(
                Decimal('0.000001'), 
                rounding=ROUND_HALF_UP
            )
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid percentage format: {value} ({str(e)})")
        
        # Use object.__setattr__ since this is a frozen dataclass
        object.__setattr__(self, 'value', decimal_value)
    
    @classmethod
    def zero(cls) -> Percentage:
        """Create zero percentage."""
        return cls(Decimal('0'))
    
    @classmethod
    def from_decimal(cls, decimal_value: Union[float, int, str, Decimal]) -> Percentage:
        """Create from decimal form (0.15 for 15%)."""
        return cls(decimal_value, as_decimal=True)
    
    @classmethod
    def from_percent(cls, percent_value: Union[float, int, str, Decimal]) -> Percentage:
        """Create from percentage form (15 for 15%)."""
        return cls(percent_value, as_decimal=False)
    
    @classmethod
    def from_basis_points(cls, bp: Union[int, str, Decimal]) -> Percentage:
        """Create from basis points (150 bp = 1.5%)."""
        try:
            bp_decimal = Decimal(str(bp))
            return cls(bp_decimal / Decimal('10000'), as_decimal=True)
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid basis points: {bp} ({str(e)})")
    
    def __add__(self, other: Percentage) -> Percentage:
        """Add two percentages."""
        if not isinstance(other, Percentage):
            raise TypeError(f"Cannot add Percentage and {type(other)}")
        
        return Percentage(self.value + other.value, as_decimal=True)
    
    def __sub__(self, other: Percentage) -> Percentage:
        """Subtract two percentages."""
        if not isinstance(other, Percentage):
            raise TypeError(f"Cannot subtract {type(other)} from Percentage")
        
        return Percentage(self.value - other.value, as_decimal=True)
    
    def __mul__(self, factor: Union[float, int, Decimal, Percentage]) -> Percentage:
        """Multiply percentage by a factor or another percentage."""
        if isinstance(factor, Percentage):
            # Multiplying percentages: 20% * 50% = 10%
            return Percentage(self.value * factor.value, as_decimal=True)
        
        try:
            if isinstance(factor, Decimal):
                decimal_factor = factor
            else:
                decimal_factor = Decimal(str(factor))
            
            if not decimal_factor.is_finite():
                raise DataValidationError(f"Multiplication factor cannot be infinite or NaN: {factor}")
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid multiplication factor: {factor} ({str(e)})")
        
        return Percentage(self.value * decimal_factor, as_decimal=True)
    
    def __rmul__(self, factor: Union[float, int, Decimal]) -> Percentage:
        """Right multiplication (factor * Percentage)."""
        return self.__mul__(factor)
    
    def __truediv__(self, divisor: Union[float, int, Decimal]) -> Percentage:
        """Divide percentage by a numeric divisor."""
        try:
            if isinstance(divisor, Decimal):
                decimal_divisor = divisor
            else:
                decimal_divisor = Decimal(str(divisor))
            
            if decimal_divisor == 0:
                raise DataValidationError("Cannot divide Percentage by zero")
            
            if not decimal_divisor.is_finite():
                raise DataValidationError(f"Division factor cannot be infinite or NaN: {divisor}")
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid division factor: {divisor} ({str(e)})")
        
        return Percentage(self.value / decimal_divisor, as_decimal=True)
    
    def __neg__(self) -> Percentage:
        """Negate percentage."""
        return Percentage(-self.value, as_decimal=True)
    
    def __abs__(self) -> Percentage:
        """Absolute value of percentage."""
        return Percentage(abs(self.value), as_decimal=True)
    
    def __eq__(self, other) -> bool:
        """Compare percentages for equality."""
        if not isinstance(other, Percentage):
            return False
        return self.value == other.value
    
    def __lt__(self, other: Percentage) -> bool:
        """Compare percentages."""
        if not isinstance(other, Percentage):
            raise TypeError(f"Cannot compare Percentage and {type(other)}")
        
        return self.value < other.value
    
    def __le__(self, other: Percentage) -> bool:
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other: Percentage) -> bool:
        """Greater than comparison."""
        return not self <= other
    
    def __ge__(self, other: Percentage) -> bool:
        """Greater than or equal comparison."""
        return not self < other
    
    def __str__(self) -> str:
        """String representation for display (as percentage)."""
        return f"{self.to_percent():.2f}%"
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Percentage({self.value})"
    
    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        return hash(self.value)
    
    @property
    def is_zero(self) -> bool:
        """Check if percentage is zero."""
        return self.value == 0
    
    @property
    def is_positive(self) -> bool:
        """Check if percentage is positive."""
        return self.value > 0
    
    @property
    def is_negative(self) -> bool:
        """Check if percentage is negative."""
        return self.value < 0
    
    def to_decimal(self) -> Decimal:
        """Get decimal representation (0.15 for 15%)."""
        return self.value
    
    def to_float(self) -> float:
        """Convert to float (use with caution due to precision loss)."""
        return float(self.value)
    
    def to_percent(self) -> Decimal:
        """Get percentage representation (15.0 for 15%)."""
        return self.value * Decimal('100')
    
    def to_basis_points(self) -> Decimal:
        """Get basis points representation (150 for 1.5%)."""
        return self.value * Decimal('10000')
    
    def format(self, precision: int = 2, include_sign: bool = False) -> str:
        """
        Format percentage as string.
        
        Args:
            precision: Number of decimal places
            include_sign: Include + sign for positive values
            
        Returns:
            Formatted percentage string
        """
        percent_value = self.to_percent()
        
        format_str = f"{{:{'+' if include_sign else ''}.{precision}f}}%"
        return format_str.format(percent_value)
    
    def format_basis_points(self, precision: int = 0) -> str:
        """Format as basis points."""
        bp_value = self.to_basis_points()
        format_str = f"{{:.{precision}f}} bp"
        return format_str.format(bp_value)
    
    def round_to_percent(self) -> Percentage:
        """Round to nearest whole percent."""
        rounded_percent = (self.value * Decimal('100')).quantize(
            Decimal('1'), 
            rounding=ROUND_HALF_UP
        )
        return Percentage(rounded_percent / Decimal('100'), as_decimal=True)
    
    def round_to_basis_point(self) -> Percentage:
        """Round to nearest basis point."""
        rounded_bp = (self.value * Decimal('10000')).quantize(
            Decimal('1'), 
            rounding=ROUND_HALF_UP
        )
        return Percentage(rounded_bp / Decimal('10000'), as_decimal=True)
    
    def apply_to(self, value: Union[Decimal, float, int]) -> Decimal:
        """
        Apply percentage to a value.
        
        Args:
            value: Value to apply percentage to
            
        Returns:
            Result of value * percentage
        """
        try:
            if isinstance(value, Decimal):
                decimal_value = value
            else:
                decimal_value = Decimal(str(value))
            
            return decimal_value * self.value
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Cannot apply percentage to invalid value: {value} ({str(e)})")
    
    def of(self, value: Union[Decimal, float, int]) -> Decimal:
        """Alias for apply_to() - more natural reading: percentage.of(100)."""
        return self.apply_to(value)
    
    def compound(self, periods: int) -> Percentage:
        """
        Calculate compound percentage over multiple periods.
        
        Args:
            periods: Number of compounding periods
            
        Returns:
            Compounded percentage
        """
        if periods < 0:
            raise DataValidationError("Compounding periods cannot be negative")
        
        if periods == 0:
            return Percentage.zero()
        
        # (1 + r)^n - 1
        compound_factor = (Decimal('1') + self.value) ** periods
        compound_percentage = compound_factor - Decimal('1')
        
        return Percentage(compound_percentage, as_decimal=True)
    
    def annualize(self, periods_per_year: Union[int, Decimal]) -> Percentage:
        """
        Annualize percentage return.
        
        Args:
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly, etc.)
            
        Returns:
            Annualized percentage
        """
        try:
            periods_decimal = Decimal(str(periods_per_year))
            
            if periods_decimal <= 0:
                raise DataValidationError("Periods per year must be positive")
            
            # (1 + r)^periods_per_year - 1
            annualized_factor = (Decimal('1') + self.value) ** periods_decimal
            annualized_percentage = annualized_factor - Decimal('1')
            
            return Percentage(annualized_percentage, as_decimal=True)
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid periods per year: {periods_per_year} ({str(e)})")


# Utility functions for Percentage operations
def average_percentage(percentages: list[Percentage]) -> Percentage:
    """Calculate average of percentages."""
    if not percentages:
        raise DataValidationError("Cannot calculate average of empty percentage list")
    
    total = Percentage.zero()
    for pct in percentages:
        total = total + pct
    
    return total / len(percentages)


def weighted_average_percentage(
    percentages: list[Percentage], 
    weights: list[Union[float, int, Decimal]]
) -> Percentage:
    """Calculate weighted average of percentages."""
    if not percentages:
        raise DataValidationError("Cannot calculate weighted average of empty percentage list")
    
    if len(percentages) != len(weights):
        raise DataValidationError("Number of percentages must match number of weights")
    
    total_weighted = Decimal('0')
    total_weight = Decimal('0')
    
    for pct, weight in zip(percentages, weights):
        weight_decimal = Decimal(str(weight))
        total_weighted += pct.value * weight_decimal
        total_weight += weight_decimal
    
    if total_weight == 0:
        raise DataValidationError("Total weight cannot be zero")
    
    return Percentage(total_weighted / total_weight, as_decimal=True)


def max_percentage(percentages: list[Percentage]) -> Percentage:
    """Find maximum percentage in list."""
    if not percentages:
        raise DataValidationError("Cannot find max of empty percentage list")
    
    return max(percentages)


def min_percentage(percentages: list[Percentage]) -> Percentage:
    """Find minimum percentage in list."""
    if not percentages:
        raise DataValidationError("Cannot find min of empty percentage list")
    
    return min(percentages)
