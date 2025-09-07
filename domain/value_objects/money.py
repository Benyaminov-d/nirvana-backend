"""
Money and Currency value objects.

Represents monetary amounts with proper currency handling,
precision, and arithmetic operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Union, Optional
import math

from shared.exceptions import DataValidationError


class Currency(Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    
    @classmethod
    def from_string(cls, currency_str: str) -> Currency:
        """Create Currency from string, case insensitive."""
        if not currency_str:
            raise DataValidationError("Currency cannot be empty")
        
        try:
            return cls(currency_str.upper())
        except ValueError:
            raise DataValidationError(f"Unsupported currency: {currency_str}")


@dataclass(frozen=True)
class Money:
    """
    Immutable money value object with currency.
    
    Handles monetary amounts with proper precision and currency safety.
    All monetary calculations preserve precision using Decimal arithmetic.
    """
    
    amount: Decimal
    currency: Currency
    
    def __init__(self, amount: Union[float, int, str, Decimal], currency: Union[str, Currency]):
        """
        Create Money instance with proper validation.
        
        Args:
            amount: Monetary amount (converted to Decimal for precision)
            currency: Currency code or Currency enum
        """
        # Handle currency conversion
        if isinstance(currency, str):
            currency_enum = Currency.from_string(currency)
        elif isinstance(currency, Currency):
            currency_enum = currency
        else:
            raise DataValidationError(f"Invalid currency type: {type(currency)}")
        
        # Handle amount conversion with validation
        try:
            if isinstance(amount, Decimal):
                decimal_amount = amount
            else:
                decimal_amount = Decimal(str(amount))
            
            # Check for invalid values
            if not decimal_amount.is_finite():
                raise DataValidationError(f"Amount cannot be infinite or NaN: {amount}")
            
            # Reasonable bounds check
            if abs(decimal_amount) > Decimal('1E15'):  # 1 quadrillion
                raise DataValidationError(f"Amount too large: {decimal_amount}")
            
            # Set precision to 4 decimal places (common for financial calculations)
            decimal_amount = decimal_amount.quantize(
                Decimal('0.0001'), 
                rounding=ROUND_HALF_UP
            )
            
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            raise DataValidationError(f"Invalid amount format: {amount} ({str(e)})")
        
        # Use object.__setattr__ since this is a frozen dataclass
        object.__setattr__(self, 'amount', decimal_amount)
        object.__setattr__(self, 'currency', currency_enum)
    
    @classmethod
    def zero(cls, currency: Union[str, Currency]) -> Money:
        """Create zero money in specified currency."""
        return cls(Decimal('0'), currency)
    
    @classmethod
    def usd(cls, amount: Union[float, int, str, Decimal]) -> Money:
        """Create USD money amount."""
        return cls(amount, Currency.USD)
    
    @classmethod
    def eur(cls, amount: Union[float, int, str, Decimal]) -> Money:
        """Create EUR money amount."""
        return cls(amount, Currency.EUR)
    
    def __add__(self, other: Money) -> Money:
        """Add two Money objects (must have same currency)."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot add Money and {type(other)}")
        
        if self.currency != other.currency:
            raise DataValidationError(
                f"Cannot add different currencies: {self.currency.value} + {other.currency.value}"
            )
        
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: Money) -> Money:
        """Subtract two Money objects (must have same currency)."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot subtract {type(other)} from Money")
        
        if self.currency != other.currency:
            raise DataValidationError(
                f"Cannot subtract different currencies: {self.currency.value} - {other.currency.value}"
            )
        
        return Money(self.amount - other.amount, self.currency)
    
    def __mul__(self, factor: Union[float, int, Decimal]) -> Money:
        """Multiply Money by a numeric factor."""
        try:
            if isinstance(factor, Decimal):
                decimal_factor = factor
            else:
                decimal_factor = Decimal(str(factor))
            
            if not decimal_factor.is_finite():
                raise DataValidationError(f"Multiplication factor cannot be infinite or NaN: {factor}")
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid multiplication factor: {factor} ({str(e)})")
        
        return Money(self.amount * decimal_factor, self.currency)
    
    def __rmul__(self, factor: Union[float, int, Decimal]) -> Money:
        """Right multiplication (factor * Money)."""
        return self.__mul__(factor)
    
    def __truediv__(self, divisor: Union[float, int, Decimal]) -> Money:
        """Divide Money by a numeric divisor."""
        try:
            if isinstance(divisor, Decimal):
                decimal_divisor = divisor
            else:
                decimal_divisor = Decimal(str(divisor))
            
            if decimal_divisor == 0:
                raise DataValidationError("Cannot divide Money by zero")
            
            if not decimal_divisor.is_finite():
                raise DataValidationError(f"Division factor cannot be infinite or NaN: {divisor}")
            
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid division factor: {divisor} ({str(e)})")
        
        return Money(self.amount / decimal_divisor, self.currency)
    
    def __neg__(self) -> Money:
        """Negate Money amount."""
        return Money(-self.amount, self.currency)
    
    def __abs__(self) -> Money:
        """Absolute value of Money amount."""
        return Money(abs(self.amount), self.currency)
    
    def __eq__(self, other) -> bool:
        """Compare Money objects for equality."""
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other: Money) -> bool:
        """Compare Money objects (must have same currency)."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money and {type(other)}")
        
        if self.currency != other.currency:
            raise DataValidationError(
                f"Cannot compare different currencies: {self.currency.value} vs {other.currency.value}"
            )
        
        return self.amount < other.amount
    
    def __le__(self, other: Money) -> bool:
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other: Money) -> bool:
        """Greater than comparison."""
        return not self <= other
    
    def __ge__(self, other: Money) -> bool:
        """Greater than or equal comparison."""
        return not self < other
    
    def __str__(self) -> str:
        """String representation for display."""
        return f"{self.amount} {self.currency.value}"
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Money(amount={self.amount}, currency={self.currency.value})"
    
    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        return hash((self.amount, self.currency))
    
    @property
    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self.amount == 0
    
    @property
    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self.amount > 0
    
    @property
    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self.amount < 0
    
    def to_float(self) -> float:
        """Convert to float (use with caution due to precision loss)."""
        return float(self.amount)
    
    def to_string(self, precision: int = 2) -> str:
        """Format as string with specified decimal precision."""
        format_str = f"{{:.{precision}f}}"
        return format_str.format(self.amount)
    
    def format_currency(self, symbol: bool = True, precision: int = 2) -> str:
        """
        Format with currency symbol/code.
        
        Args:
            symbol: Use currency symbol if True, code if False
            precision: Number of decimal places
            
        Returns:
            Formatted string like "$1,234.56" or "1,234.56 USD"
        """
        # Currency symbols
        symbols = {
            Currency.USD: "$",
            Currency.EUR: "€", 
            Currency.GBP: "£",
            Currency.JPY: "¥",
            Currency.CAD: "C$",
            Currency.AUD: "A$",
            Currency.CHF: "CHF ",
            Currency.CNY: "¥"
        }
        
        # Format the number with commas
        format_str = f"{{:,.{precision}f}}"
        amount_str = format_str.format(self.amount)
        
        if symbol and self.currency in symbols:
            if self.currency in [Currency.USD, Currency.CAD, Currency.AUD]:
                return f"{symbols[self.currency]}{amount_str}"
            else:
                return f"{amount_str} {symbols[self.currency]}"
        else:
            return f"{amount_str} {self.currency.value}"
    
    def round_to_cents(self) -> Money:
        """Round to nearest cent (2 decimal places)."""
        rounded_amount = self.amount.quantize(
            Decimal('0.01'), 
            rounding=ROUND_HALF_UP
        )
        return Money(rounded_amount, self.currency)
    
    def allocate(self, ratios: list[float]) -> list[Money]:
        """
        Allocate Money amount across multiple ratios proportionally.
        
        Handles rounding to ensure total allocation equals original amount.
        
        Args:
            ratios: List of proportions (should sum to 1.0)
            
        Returns:
            List of Money objects allocated proportionally
        """
        if not ratios:
            raise DataValidationError("Ratios list cannot be empty")
        
        total_ratio = sum(ratios)
        if abs(total_ratio - 1.0) > 0.001:  # Allow small floating point errors
            raise DataValidationError(f"Ratios must sum to 1.0, got: {total_ratio}")
        
        # Calculate allocations
        allocations = []
        remaining = self.amount
        
        for i, ratio in enumerate(ratios[:-1]):  # All but last
            allocation_amount = (self.amount * Decimal(str(ratio))).quantize(
                Decimal('0.01'), 
                rounding=ROUND_HALF_UP
            )
            allocations.append(Money(allocation_amount, self.currency))
            remaining -= allocation_amount
        
        # Last allocation gets remainder to handle rounding
        allocations.append(Money(remaining, self.currency))
        
        return allocations


# Utility functions for Money operations
def sum_money(money_list: list[Money]) -> Optional[Money]:
    """
    Sum a list of Money objects (must all have same currency).
    
    Args:
        money_list: List of Money objects
        
    Returns:
        Sum as Money object, or None if empty list
    """
    if not money_list:
        return None
    
    # Check all currencies are the same
    first_currency = money_list[0].currency
    for money in money_list[1:]:
        if money.currency != first_currency:
            raise DataValidationError(
                f"Cannot sum different currencies: {first_currency.value} and {money.currency.value}"
            )
    
    total = Money.zero(first_currency)
    for money in money_list:
        total = total + money
    
    return total


def max_money(money_list: list[Money]) -> Optional[Money]:
    """Find maximum Money amount (must all have same currency)."""
    if not money_list:
        return None
    
    # Check all currencies are the same
    first_currency = money_list[0].currency
    for money in money_list[1:]:
        if money.currency != first_currency:
            raise DataValidationError(
                f"Cannot compare different currencies: {first_currency.value} and {money.currency.value}"
            )
    
    return max(money_list)


def min_money(money_list: list[Money]) -> Optional[Money]:
    """Find minimum Money amount (must all have same currency)."""
    if not money_list:
        return None
    
    # Check all currencies are the same
    first_currency = money_list[0].currency
    for money in money_list[1:]:
        if money.currency != first_currency:
            raise DataValidationError(
                f"Cannot compare different currencies: {first_currency.value} and {money.currency.value}"
            )
    
    return min(money_list)
