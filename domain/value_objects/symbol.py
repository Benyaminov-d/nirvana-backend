"""
Symbol and ISIN value objects for financial instruments.

Provides type-safe representation of financial instrument identifiers
with validation and normalization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re

from shared.exceptions import DataValidationError


@dataclass(frozen=True)
class Symbol:
    """
    Immutable financial instrument symbol.
    
    Represents ticker symbols with proper validation and normalization.
    Examples: AAPL, MSFT, BRK.B, AAPL.US
    """
    
    value: str
    
    def __init__(self, symbol: str):
        """
        Create Symbol with validation and normalization.
        
        Args:
            symbol: Raw symbol string
            
        Raises:
            DataValidationError: If symbol format is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise DataValidationError("Symbol cannot be None or empty")
        
        # Normalize: strip whitespace and convert to uppercase
        normalized = symbol.strip().upper()
        
        if not normalized:
            raise DataValidationError("Symbol cannot be empty after normalization")
        
        # Length validation
        if len(normalized) > 32:
            raise DataValidationError(f"Symbol too long: {normalized} (max 32 characters)")
        
        # Format validation: alphanumeric plus dots, dashes, underscores
        if not re.match(r'^[A-Z0-9._-]+$', normalized):
            raise DataValidationError(f"Symbol contains invalid characters: {normalized}")
        
        # Additional format rules
        if normalized.startswith('.') or normalized.endswith('.'):
            raise DataValidationError(f"Symbol cannot start or end with '.': {normalized}")
        
        if '..' in normalized or '--' in normalized:
            raise DataValidationError(f"Symbol cannot contain consecutive separators: {normalized}")
        
        # Common anti-patterns
        if normalized.lower() in ['null', 'none', 'undefined', '']:
            raise DataValidationError(f"Invalid symbol value: {normalized}")
        
        object.__setattr__(self, 'value', normalized)
    
    @property
    def base_symbol(self) -> str:
        """Get base symbol without exchange suffix (AAPL.US -> AAPL)."""
        if '.' in self.value:
            # Split on last dot to handle symbols like BRK.B.US
            parts = self.value.rsplit('.', 1)
            if len(parts) == 2 and len(parts[1]) <= 3 and parts[1].isalpha():
                # Looks like exchange suffix
                return parts[0]
        return self.value
    
    @property
    def exchange_suffix(self) -> Optional[str]:
        """Get exchange suffix if present (.US from AAPL.US)."""
        if '.' in self.value:
            parts = self.value.rsplit('.', 1)
            if len(parts) == 2 and len(parts[1]) <= 3 and parts[1].isalpha():
                return parts[1]
        return None
    
    @property
    def has_exchange_suffix(self) -> bool:
        """Check if symbol has exchange suffix."""
        return self.exchange_suffix is not None
    
    def with_suffix(self, suffix: str) -> Symbol:
        """Create new Symbol with exchange suffix."""
        if not suffix or not isinstance(suffix, str):
            raise DataValidationError("Exchange suffix cannot be empty")
        
        suffix_normalized = suffix.strip().upper()
        
        if not re.match(r'^[A-Z]{1,4}$', suffix_normalized):
            raise DataValidationError(f"Invalid exchange suffix format: {suffix}")
        
        # Remove existing suffix if present
        base = self.base_symbol
        return Symbol(f"{base}.{suffix_normalized}")
    
    def without_suffix(self) -> Symbol:
        """Create new Symbol without exchange suffix."""
        return Symbol(self.base_symbol)
    
    def is_similar_to(self, other: Symbol) -> bool:
        """Check if symbols are similar (same base symbol)."""
        if not isinstance(other, Symbol):
            return False
        return self.base_symbol == other.base_symbol
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"Symbol('{self.value}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, Symbol):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        return hash(self.value)
    
    def __lt__(self, other: Symbol) -> bool:
        """Less than comparison for sorting."""
        if not isinstance(other, Symbol):
            raise TypeError(f"Cannot compare Symbol and {type(other)}")
        return self.value < other.value


@dataclass(frozen=True)
class ISIN:
    """
    Immutable International Securities Identification Number.
    
    12-character alphanumeric code that uniquely identifies securities.
    Format: 2-letter country code + 9-character identifier + 1 check digit
    Example: US0378331005 (Apple Inc.)
    """
    
    value: str
    
    def __init__(self, isin: str):
        """
        Create ISIN with validation.
        
        Args:
            isin: ISIN string
            
        Raises:
            DataValidationError: If ISIN format is invalid
        """
        if not isin or not isinstance(isin, str):
            raise DataValidationError("ISIN cannot be None or empty")
        
        # Normalize: strip whitespace and convert to uppercase
        normalized = isin.strip().upper()
        
        if not normalized:
            raise DataValidationError("ISIN cannot be empty after normalization")
        
        # Length validation
        if len(normalized) != 12:
            raise DataValidationError(f"ISIN must be 12 characters: {normalized} ({len(normalized)} characters)")
        
        # Format validation: alphanumeric only
        if not normalized.isalnum():
            raise DataValidationError(f"ISIN must be alphanumeric: {normalized}")
        
        # Country code validation (first 2 characters must be letters)
        country_code = normalized[:2]
        if not country_code.isalpha():
            raise DataValidationError(f"ISIN country code must be alphabetic: {country_code}")
        
        # Identifier validation (characters 3-11 must be alphanumeric)
        identifier = normalized[2:11]
        if not identifier.isalnum():
            raise DataValidationError(f"ISIN identifier must be alphanumeric: {identifier}")
        
        # Check digit validation (last character must be digit)
        check_digit = normalized[11]
        if not check_digit.isdigit():
            raise DataValidationError(f"ISIN check digit must be numeric: {check_digit}")
        
        # Verify check digit using Luhn algorithm
        if not self._verify_check_digit(normalized):
            raise DataValidationError(f"ISIN has invalid check digit: {normalized}")
        
        object.__setattr__(self, 'value', normalized)
    
    @staticmethod
    def _verify_check_digit(isin: str) -> bool:
        """
        Verify ISIN check digit using modified Luhn algorithm.
        
        Args:
            isin: 12-character ISIN
            
        Returns:
            True if check digit is valid
        """
        # Convert letters to numbers (A=10, B=11, ..., Z=35)
        converted = ""
        for char in isin[:-1]:  # Exclude check digit
            if char.isdigit():
                converted += char
            else:
                converted += str(ord(char) - ord('A') + 10)
        
        # Apply Luhn algorithm
        total = 0
        reverse_digits = converted[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Every second digit
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10  # Sum digits if > 9
            total += n
        
        # Check digit should make total divisible by 10
        calculated_check_digit = (10 - (total % 10)) % 10
        actual_check_digit = int(isin[-1])
        
        return calculated_check_digit == actual_check_digit
    
    @classmethod
    def generate(cls, country_code: str, identifier: str) -> ISIN:
        """
        Generate ISIN with calculated check digit.
        
        Args:
            country_code: 2-letter country code
            identifier: 9-character identifier
            
        Returns:
            Valid ISIN with check digit
        """
        if not country_code or len(country_code) != 2 or not country_code.isalpha():
            raise DataValidationError("Country code must be 2 alphabetic characters")
        
        if not identifier or len(identifier) != 9 or not identifier.isalnum():
            raise DataValidationError("Identifier must be 9 alphanumeric characters")
        
        # Normalize inputs
        country_normalized = country_code.upper()
        identifier_normalized = identifier.upper()
        
        # Calculate check digit
        partial_isin = country_normalized + identifier_normalized
        
        # Convert to digits for Luhn algorithm
        converted = ""
        for char in partial_isin:
            if char.isdigit():
                converted += char
            else:
                converted += str(ord(char) - ord('A') + 10)
        
        # Calculate check digit
        total = 0
        reverse_digits = converted[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Every second digit
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n
        
        check_digit = (10 - (total % 10)) % 10
        
        return cls(partial_isin + str(check_digit))
    
    @property
    def country_code(self) -> str:
        """Get 2-letter country code."""
        return self.value[:2]
    
    @property
    def identifier(self) -> str:
        """Get 9-character identifier."""
        return self.value[2:11]
    
    @property
    def check_digit(self) -> str:
        """Get check digit."""
        return self.value[11]
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"ISIN('{self.value}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, ISIN):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        return hash(self.value)
    
    def __lt__(self, other: ISIN) -> bool:
        """Less than comparison for sorting."""
        if not isinstance(other, ISIN):
            raise TypeError(f"Cannot compare ISIN and {type(other)}")
        return self.value < other.value


# Utility functions
def parse_symbol(symbol_str: str) -> Symbol:
    """
    Parse string to Symbol with validation.
    
    Args:
        symbol_str: Symbol string to parse
        
    Returns:
        Validated Symbol object
    """
    return Symbol(symbol_str)


def parse_isin(isin_str: str) -> ISIN:
    """
    Parse string to ISIN with validation.
    
    Args:
        isin_str: ISIN string to parse
        
    Returns:
        Validated ISIN object
    """
    return ISIN(isin_str)


def symbols_from_string(symbols_str: str, separator: str = ",") -> list[Symbol]:
    """
    Parse comma-separated symbol string to list of Symbol objects.
    
    Args:
        symbols_str: String containing symbols
        separator: Separator character
        
    Returns:
        List of Symbol objects
    """
    if not symbols_str or not isinstance(symbols_str, str):
        return []
    
    symbol_strings = [s.strip() for s in symbols_str.split(separator) if s.strip()]
    return [Symbol(s) for s in symbol_strings]
