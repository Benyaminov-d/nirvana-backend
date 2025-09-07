"""
Financial Instrument domain model.

Pure business entity representing a financial instrument
with type-safe value objects and business rules.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import date, datetime
from uuid import UUID, uuid4

from domain.value_objects.symbol import Symbol, ISIN
from domain.value_objects.instrument import InstrumentClassification, InstrumentType, Country, Exchange
from domain.value_objects.money import Money, Currency
from domain.value_objects.date_range import DateRange
from shared.exceptions import DataValidationError, BusinessLogicError


@dataclass
class FinancialInstrument:
    """
    Domain model for financial instruments.
    
    Represents a tradeable financial security with business rules
    and validations independent of database schema.
    """
    
    # Core identity
    id: UUID
    symbol: Symbol
    name: str
    classification: InstrumentClassification
    
    # Optional identifiers
    isin: Optional[ISIN] = None
    alternative_symbols: List[Symbol] = None
    
    # Market data
    currency: Optional[Currency] = None
    market_cap: Optional[Money] = None
    
    # Data quality flags
    is_active: bool = True
    has_sufficient_history: bool = True
    data_quality_score: float = 1.0
    
    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Generate UUID if not provided
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())
        
        # Initialize empty lists if None
        if self.alternative_symbols is None:
            object.__setattr__(self, 'alternative_symbols', [])
        
        # Set timestamps if not provided
        now = datetime.utcnow()
        if self.created_at is None:
            object.__setattr__(self, 'created_at', now)
        if self.updated_at is None:
            object.__setattr__(self, 'updated_at', now)
        
        # Validate business rules
        self._validate()
    
    def _validate(self):
        """Validate business rules."""
        # Name validation
        if not self.name or not self.name.strip():
            raise DataValidationError("Instrument name cannot be empty")
        
        if len(self.name) > 500:
            raise DataValidationError("Instrument name too long (max 500 characters)")
        
        # Currency consistency
        if self.currency and self.classification.country:
            expected_currency = Currency.from_string(self.classification.currency)
            if self.currency != expected_currency:
                # Allow but warn about currency mismatches
                pass  # Could log warning here
        
        # Market cap validation
        if self.market_cap and self.currency:
            if self.market_cap.currency != self.currency:
                raise DataValidationError(
                    f"Market cap currency {self.market_cap.currency.value} "
                    f"does not match instrument currency {self.currency.value}"
                )
        
        # Data quality score bounds
        if not 0.0 <= self.data_quality_score <= 1.0:
            raise DataValidationError(
                f"Data quality score must be between 0.0 and 1.0: {self.data_quality_score}"
            )
        
        # Alternative symbols validation
        for alt_symbol in self.alternative_symbols:
            if not isinstance(alt_symbol, Symbol):
                raise DataValidationError("Alternative symbols must be Symbol objects")
            if alt_symbol == self.symbol:
                raise DataValidationError("Alternative symbol cannot be the same as primary symbol")
    
    @classmethod
    def create(
        cls,
        symbol: str,
        name: str,
        instrument_type: str,
        country_code: Optional[str] = None,
        exchange_code: Optional[str] = None,
        isin: Optional[str] = None,
        currency: Optional[str] = None,
        **kwargs
    ) -> FinancialInstrument:
        """
        Factory method to create FinancialInstrument from basic parameters.
        
        Args:
            symbol: Primary symbol string
            name: Instrument name
            instrument_type: Type of instrument
            country_code: Country code
            exchange_code: Exchange code
            isin: ISIN string
            currency: Currency code
            **kwargs: Additional parameters
            
        Returns:
            FinancialInstrument instance
        """
        # Create value objects
        symbol_obj = Symbol(symbol)
        classification = InstrumentClassification.from_strings(
            instrument_type, country_code, exchange_code
        )
        
        isin_obj = None
        if isin:
            isin_obj = ISIN(isin)
        
        currency_obj = None
        if currency:
            currency_obj = Currency.from_string(currency)
        elif classification.currency:
            currency_obj = Currency.from_string(classification.currency)
        
        return cls(
            id=uuid4(),
            symbol=symbol_obj,
            name=name.strip(),
            classification=classification,
            isin=isin_obj,
            currency=currency_obj,
            **kwargs
        )
    
    @property
    def is_equity(self) -> bool:
        """Check if instrument is equity-based."""
        return self.classification.instrument_type.is_equity
    
    @property
    def is_fund(self) -> bool:
        """Check if instrument is fund-based."""
        return self.classification.instrument_type.is_fund
    
    @property
    def is_derivative(self) -> bool:
        """Check if instrument is a derivative."""
        return self.classification.instrument_type.is_derivative
    
    @property
    def supports_cvar_analysis(self) -> bool:
        """Check if instrument supports CVaR risk analysis."""
        return (
            self.classification.supports_cvar and
            self.is_active and
            self.has_sufficient_history and
            self.data_quality_score >= 0.5  # Minimum quality threshold
        )
    
    @property
    def primary_market(self) -> str:
        """Get primary market identifier."""
        if self.classification.exchange:
            return self.classification.exchange.code
        elif self.classification.country:
            return self.classification.country.code
        else:
            return "UNKNOWN"
    
    @property
    def display_name(self) -> str:
        """Get display name for UI."""
        if len(self.name) > 50:
            return f"{self.name[:47]}..."
        return self.name
    
    @property
    def full_symbol(self) -> str:
        """Get symbol with exchange suffix if applicable."""
        if self.classification.exchange:
            suffix = self.classification.get_eodhd_suffix()
            if suffix and not self.symbol.value.endswith(suffix):
                return f"{self.symbol.value}{suffix}"
        return self.symbol.value
    
    def add_alternative_symbol(self, alt_symbol: str) -> None:
        """
        Add alternative symbol.
        
        Args:
            alt_symbol: Alternative symbol string
        """
        symbol_obj = Symbol(alt_symbol)
        
        # Check for duplicates
        if symbol_obj in self.alternative_symbols or symbol_obj == self.symbol:
            raise BusinessLogicError(f"Symbol {alt_symbol} already exists for this instrument")
        
        self.alternative_symbols.append(symbol_obj)
        self._touch()
    
    def remove_alternative_symbol(self, alt_symbol: str) -> bool:
        """
        Remove alternative symbol.
        
        Args:
            alt_symbol: Alternative symbol to remove
            
        Returns:
            True if symbol was removed, False if not found
        """
        symbol_obj = Symbol(alt_symbol)
        
        if symbol_obj in self.alternative_symbols:
            self.alternative_symbols.remove(symbol_obj)
            self._touch()
            return True
        return False
    
    def has_symbol(self, symbol: str) -> bool:
        """
        Check if instrument has given symbol (primary or alternative).
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if instrument has this symbol
        """
        symbol_obj = Symbol(symbol)
        return symbol_obj == self.symbol or symbol_obj in self.alternative_symbols
    
    def update_data_quality(self, quality_score: float, has_sufficient_history: bool = None) -> None:
        """
        Update data quality metrics.
        
        Args:
            quality_score: Quality score between 0.0 and 1.0
            has_sufficient_history: Whether instrument has sufficient history
        """
        if not 0.0 <= quality_score <= 1.0:
            raise DataValidationError(f"Quality score must be between 0.0 and 1.0: {quality_score}")
        
        self.data_quality_score = quality_score
        
        if has_sufficient_history is not None:
            self.has_sufficient_history = has_sufficient_history
        
        self._touch()
    
    def deactivate(self, reason: str = None) -> None:
        """
        Deactivate instrument.
        
        Args:
            reason: Optional reason for deactivation
        """
        self.is_active = False
        self._touch()
        
        # Could log deactivation reason here
        if reason:
            pass  # Log reason
    
    def reactivate(self) -> None:
        """Reactivate instrument."""
        self.is_active = True
        self._touch()
    
    def set_market_cap(self, amount: float, currency: str = None) -> None:
        """
        Set market capitalization.
        
        Args:
            amount: Market cap amount
            currency: Currency code (defaults to instrument currency)
        """
        if currency is None:
            if self.currency is None:
                raise BusinessLogicError("Cannot set market cap without currency information")
            currency_obj = self.currency
        else:
            currency_obj = Currency.from_string(currency)
        
        self.market_cap = Money(amount, currency_obj)
        self._touch()
    
    def get_risk_category(self) -> str:
        """
        Get risk category based on instrument characteristics.
        
        Returns:
            Risk category string
        """
        if self.is_derivative:
            return "HIGH"
        elif self.classification.instrument_type == InstrumentType.CRYPTOCURRENCY:
            return "VERY_HIGH"
        elif self.classification.instrument_type == InstrumentType.COMMODITY:
            return "HIGH"
        elif self.is_fund:
            return "MEDIUM"
        elif self.is_equity:
            if self.classification.country == Country.US:
                return "MEDIUM"
            else:
                return "MEDIUM_HIGH"
        elif self.classification.instrument_type == InstrumentType.BOND:
            return "LOW"
        else:
            return "MEDIUM"
    
    def is_eligible_for_analysis(self, analysis_type: str) -> bool:
        """
        Check if instrument is eligible for specific analysis.
        
        Args:
            analysis_type: Type of analysis ('cvar', 'portfolio', 'comparison')
            
        Returns:
            True if eligible
        """
        if not self.is_active:
            return False
        
        if analysis_type == "cvar":
            return self.supports_cvar_analysis
        elif analysis_type == "portfolio":
            return (
                self.has_sufficient_history and
                self.data_quality_score >= 0.3 and
                not self.is_derivative  # Derivatives require special handling
            )
        elif analysis_type == "comparison":
            return (
                self.has_sufficient_history and
                self.data_quality_score >= 0.5
            )
        else:
            return True
    
    def _touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "symbol": self.symbol.value,
            "name": self.name,
            "classification": self.classification.to_dict(),
            "isin": self.isin.value if self.isin else None,
            "alternative_symbols": [s.value for s in self.alternative_symbols],
            "currency": self.currency.value if self.currency else None,
            "market_cap": {
                "amount": float(self.market_cap.amount),
                "currency": self.market_cap.currency.value
            } if self.market_cap else None,
            "is_active": self.is_active,
            "has_sufficient_history": self.has_sufficient_history,
            "data_quality_score": self.data_quality_score,
            "supports_cvar": self.supports_cvar_analysis,
            "risk_category": self.get_risk_category(),
            "primary_market": self.primary_market,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.symbol.value} - {self.display_name}"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"FinancialInstrument(symbol='{self.symbol.value}', name='{self.name}', type='{self.classification.instrument_type.value}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, FinancialInstrument):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dictionaries."""
        return hash(self.id)
