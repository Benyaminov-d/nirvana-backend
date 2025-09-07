"""
Instrument-related value objects.

Defines types for financial instruments, exchanges, and countries
with validation and normalization logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set, Dict, Any

from shared.exceptions import DataValidationError


class InstrumentType(Enum):
    """Standard financial instrument types."""
    COMMON_STOCK = "Common Stock"
    MUTUAL_FUND = "Mutual Fund"
    ETF = "ETF"
    INDEX = "Index"
    BOND = "Bond"
    REIT = "REIT"
    COMMODITY = "Commodity"
    CURRENCY = "Currency"
    CRYPTOCURRENCY = "Cryptocurrency"
    OPTION = "Option"
    FUTURE = "Future"
    WARRANT = "Warrant"
    PREFERRED_STOCK = "Preferred Stock"
    
    @classmethod
    def from_string(cls, type_str: str) -> InstrumentType:
        """Create InstrumentType from string with normalization."""
        if not type_str:
            raise DataValidationError("Instrument type cannot be empty")
        
        type_str = type_str.strip()
        
        # Direct match
        for inst_type in cls:
            if inst_type.value.lower() == type_str.lower():
                return inst_type
        
        # Alias matching
        aliases = {
            "stock": cls.COMMON_STOCK,
            "equity": cls.COMMON_STOCK,
            "share": cls.COMMON_STOCK,
            "fund": cls.MUTUAL_FUND,
            "mutual fund": cls.MUTUAL_FUND,
            "etf": cls.ETF,
            "exchange traded fund": cls.ETF,
            "index": cls.INDEX,
            "bond": cls.BOND,
            "fixed income": cls.BOND,
            "reit": cls.REIT,
            "real estate investment trust": cls.REIT,
            "commodity": cls.COMMODITY,
            "currency": cls.CURRENCY,
            "forex": cls.CURRENCY,
            "crypto": cls.CRYPTOCURRENCY,
            "cryptocurrency": cls.CRYPTOCURRENCY,
            "digital currency": cls.CRYPTOCURRENCY,
            "option": cls.OPTION,
            "future": cls.FUTURE,
            "futures": cls.FUTURE,
            "warrant": cls.WARRANT,
            "preferred": cls.PREFERRED_STOCK,
            "preferred stock": cls.PREFERRED_STOCK
        }
        
        normalized_key = type_str.lower().strip()
        if normalized_key in aliases:
            return aliases[normalized_key]
        
        raise DataValidationError(f"Unknown instrument type: {type_str}")
    
    @property
    def is_equity(self) -> bool:
        """Check if instrument type is equity-based."""
        return self in {
            self.COMMON_STOCK, 
            self.PREFERRED_STOCK,
            self.REIT
        }
    
    @property
    def is_fund(self) -> bool:
        """Check if instrument type is fund-based."""
        return self in {
            self.MUTUAL_FUND,
            self.ETF
        }
    
    @property
    def is_derivative(self) -> bool:
        """Check if instrument type is a derivative."""
        return self in {
            self.OPTION,
            self.FUTURE,
            self.WARRANT
        }
    
    @property
    def supports_cvar_calculation(self) -> bool:
        """Check if instrument type supports CVaR calculation."""
        # Most instruments support CVaR except some derivatives
        return self not in {
            self.OPTION,  # Options have complex payoff structures
            self.WARRANT  # Warrants have complex payoff structures
        }


class Country(Enum):
    """Supported countries with metadata."""
    US = ("US", "United States", "USD")
    CA = ("CA", "Canada", "CAD") 
    GB = ("GB", "United Kingdom", "GBP")
    DE = ("DE", "Germany", "EUR")
    FR = ("FR", "France", "EUR")
    IT = ("IT", "Italy", "EUR")
    ES = ("ES", "Spain", "EUR")
    NL = ("NL", "Netherlands", "EUR")
    JP = ("JP", "Japan", "JPY")
    AU = ("AU", "Australia", "AUD")
    CH = ("CH", "Switzerland", "CHF")
    SE = ("SE", "Sweden", "SEK")
    NO = ("NO", "Norway", "NOK")
    DK = ("DK", "Denmark", "DKK")
    
    def __init__(self, code: str, country_name: str, currency: str):
        self.code = code
        self.country_name = country_name
        self.currency = currency
    
    @classmethod
    def from_code(cls, country_code: str) -> Country:
        """Create Country from ISO country code."""
        if not country_code:
            raise DataValidationError("Country code cannot be empty")
        
        code_upper = country_code.strip().upper()
        
        for country in cls:
            if country.code == code_upper:
                return country
        
        raise DataValidationError(f"Unsupported country code: {country_code}")
    
    @classmethod
    def from_name(cls, country_name: str) -> Country:
        """Create Country from country name."""
        if not country_name:
            raise DataValidationError("Country name cannot be empty")
        
        name_lower = country_name.strip().lower()
        
        for country in cls:
            if country.country_name.lower() == name_lower:
                return country
        
        raise DataValidationError(f"Unsupported country name: {country_name}")
    
    @property
    def is_eu(self) -> bool:
        """Check if country is in European Union."""
        eu_countries = {self.DE, self.FR, self.IT, self.ES, self.NL}
        return self in eu_countries
    
    @property
    def is_european(self) -> bool:
        """Check if country is in Europe."""
        european_countries = {
            self.GB, self.DE, self.FR, self.IT, self.ES, 
            self.NL, self.CH, self.SE, self.NO, self.DK
        }
        return self in european_countries


class Exchange(Enum):
    """Major financial exchanges with metadata."""
    NYSE = ("NYSE", "New York Stock Exchange", Country.US)
    NASDAQ = ("NASDAQ", "NASDAQ", Country.US)
    AMEX = ("AMEX", "American Stock Exchange", Country.US)
    TSX = ("TSX", "Toronto Stock Exchange", Country.CA)
    TSXV = ("TSXV", "TSX Venture Exchange", Country.CA)
    LSE = ("LSE", "London Stock Exchange", Country.GB)
    XETRA = ("XETRA", "Xetra", Country.DE)
    FSE = ("FSE", "Frankfurt Stock Exchange", Country.DE)
    EPA = ("EPA", "Euronext Paris", Country.FR)
    BIT = ("BIT", "Borsa Italiana", Country.IT)
    AMS = ("AMS", "Euronext Amsterdam", Country.NL)
    TYO = ("TYO", "Tokyo Stock Exchange", Country.JP)
    ASX = ("ASX", "Australian Securities Exchange", Country.AU)
    SIX = ("SIX", "SIX Swiss Exchange", Country.CH)
    
    def __init__(self, code: str, exchange_name: str, country: Country):
        self.code = code
        self.exchange_name = exchange_name
        self.country = country
    
    @classmethod
    def from_code(cls, exchange_code: str) -> Exchange:
        """Create Exchange from exchange code."""
        if not exchange_code:
            raise DataValidationError("Exchange code cannot be empty")
        
        code_upper = exchange_code.strip().upper()
        
        for exchange in cls:
            if exchange.code == code_upper:
                return exchange
        
        raise DataValidationError(f"Unsupported exchange code: {exchange_code}")
    
    @classmethod
    def for_country(cls, country: Country) -> list[Exchange]:
        """Get all exchanges for a country."""
        return [exchange for exchange in cls if exchange.country == country]
    
    @property
    def eodhd_suffix(self) -> str:
        """Get EODHD API suffix for this exchange."""
        suffix_map = {
            self.NYSE: ".US",
            self.NASDAQ: ".US", 
            self.AMEX: ".US",
            self.TSX: ".TO",
            self.TSXV: ".TO",
            self.LSE: ".LSE",
            self.XETRA: ".DE",
            self.FSE: ".DE",
            self.EPA: ".PA",
            self.BIT: ".MI",
            self.AMS: ".AS",
            self.TYO: ".T",
            self.ASX: ".AX",
            self.SIX: ".SW"
        }
        
        return suffix_map.get(self, ".US")  # Default to US
    
    @property
    def is_major(self) -> bool:
        """Check if exchange is considered major."""
        major_exchanges = {
            self.NYSE, self.NASDAQ, self.LSE, 
            self.TYO, self.XETRA, self.EPA, self.ASX
        }
        return self in major_exchanges


@dataclass(frozen=True)
class InstrumentClassification:
    """
    Comprehensive instrument classification.
    
    Combines instrument type, country, and exchange information
    with validation and utility methods.
    """
    
    instrument_type: InstrumentType
    country: Optional[Country]
    exchange: Optional[Exchange]
    
    def __init__(
        self,
        instrument_type: InstrumentType,
        country: Optional[Country] = None,
        exchange: Optional[Exchange] = None
    ):
        """
        Create InstrumentClassification with validation.
        
        Args:
            instrument_type: Type of financial instrument
            country: Country of listing
            exchange: Exchange where traded
        """
        if not isinstance(instrument_type, InstrumentType):
            raise DataValidationError("instrument_type must be InstrumentType enum")
        
        # Validate country/exchange consistency
        if exchange is not None and country is not None:
            if exchange.country != country:
                raise DataValidationError(
                    f"Exchange {exchange.code} is not in country {country.code}"
                )
        
        # If exchange is specified but country is not, infer country
        if exchange is not None and country is None:
            country = exchange.country
        
        object.__setattr__(self, 'instrument_type', instrument_type)
        object.__setattr__(self, 'country', country)
        object.__setattr__(self, 'exchange', exchange)
    
    @classmethod
    def from_strings(
        cls,
        instrument_type: str,
        country_code: Optional[str] = None,
        exchange_code: Optional[str] = None
    ) -> InstrumentClassification:
        """
        Create from string representations.
        
        Args:
            instrument_type: Instrument type string
            country_code: Country code string
            exchange_code: Exchange code string
            
        Returns:
            InstrumentClassification object
        """
        inst_type = InstrumentType.from_string(instrument_type)
        
        country = None
        if country_code:
            country = Country.from_code(country_code)
        
        exchange = None
        if exchange_code:
            exchange = Exchange.from_code(exchange_code)
        
        return cls(inst_type, country, exchange)
    
    @property
    def supports_cvar(self) -> bool:
        """Check if this instrument classification supports CVaR calculation."""
        return self.instrument_type.supports_cvar_calculation
    
    @property
    def is_equity_like(self) -> bool:
        """Check if instrument behaves like equity for analysis."""
        return self.instrument_type.is_equity or self.instrument_type == InstrumentType.ETF
    
    @property
    def is_international(self) -> bool:
        """Check if instrument is international (non-US)."""
        return self.country is not None and self.country != Country.US
    
    @property
    def currency(self) -> Optional[str]:
        """Get likely trading currency."""
        if self.country:
            return self.country.currency
        return None
    
    @property
    def market_tier(self) -> str:
        """Get market tier classification."""
        if self.exchange is None:
            return "unknown"
        
        if self.exchange.is_major:
            return "major"
        elif self.exchange.country in {Country.US, Country.CA, Country.GB, Country.DE, Country.JP}:
            return "developed"
        else:
            return "regional"
    
    def get_eodhd_suffix(self) -> str:
        """Get EODHD API suffix for data retrieval."""
        if self.exchange:
            return self.exchange.eodhd_suffix
        elif self.country:
            # Fallback based on country
            suffix_map = {
                Country.US: ".US",
                Country.CA: ".TO",
                Country.GB: ".LSE",
                Country.DE: ".DE", 
                Country.FR: ".PA",
                Country.JP: ".T",
                Country.AU: ".AX"
            }
            return suffix_map.get(self.country, ".US")
        else:
            return ".US"  # Default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instrument_type": self.instrument_type.value,
            "country": self.country.code if self.country else None,
            "exchange": self.exchange.code if self.exchange else None,
            "currency": self.currency,
            "supports_cvar": self.supports_cvar,
            "is_equity_like": self.is_equity_like,
            "market_tier": self.market_tier,
            "eodhd_suffix": self.get_eodhd_suffix()
        }
    
    def __str__(self) -> str:
        """String representation."""
        parts = [self.instrument_type.value]
        if self.exchange:
            parts.append(f"on {self.exchange.code}")
        elif self.country:
            parts.append(f"in {self.country.code}")
        return " ".join(parts)
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"InstrumentClassification(type={self.instrument_type.value}, country={self.country.code if self.country else None}, exchange={self.exchange.code if self.exchange else None})"


# Utility functions
def classify_instrument(
    instrument_type_str: str,
    country_code: Optional[str] = None,
    exchange_code: Optional[str] = None
) -> InstrumentClassification:
    """
    Create InstrumentClassification from string inputs.
    
    Args:
        instrument_type_str: Instrument type string
        country_code: Country code
        exchange_code: Exchange code
        
    Returns:
        InstrumentClassification object
    """
    return InstrumentClassification.from_strings(
        instrument_type_str, 
        country_code, 
        exchange_code
    )


def get_supported_countries() -> list[Country]:
    """Get list of all supported countries."""
    return list(Country)


def get_supported_exchanges() -> list[Exchange]:
    """Get list of all supported exchanges.""" 
    return list(Exchange)


def get_supported_instrument_types() -> list[InstrumentType]:
    """Get list of all supported instrument types."""
    return list(InstrumentType)


def get_cvar_compatible_types() -> list[InstrumentType]:
    """Get instrument types that support CVaR calculation."""
    return [t for t in InstrumentType if t.supports_cvar_calculation]
