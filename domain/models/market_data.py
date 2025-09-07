"""
Market Data domain models.

Models for price history, quotes, and market data
with type-safe value objects and business validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from uuid import UUID, uuid4

from domain.value_objects.symbol import Symbol
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage
from domain.value_objects.date_range import DateRange
from shared.exceptions import DataValidationError


@dataclass
class PricePoint:
    """Single price observation."""
    date: date
    open_price: Optional[Money]
    high_price: Optional[Money]
    low_price: Optional[Money]
    close_price: Money
    adjusted_close: Optional[Money]
    volume: Optional[int] = None
    
    def __post_init__(self):
        """Validate price point."""
        if self.volume is not None and self.volume < 0:
            raise DataValidationError("Volume cannot be negative")


@dataclass
class PriceHistory:
    """Complete price history for an instrument."""
    symbol: Symbol
    prices: List[PricePoint]
    data_period: DateRange
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate price history."""
        if not self.prices:
            raise DataValidationError("Price history cannot be empty")
        
        # Sort prices by date
        self.prices.sort(key=lambda p: p.date)
    
    @property
    def returns(self) -> List[Percentage]:
        """Calculate returns from price history."""
        if len(self.prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.prices)):
            prev_price = self.prices[i-1].close_price
            curr_price = self.prices[i].close_price
            
            if prev_price.amount > 0:
                return_val = (curr_price.amount - prev_price.amount) / prev_price.amount
                returns.append(Percentage.from_decimal(return_val))
        
        return returns


@dataclass
class MarketQuote:
    """Real-time market quote."""
    symbol: Symbol
    price: Money
    change: Optional[Money] = None
    change_percent: Optional[Percentage] = None
    volume: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
