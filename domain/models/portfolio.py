"""
Portfolio domain models.

Models for portfolios and positions using type-safe value objects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

from domain.value_objects.symbol import Symbol
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage
from shared.exceptions import BusinessLogicError


@dataclass
class Position:
    """Portfolio position in a financial instrument."""
    symbol: Symbol
    quantity: int
    average_cost: Money
    current_value: Optional[Money] = None
    
    @property
    def market_value(self) -> Optional[Money]:
        """Get current market value."""
        return self.current_value
    
    @property
    def total_cost(self) -> Money:
        """Get total cost basis."""
        return self.average_cost * abs(self.quantity)
    
    @property
    def unrealized_pnl(self) -> Optional[Money]:
        """Calculate unrealized P&L."""
        if not self.current_value:
            return None
        
        market_value = self.current_value * abs(self.quantity)
        return market_value - self.total_cost


@dataclass  
class Portfolio:
    """Investment portfolio containing multiple positions."""
    id: UUID
    name: str
    positions: List[Position] = field(default_factory=list)
    base_currency: str = "USD"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate portfolio."""
        if not self.name.strip():
            raise BusinessLogicError("Portfolio name cannot be empty")
    
    def add_position(self, position: Position) -> None:
        """Add position to portfolio."""
        # Check for duplicate symbols
        if any(p.symbol == position.symbol for p in self.positions):
            raise BusinessLogicError(f"Position for {position.symbol.value} already exists")
        
        self.positions.append(position)
    
    @property
    def total_positions(self) -> int:
        """Get number of positions."""
        return len(self.positions)
    
    @property
    def symbols(self) -> List[Symbol]:
        """Get all symbols in portfolio."""
        return [p.symbol for p in self.positions]
