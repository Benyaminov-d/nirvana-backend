"""
Date range value object for time-based calculations.

Represents periods of time with validation and utility methods
for financial analysis and data processing.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Union, Optional
import calendar

from shared.exceptions import DataValidationError


@dataclass(frozen=True)
class DateRange:
    """
    Immutable date range value object.
    
    Represents a period between two dates with validation
    and utility methods for financial calculations.
    """
    
    start_date: date
    end_date: date
    
    def __init__(
        self, 
        start: Union[str, date, datetime], 
        end: Union[str, date, datetime]
    ):
        """
        Create DateRange with validation.
        
        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
        """
        # Convert inputs to date objects
        start_date = self._to_date(start, "start")
        end_date = self._to_date(end, "end")
        
        # Validate range
        if start_date > end_date:
            raise DataValidationError(f"Start date {start_date} cannot be after end date {end_date}")
        
        # Reasonable bounds check
        min_date = date(1900, 1, 1)
        max_date = date(2100, 12, 31)
        
        if start_date < min_date or end_date > max_date:
            raise DataValidationError(f"Date range {start_date} to {end_date} outside reasonable bounds ({min_date} to {max_date})")
        
        object.__setattr__(self, 'start_date', start_date)
        object.__setattr__(self, 'end_date', end_date)
    
    @staticmethod
    def _to_date(value: Union[str, date, datetime], field_name: str) -> date:
        """Convert various input types to date."""
        if isinstance(value, date):
            return value
        elif isinstance(value, datetime):
            return value.date()
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                raise DataValidationError(f"{field_name} date cannot be empty")
            
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
            
            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00')).date()
            except ValueError:
                pass
                
            raise DataValidationError(f"Invalid {field_name} date format: {value}")
        else:
            raise DataValidationError(f"{field_name} must be date, datetime, or string")
    
    @classmethod
    def from_days(cls, end_date: Union[str, date, datetime], days: int) -> DateRange:
        """
        Create DateRange by going back specified days from end date.
        
        Args:
            end_date: End date
            days: Number of days to go back
            
        Returns:
            DateRange spanning the specified period
        """
        if days < 0:
            raise DataValidationError("Days must be non-negative")
        
        end = cls._to_date(end_date, "end")
        start = end - timedelta(days=days)
        
        return cls(start, end)
    
    @classmethod
    def from_years(cls, end_date: Union[str, date, datetime], years: float) -> DateRange:
        """
        Create DateRange by going back specified years from end date.
        
        Args:
            end_date: End date
            years: Number of years to go back
            
        Returns:
            DateRange spanning the specified period
        """
        if years < 0:
            raise DataValidationError("Years must be non-negative")
        
        end = cls._to_date(end_date, "end")
        
        # Calculate approximate start date
        days_back = int(years * 365.25)  # Account for leap years
        start = end - timedelta(days=days_back)
        
        return cls(start, end)
    
    @classmethod
    def last_n_years(cls, years: float, reference_date: Optional[Union[str, date, datetime]] = None) -> DateRange:
        """
        Create DateRange for last N years from reference date (default today).
        
        Args:
            years: Number of years
            reference_date: Reference date (default: today)
            
        Returns:
            DateRange for the period
        """
        if reference_date is None:
            reference_date = date.today()
        
        return cls.from_years(reference_date, years)
    
    @classmethod
    def year_to_date(cls, year: Optional[int] = None) -> DateRange:
        """
        Create DateRange for year-to-date period.
        
        Args:
            year: Year (default: current year)
            
        Returns:
            DateRange from Jan 1 to today (or end of year if past year)
        """
        if year is None:
            year = date.today().year
        
        start = date(year, 1, 1)
        
        if year == date.today().year:
            end = date.today()
        else:
            end = date(year, 12, 31)
        
        return cls(start, end)
    
    @classmethod
    def full_year(cls, year: int) -> DateRange:
        """
        Create DateRange for full calendar year.
        
        Args:
            year: Calendar year
            
        Returns:
            DateRange from Jan 1 to Dec 31
        """
        return cls(date(year, 1, 1), date(year, 12, 31))
    
    @property
    def days(self) -> int:
        """Get number of days in range (inclusive)."""
        return (self.end_date - self.start_date).days + 1
    
    @property
    def years(self) -> float:
        """Get approximate number of years in range."""
        return self.days / 365.25
    
    @property
    def months(self) -> float:
        """Get approximate number of months in range."""
        return self.days / 30.44  # Average month length
    
    @property
    def business_days(self) -> int:
        """Get approximate number of business days (excludes weekends)."""
        # Rough estimation: 5/7 of total days
        return int(self.days * 5 / 7)
    
    @property
    def trading_days(self) -> int:
        """Get approximate number of trading days (~252 per year)."""
        return int(self.years * 252)
    
    def contains(self, check_date: Union[str, date, datetime]) -> bool:
        """
        Check if date falls within range (inclusive).
        
        Args:
            check_date: Date to check
            
        Returns:
            True if date is within range
        """
        check = self._to_date(check_date, "check")
        return self.start_date <= check <= self.end_date
    
    def overlaps(self, other: DateRange) -> bool:
        """
        Check if this range overlaps with another.
        
        Args:
            other: DateRange to check against
            
        Returns:
            True if ranges overlap
        """
        if not isinstance(other, DateRange):
            raise TypeError(f"Cannot compare DateRange with {type(other)}")
        
        return not (self.end_date < other.start_date or self.start_date > other.end_date)
    
    def intersection(self, other: DateRange) -> Optional[DateRange]:
        """
        Get intersection with another DateRange.
        
        Args:
            other: DateRange to intersect with
            
        Returns:
            Intersecting DateRange, or None if no overlap
        """
        if not isinstance(other, DateRange):
            raise TypeError(f"Cannot intersect DateRange with {type(other)}")
        
        if not self.overlaps(other):
            return None
        
        start = max(self.start_date, other.start_date)
        end = min(self.end_date, other.end_date)
        
        return DateRange(start, end)
    
    def union(self, other: DateRange) -> DateRange:
        """
        Get union with another DateRange.
        
        Args:
            other: DateRange to union with
            
        Returns:
            Combined DateRange spanning both periods
        """
        if not isinstance(other, DateRange):
            raise TypeError(f"Cannot union DateRange with {type(other)}")
        
        start = min(self.start_date, other.start_date)
        end = max(self.end_date, other.end_date)
        
        return DateRange(start, end)
    
    def shift(self, days: int) -> DateRange:
        """
        Shift range by specified number of days.
        
        Args:
            days: Number of days to shift (positive = forward, negative = backward)
            
        Returns:
            New DateRange shifted by specified days
        """
        delta = timedelta(days=days)
        return DateRange(
            self.start_date + delta,
            self.end_date + delta
        )
    
    def extend(self, days_before: int = 0, days_after: int = 0) -> DateRange:
        """
        Extend range by specified days before/after.
        
        Args:
            days_before: Days to extend before start date
            days_after: Days to extend after end date
            
        Returns:
            Extended DateRange
        """
        new_start = self.start_date - timedelta(days=days_before)
        new_end = self.end_date + timedelta(days=days_after)
        
        return DateRange(new_start, new_end)
    
    def split_by_years(self) -> list[DateRange]:
        """
        Split range into yearly chunks.
        
        Returns:
            List of DateRange objects, one per calendar year
        """
        if self.start_date.year == self.end_date.year:
            return [self]
        
        ranges = []
        current_year = self.start_date.year
        
        while current_year <= self.end_date.year:
            if current_year == self.start_date.year:
                # First year: start_date to end of year
                year_start = self.start_date
                year_end = date(current_year, 12, 31)
            elif current_year == self.end_date.year:
                # Last year: start of year to end_date
                year_start = date(current_year, 1, 1)
                year_end = self.end_date
            else:
                # Full year
                year_start = date(current_year, 1, 1)
                year_end = date(current_year, 12, 31)
            
            ranges.append(DateRange(year_start, year_end))
            current_year += 1
        
        return ranges
    
    def format(self, format_string: str = "%Y-%m-%d") -> str:
        """
        Format date range as string.
        
        Args:
            format_string: Date format string
            
        Returns:
            Formatted string like "2020-01-01 to 2023-12-31"
        """
        start_str = self.start_date.strftime(format_string)
        end_str = self.end_date.strftime(format_string)
        return f"{start_str} to {end_str}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "days": self.days,
            "years": round(self.years, 3),
            "trading_days": self.trading_days
        }
    
    def __str__(self) -> str:
        """String representation."""
        return self.format()
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"DateRange('{self.start_date}', '{self.end_date}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, DateRange):
            return False
        return self.start_date == other.start_date and self.end_date == other.end_date
    
    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        return hash((self.start_date, self.end_date))
    
    def __lt__(self, other: DateRange) -> bool:
        """Less than comparison (by start date)."""
        if not isinstance(other, DateRange):
            raise TypeError(f"Cannot compare DateRange with {type(other)}")
        return self.start_date < other.start_date
    
    def __contains__(self, check_date: Union[str, date, datetime]) -> bool:
        """Check if date is in range using 'in' operator."""
        return self.contains(check_date)
