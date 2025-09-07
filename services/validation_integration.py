"""
Validation Integration Service.

Integrates the detailed ValidationFlags from nirvana_risk library
with the backend database storage and price_series updates.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session  # type: ignore

from core.db import get_db_session
from core.models import PriceSeries, ValidationFlags as DBValidationFlags

# Import the library ValidationFlags and aggregator
try:
    from nirvana_risk.timeseries import (
        ValidationFlags as LibValidationFlags,
        aggregate_validation_flags
    )
    from nirvana_risk.pipeline.errors import InsufficientHistoryError, InsufficientDataError
except ImportError:
    # Fallback if library is not available
    LibValidationFlags = None
    aggregate_validation_flags = None
    InsufficientHistoryError = Exception
    InsufficientDataError = Exception

_logger = logging.getLogger("nirvana.validation_integration")


class ValidationIntegrationService:
    """Service for integrating ValidationFlags with database storage."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize with optional database session."""
        self._session = db_session
        self._own_session = db_session is None
    
    def _get_session(self) -> Session:
        """Get database session (create if needed)."""
        if self._session is None:
            self._session = get_db_session()
        return self._session
    
    def _close_session(self) -> None:
        """Close session if we own it."""
        if self._own_session and self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def process_validation_result(
        self,
        symbol: str,
        validation_data: Dict[str, Any],
        country: Optional[str] = None,
        as_of_date: Optional[date] = None
    ) -> Optional[DBValidationFlags]:
        """
        Process validation results from load_prices() and related functions.
        
        Args:
            symbol: Symbol being validated
            validation_data: Results from load_prices(), anomaly detection, etc.
            country: Country code 
            as_of_date: Validation date (defaults to today)
        
        Returns:
            Database ValidationFlags record or None if library not available
        """
        if LibValidationFlags is None or aggregate_validation_flags is None:
            _logger.warning("nirvana_risk library not available, skipping detailed validation")
            return None
            
        session = self._get_session()
        
        if as_of_date is None:
            as_of_date = date.today()
        
        try:
            # Extract validation components from the data
            validation_result = None
            liquidity_decision = None
            anomaly_report = None
            history_error = None
            years_actual = None
            years_required = None
            returns_count = None
            
            # Parse validation_data from various sources
            success = validation_data.get("success", True)
            error_code = validation_data.get("code", "")
            
            # History errors
            if not success and error_code == "insufficient_history":
                years_actual = validation_data.get("years", 0.0)
                years_required = validation_data.get("min_years", 10.0)  # Default to 10 years
                history_error = InsufficientHistoryError(years=years_actual, min_years=years_required)
            elif not success and error_code == "insufficient_data":
                history_error = InsufficientDataError("insufficient data")
                returns_count = validation_data.get("returns_count")
            
            # Get anomaly report if available
            anomaly_report = validation_data.get("anomalies_report")
            
            # Get liquidity decision if available  
            liquidity_decision = validation_data.get("liquidity_decision")
            
            # Create ValidationFlags using library aggregator
            lib_flags = aggregate_validation_flags(
                symbol=symbol,
                validation_result=validation_result,
                liquidity_decision=liquidity_decision,
                anomaly_report=anomaly_report,
                history_error=history_error,
                years_actual=years_actual,
                years_required=years_required,
                returns_count=returns_count
            )
            
            # Convert library ValidationFlags to database model
            db_flags = self._convert_to_db_model(
                lib_flags, 
                symbol=symbol, 
                country=country, 
                as_of_date=as_of_date
            )
            
            # Save to database
            db_record = self._save_validation_flags(session, db_flags)
            
            # Update both flags in price_series
            has_history_issues = (
                lib_flags.insufficient_total_history or 
                lib_flags.insufficient_data_after_cleanup
            )
            self._sync_price_series_flags(session, symbol, country, has_history_issues, lib_flags.valid)
            
            session.commit()
            return db_record
            
        except Exception as e:
            session.rollback()
            _logger.error(f"Failed to process validation result for {symbol}: {e}")
            raise
        finally:
            if self._own_session:
                self._close_session()
    
    def _convert_to_db_model(
        self, 
        lib_flags: LibValidationFlags, 
        symbol: str, 
        country: Optional[str], 
        as_of_date: date
    ) -> DBValidationFlags:
        """Convert library ValidationFlags to database model."""
        return DBValidationFlags(
            symbol=symbol,
            country=country,
            as_of_date=as_of_date,
            valid=1 if lib_flags.valid else 0,
            
            # History criteria
            insufficient_total_history=1 if lib_flags.insufficient_total_history else 0,
            insufficient_data_after_cleanup=1 if lib_flags.insufficient_data_after_cleanup else 0,
            
            # Structural criteria
            backward_dates=1 if lib_flags.backward_dates else 0,
            zero_or_negative_prices=1 if lib_flags.zero_or_negative_prices else 0,
            extreme_price_jumps=1 if lib_flags.extreme_price_jumps else 0,
            
            # Liquidity criteria
            critical_years=1 if lib_flags.critical_years else 0,
            multiple_violations_last252=1 if lib_flags.multiple_violations_last252 else 0,
            multiple_weak_years=1 if lib_flags.multiple_weak_years else 0,
            low_liquidity_warning=1 if lib_flags.low_liquidity_warning else 0,
            
            # Anomaly criteria  
            robust_outliers=1 if lib_flags.robust_outliers else 0,
            price_discontinuities=1 if lib_flags.price_discontinuities else 0,
            long_plateaus=1 if lib_flags.long_plateaus else 0,
            illiquid_spikes=1 if lib_flags.illiquid_spikes else 0,
            
            # Analytics data
            liquidity_metrics=lib_flags.liquidity_metrics,
            anomaly_details=lib_flags.anomaly_details,
            validation_summary=lib_flags.validation_summary,
        )
    
    def _save_validation_flags(
        self, 
        session: Session, 
        db_flags: DBValidationFlags
    ) -> DBValidationFlags:
        """Save or update ValidationFlags in database."""
        # Try to find existing record
        existing = session.query(DBValidationFlags).filter(
            DBValidationFlags.symbol == db_flags.symbol,
            DBValidationFlags.country == db_flags.country,
            DBValidationFlags.as_of_date == db_flags.as_of_date
        ).one_or_none()
        
        if existing:
            # Update existing record
            for attr in [
                'valid', 'insufficient_total_history', 'insufficient_data_after_cleanup',
                'backward_dates', 'zero_or_negative_prices', 'extreme_price_jumps',
                'critical_years', 'multiple_violations_last252', 'multiple_weak_years', 'low_liquidity_warning',
                'robust_outliers', 'price_discontinuities', 'long_plateaus', 'illiquid_spikes',
                'liquidity_metrics', 'anomaly_details', 'validation_summary'
            ]:
                setattr(existing, attr, getattr(db_flags, attr))
            existing.updated_at = datetime.utcnow()
            return existing
        else:
            # Create new record
            session.add(db_flags)
            return db_flags
    
    def _sync_price_series_flags(
        self, 
        session: Session, 
        symbol: str, 
        country: Optional[str], 
        has_history_issues: bool,
        is_valid: bool
    ) -> None:
        """Sync price_series flags with ValidationFlags."""
        price_series = session.query(PriceSeries).filter(
            PriceSeries.symbol == symbol,
            PriceSeries.country == country
        ).one_or_none()
        
        if price_series:
            # insufficient_history: only for data history problems
            price_series.insufficient_history = 1 if has_history_issues else 0
            
            # valid: general validity flag (any rejection reasons)
            price_series.valid = 1 if is_valid else 0
            
            _logger.debug(
                f"Updated {symbol}: insufficient_history={price_series.insufficient_history}, "
                f"valid={price_series.valid}"
            )


# Convenience functions
def process_ticker_validation(
    symbol: str,
    validation_data: Dict[str, Any],
    country: Optional[str] = None,
    as_of_date: Optional[date] = None,
    db_session: Optional[Session] = None
) -> Optional[DBValidationFlags]:
    """
    Process validation result for a single ticker.
    
    This is the main integration point for ticker processing pipelines.
    """
    service = ValidationIntegrationService(db_session)
    return service.process_validation_result(
        symbol=symbol,
        validation_data=validation_data,
        country=country,
        as_of_date=as_of_date
    )


def sync_insufficient_history_flags(limit: int = 1000) -> int:
    """
    Sync existing insufficient_history flags from price_series to validation_flags.
    
    This is useful for migrating existing data.
    
    Returns:
        Number of records processed
    """
    if LibValidationFlags is None:
        _logger.warning("nirvana_risk library not available, skipping migration")
        return 0
        
    session = get_db_session()
    count = 0
    
    try:
        # Get price_series records that have insufficient_history set but no validation_flags
        price_series_records = session.query(PriceSeries).filter(
            PriceSeries.insufficient_history.isnot(None)
        ).limit(limit).all()
        
        for ps in price_series_records:
            # Check if validation flags already exist
            existing = session.query(DBValidationFlags).filter(
                DBValidationFlags.symbol == ps.symbol,
                DBValidationFlags.country == ps.country
            ).first()
            
            if not existing:
                # Create basic validation flags from insufficient_history
                validation_data = {
                    "success": ps.insufficient_history != 1,
                    "code": "insufficient_history" if ps.insufficient_history == 1 else "success"
                }
                
                service = ValidationIntegrationService(session)
                service.process_validation_result(
                    symbol=ps.symbol,
                    validation_data=validation_data,
                    country=ps.country,
                    as_of_date=date.today()
                )
                count += 1
                
        session.commit()
        return count
        
    except Exception as e:
        session.rollback()
        _logger.error(f"Failed to sync insufficient_history flags: {e}")
        raise
    finally:
        session.close()
