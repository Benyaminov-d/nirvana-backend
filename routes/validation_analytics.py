"""
Validation Analytics API endpoints.

Provides detailed analytics on data validation, knockout policy effectiveness,
and symbol rejection reasons for comprehensive dashboard reporting.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
import concurrent.futures
import uuid
import threading

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import and_, func, select  # type: ignore


from utils.auth import basic_auth_if_configured as _basic_auth_if_configured
from core.db import get_db_session
from core.models import PriceSeries, ValidationFlags

_logger = logging.getLogger("nirvana.validation_analytics")

router = APIRouter()

# Global storage for background job statuses
_job_statuses: Dict[str, Dict[str, Any]] = {}
_job_statuses_lock = threading.Lock()


@router.get("/validation/analytics/summary")
def get_validation_summary(
    country: Optional[str] = Query(None, description="Filter by country"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated list of instrument types to filter by"),
    as_of_date: Optional[str] = Query(None, description="Filter by specific date (YYYY-MM-DD)"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Get comprehensive validation summary statistics.
    
    Returns detailed analytics on symbol validation status, rejection reasons,
    and trends for data quality monitoring.
    """
    session = get_db_session()
    
    try:
        # Base queries
        ps_query = session.query(PriceSeries)
        vf_query = session.query(ValidationFlags)
        
        # Apply filters
        if country:
            ps_query = ps_query.filter(PriceSeries.country == country)
            vf_query = vf_query.filter(ValidationFlags.country == country)
            
        if instrument_types:
            # Parse comma-separated instrument types
            types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
            if types_list:
                ps_query = ps_query.filter(PriceSeries.instrument_type.in_(types_list))
                # Apply instrument_types filter to vf_query via join with PriceSeries (BOTH symbol AND country)
                vf_query = vf_query.join(
                    PriceSeries, 
                    and_(
                        PriceSeries.symbol == ValidationFlags.symbol,
                        PriceSeries.country == ValidationFlags.country
                    )
                ).filter(PriceSeries.instrument_type.in_(types_list))
            
        if as_of_date:
            try:
                filter_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
                vf_query = vf_query.filter(ValidationFlags.as_of_date == filter_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Total symbols in database
        total_symbols = ps_query.count()
        
        # Validation flags statistics
        if not as_of_date:
            # Get latest validation for each symbol with filtering
            latest_validations_base = session.query(
                ValidationFlags.symbol,
                ValidationFlags.country,
                func.max(ValidationFlags.as_of_date).label('latest_date')
            )
            
            # Apply filters to the subquery
            if country:
                latest_validations_base = latest_validations_base.filter(ValidationFlags.country == country)
            if instrument_types:
                # Parse comma-separated instrument types
                types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
                if types_list:
                    latest_validations_base = latest_validations_base.join(
                        PriceSeries, ValidationFlags.symbol == PriceSeries.symbol
                    ).filter(PriceSeries.instrument_type.in_(types_list))
                
            latest_validations = latest_validations_base.group_by(ValidationFlags.symbol, ValidationFlags.country).subquery()
            
            vf_query = session.query(ValidationFlags).join(
                latest_validations,
                and_(
                    ValidationFlags.symbol == latest_validations.c.symbol,
                    ValidationFlags.country == latest_validations.c.country,
                    ValidationFlags.as_of_date == latest_validations.c.latest_date
                )
            )
        
        # Basic counts
        total_validated = vf_query.count()
        valid_symbols = vf_query.filter(ValidationFlags.valid == 1).count()
        invalid_symbols = vf_query.filter(ValidationFlags.valid == 0).count()
        
        # Rejection reason counts
        rejection_stats = {}
        
        # History issues
        rejection_stats["insufficient_total_history"] = vf_query.filter(
            ValidationFlags.insufficient_total_history == 1
        ).count()
        rejection_stats["insufficient_data_after_cleanup"] = vf_query.filter(
            ValidationFlags.insufficient_data_after_cleanup == 1
        ).count()
        
        # Structural issues
        rejection_stats["backward_dates"] = vf_query.filter(
            ValidationFlags.backward_dates == 1
        ).count()
        rejection_stats["zero_or_negative_prices"] = vf_query.filter(
            ValidationFlags.zero_or_negative_prices == 1
        ).count()
        rejection_stats["extreme_price_jumps"] = vf_query.filter(
            ValidationFlags.extreme_price_jumps == 1
        ).count()
        
        # Liquidity issues
        rejection_stats["critical_years"] = vf_query.filter(
            ValidationFlags.critical_years == 1
        ).count()
        rejection_stats["multiple_violations_last252"] = vf_query.filter(
            ValidationFlags.multiple_violations_last252 == 1
        ).count()
        rejection_stats["multiple_weak_years"] = vf_query.filter(
            ValidationFlags.multiple_weak_years == 1
        ).count()
        rejection_stats["low_liquidity_warning"] = vf_query.filter(
            ValidationFlags.low_liquidity_warning == 1
        ).count()
        
        # Anomaly issues
        rejection_stats["robust_outliers"] = vf_query.filter(
            ValidationFlags.robust_outliers == 1
        ).count()
        rejection_stats["price_discontinuities"] = vf_query.filter(
            ValidationFlags.price_discontinuities == 1
        ).count()
        rejection_stats["long_plateaus"] = vf_query.filter(
            ValidationFlags.long_plateaus == 1
        ).count()
        rejection_stats["illiquid_spikes"] = vf_query.filter(
            ValidationFlags.illiquid_spikes == 1
        ).count()
        
        # Calculate percentages
        valid_percentage = (valid_symbols / total_validated * 100) if total_validated > 0 else 0
        coverage_percentage = (total_validated / total_symbols * 100) if total_symbols > 0 else 0
        
        # Top rejection reasons
        rejection_reasons_sorted = sorted(
            rejection_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate category totals matching frontend expectations
        historical_failures = rejection_stats["insufficient_total_history"] + rejection_stats["insufficient_data_after_cleanup"]
        structural_failures = sum([
            rejection_stats["backward_dates"],
            rejection_stats["zero_or_negative_prices"],
            rejection_stats["extreme_price_jumps"],
            rejection_stats["price_discontinuities"]
        ])
        liquidity_warnings = sum([
            rejection_stats["low_liquidity_warning"],
            rejection_stats["long_plateaus"],
            rejection_stats["illiquid_spikes"]
        ])
        statistical_failures = sum([
            rejection_stats["critical_years"],
            rejection_stats["multiple_violations_last252"],
            rejection_stats["multiple_weak_years"],
            rejection_stats["robust_outliers"]
        ])

        # Build response matching frontend ValidationSummary interface
        result = {
            "total_symbols": total_symbols,
            "total_validated": total_validated,
            "valid_symbols": valid_symbols,
            "invalid_symbols": invalid_symbols,
            
            # Breakdown by category
            "historical_sufficiency_failures": historical_failures,
            "structural_integrity_failures": structural_failures,
            "liquidity_activity_warnings": liquidity_warnings,
            "statistical_anomaly_failures": statistical_failures,
            
            # Individual flag counts
            "flag_counts": {
                "insufficient_total_history": rejection_stats["insufficient_total_history"],
                "insufficient_data_after_cleanup": rejection_stats["insufficient_data_after_cleanup"],
                "backward_dates": rejection_stats["backward_dates"],
                "zero_or_negative_prices": rejection_stats["zero_or_negative_prices"],
                "extreme_price_jumps": rejection_stats["extreme_price_jumps"],
                "price_discontinuities": rejection_stats["price_discontinuities"],
                "low_liquidity_warning": rejection_stats["low_liquidity_warning"],
                "long_plateaus": rejection_stats["long_plateaus"],
                "illiquid_spikes": rejection_stats["illiquid_spikes"],
                "critical_years": rejection_stats["critical_years"],
                "multiple_violations_last252": rejection_stats["multiple_violations_last252"],
                "multiple_weak_years": rejection_stats["multiple_weak_years"],
                "robust_outliers": rejection_stats["robust_outliers"]
            },
            
            # Percentage calculations 
            # valid/invalid percentages are relative to total_validated (how many of analyzed symbols are valid/invalid)
            "valid_percentage": round((valid_symbols / max(1, total_validated)) * 100, 1),
            "invalid_percentage": round((invalid_symbols / max(1, total_validated)) * 100, 1),
            # validated percentage is relative to total_symbols (coverage)
            "validated_percentage": round((total_validated / max(1, total_symbols)) * 100, 1),
            
            # Category percentages (relative to total_validated - what % of analyzed symbols have these issues)
            "historical_failures_percentage": round((historical_failures / max(1, total_validated)) * 100, 1),
            "structural_failures_percentage": round((structural_failures / max(1, total_validated)) * 100, 1),
            "liquidity_warnings_percentage": round((liquidity_warnings / max(1, total_validated)) * 100, 1),
            "statistical_failures_percentage": round((statistical_failures / max(1, total_validated)) * 100, 1),
            
            # Success rates by category (among validated symbols)
            "category_success_rates": {
                "historical_sufficiency": round(((total_validated - historical_failures) / max(1, total_validated)) * 100, 1),
                "structural_integrity": round(((total_validated - structural_failures) / max(1, total_validated)) * 100, 1),
                "liquidity_activity": round(((total_validated - liquidity_warnings) / max(1, total_validated)) * 100, 1),
                "statistical_anomaly": round(((total_validated - statistical_failures) / max(1, total_validated)) * 100, 1)
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        _logger.error(f"Failed to get validation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.get("/validation/analytics/search")
def search_validation_flags(
    limit: int = Query(50, description="Number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    country: Optional[str] = Query(None, description="Filter by country"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated list of instrument types to filter by"),
    valid: Optional[bool] = Query(None, description="Filter by validity"),
    has_warnings: Optional[bool] = Query(None, description="Filter by warnings"),
    flags: Optional[str] = Query(None, description="Comma-separated list of flags to filter by"),
    symbol_search: Optional[str] = Query(None, description="Search symbols by pattern"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    sort_by: Optional[str] = Query(None, description="Sort field: 'symbol', 'country', 'valid', 'years_actual', 'updated_at'"),
    sort_order: Optional[str] = Query("desc", description="Sort order: 'asc' or 'desc'"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Search validation flags with filtering and pagination.
    
    Returns paginated list of validation flags with detailed rejection reasons
    and warnings for comprehensive analysis.
    """
    session = get_db_session()
    
    try:
        # Base query
        query = session.query(ValidationFlags).join(
            PriceSeries, ValidationFlags.symbol == PriceSeries.symbol
        )
        
        # Apply filters
        if country:
            query = query.filter(ValidationFlags.country == country)
        if instrument_types:
            # Parse comma-separated instrument types
            types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
            if types_list:
                query = query.filter(PriceSeries.instrument_type.in_(types_list))
        if valid is not None:
            query = query.filter(ValidationFlags.valid == (1 if valid else 0))
        if symbol_search:
            query = query.filter(ValidationFlags.symbol.ilike(f"%{symbol_search}%"))
        if date_from:
            try:
                from_date = datetime.strptime(date_from, "%Y-%m-%d").date()
                query = query.filter(ValidationFlags.as_of_date >= from_date)
            except ValueError:
                pass
        if date_to:
            try:
                to_date = datetime.strptime(date_to, "%Y-%m-%d").date()
                query = query.filter(ValidationFlags.as_of_date <= to_date)
            except ValueError:
                pass
        
        # Filter by specific flags if provided
        if flags:
            flag_list = [f.strip() for f in flags.split(',') if f.strip()]
            valid_flags = [flag for flag in flag_list if hasattr(ValidationFlags, flag)]
            if valid_flags:
                # Use OR logic - symbol should have ANY of the selected flags
                from sqlalchemy import or_
                flag_conditions = [getattr(ValidationFlags, flag) == 1 for flag in valid_flags]
                query = query.filter(or_(*flag_conditions))
        
        # Filter by warnings if specified
        if has_warnings is not None:
            if has_warnings:
                # Has any warning flags
                query = query.filter(
                    (ValidationFlags.low_liquidity_warning == 1) |
                    (ValidationFlags.long_plateaus == 1) |
                    (ValidationFlags.illiquid_spikes == 1)
                )
            else:
                # No warning flags
                query = query.filter(
                    (ValidationFlags.low_liquidity_warning == 0) &
                    (ValidationFlags.long_plateaus == 0) &
                    (ValidationFlags.illiquid_spikes == 0)
                )
        
        # Get total count
        total = query.count()
        
        # Apply sorting
        sort_field = sort_by.lower() if sort_by else 'updated_at'
        is_asc = sort_order.lower() == 'asc' if sort_order else False
        
        if sort_field == 'symbol':
            order_expr = ValidationFlags.symbol.asc() if is_asc else ValidationFlags.symbol.desc()
        elif sort_field == 'country':
            order_expr = ValidationFlags.country.asc() if is_asc else ValidationFlags.country.desc()
        elif sort_field == 'valid':
            order_expr = ValidationFlags.valid.asc() if is_asc else ValidationFlags.valid.desc()
        elif sort_field == 'updated_at':
            order_expr = ValidationFlags.updated_at.asc() if is_asc else ValidationFlags.updated_at.desc()
        elif sort_field == 'years_actual':
            # For years_actual sorting, we need to extract from JSON - use PostgreSQL JSON functions
            # For PostgreSQL: validation_summary->>'years_actual'::numeric
            # For SQLite: we'll sort in Python after fetching
            order_expr = ValidationFlags.updated_at.desc()  # Fallback ordering for now
        else:
            order_expr = ValidationFlags.updated_at.desc()  # Default fallback
        
        # Apply pagination and ordering
        items = query.order_by(order_expr, ValidationFlags.symbol).offset(offset).limit(limit).all()
        
        # Format results
        results = []
        for vf in items:
            # Get price series data (MUST match both symbol AND country)
            ps = session.query(PriceSeries).filter(
                PriceSeries.symbol == vf.symbol,
                PriceSeries.country == vf.country
            ).first()
            
            # Extract years_actual from validation_summary JSON
            years_actual = None
            if vf.validation_summary:
                try:
                    import json
                    if isinstance(vf.validation_summary, str):
                        summary = json.loads(vf.validation_summary)
                    else:
                        summary = vf.validation_summary
                    years_actual = summary.get('years_actual')
                    if years_actual is not None:
                        years_actual = float(years_actual)
                except Exception:
                    years_actual = None
            
            result = {
                "id": vf.id,
                "symbol": vf.symbol,
                "country": vf.country,
                "valid": bool(vf.valid),
                "as_of_date": vf.as_of_date.isoformat() if vf.as_of_date else None,
                "years_actual": years_actual,  # Add years_actual field
                "rejection_reasons": _get_flag_reasons(vf),
                "warnings": _get_flag_warnings(vf),
                "instrument_info": {
                    "name": ps.name if ps else None,
                    "exchange": ps.exchange if ps else None,
                    "instrument_type": ps.instrument_type if ps else None,
                    "current_insufficient_history": ps.insufficient_history if ps else None,
                    "current_valid_status": ps.valid if ps else None
                },
                "validation_summary": vf.validation_summary,
                "created_at": vf.created_at.isoformat() if vf.created_at else None,
                "updated_at": vf.updated_at.isoformat() if vf.updated_at else None
            }
            results.append(result)
        
        # Handle years_actual sorting in Python if needed
        if sort_field == 'years_actual':
            def sort_key(item):
                years = item.get('years_actual')
                # Put None values at the end
                return (years is None, years if years is not None else 0)
            
            results.sort(key=sort_key, reverse=not is_asc)
        
        response = {
            "items": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        _logger.error(f"Failed to search validation flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.get("/validation/analytics/by-symbol")
def get_validation_by_symbol(
    symbol: str = Query(..., description="Symbol to analyze"),
    limit: int = Query(10, description="Number of recent validations to return"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """Get detailed validation history for a specific symbol."""
    session = get_db_session()
    
    try:
        # Get recent validation flags for the symbol
        validation_flags = session.query(ValidationFlags).filter(
            ValidationFlags.symbol == symbol
        ).order_by(ValidationFlags.as_of_date.desc()).limit(limit).all()
        
        if not validation_flags:
            raise HTTPException(status_code=404, detail=f"No validation data found for symbol {symbol}")
        
        # Get price series info
        price_series = session.query(PriceSeries).filter(
            PriceSeries.symbol == symbol
        ).first()
        
        # Format results
        validations = []
        for vf in validation_flags:
            validation_data = {
                "as_of_date": vf.as_of_date.isoformat(),
                "valid": bool(vf.valid),
                "rejection_reasons": _get_flag_reasons(vf),
                "warnings": _get_flag_warnings(vf),
                "liquidity_metrics": vf.liquidity_metrics,
                "anomaly_details": vf.anomaly_details,
                "validation_summary": vf.validation_summary,
                "created_at": vf.created_at.isoformat(),
                "updated_at": vf.updated_at.isoformat()
            }
            validations.append(validation_data)
        
        result = {
            "symbol": symbol,
            "instrument_info": {
                "name": price_series.name if price_series else None,
                "country": price_series.country if price_series else None,
                "exchange": price_series.exchange if price_series else None,
                "instrument_type": price_series.instrument_type if price_series else None,
                "current_insufficient_history": price_series.insufficient_history if price_series else None,
                "current_valid_status": price_series.valid if price_series else None
            },
            "validation_history": validations,
            "summary": {
                "total_validations": len(validations),
                "latest_status": "valid" if validations[0]["valid"] else "invalid",
                "latest_as_of": validations[0]["as_of_date"],
                "has_recent_issues": not validations[0]["valid"] if validations else False
            }
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Failed to get validation data for symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.get("/validation/analytics/trends")
def get_validation_trends(
    days: int = Query(30, description="Number of days to analyze"),
    country: Optional[str] = Query(None, description="Filter by country"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """Get validation trends over time."""
    session = get_db_session()
    
    try:
        # Get validation trends over the last N days
        start_date = date.today() - timedelta(days=days)
        
        query = session.query(
            ValidationFlags.as_of_date,
            func.count(ValidationFlags.id).label('total'),
            func.sum(ValidationFlags.valid).label('valid_count'),
            func.sum(1 - ValidationFlags.valid).label('invalid_count')
        ).filter(ValidationFlags.as_of_date >= start_date)
        
        if country:
            query = query.filter(ValidationFlags.country == country)
            
        trends = query.group_by(ValidationFlags.as_of_date).order_by(ValidationFlags.as_of_date).all()
        
        # Format trend data
        trend_data = []
        for trend in trends:
            valid_percentage = (trend.valid_count / trend.total * 100) if trend.total > 0 else 0
            trend_data.append({
                "date": trend.as_of_date.isoformat(),
                "total_validations": trend.total,
                "valid_count": trend.valid_count,
                "invalid_count": trend.invalid_count,
                "valid_percentage": round(valid_percentage, 2)
            })
        
        # Calculate overall trend
        if len(trend_data) >= 2:
            recent_avg = sum(t["valid_percentage"] for t in trend_data[-7:]) / min(7, len(trend_data))
            older_avg = sum(t["valid_percentage"] for t in trend_data[:-7]) / max(1, len(trend_data) - 7)
            trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        else:
            trend_direction = "insufficient_data"
        
        result = {
            "period": {
                "days": days,
                "start_date": start_date.isoformat(),
                "end_date": date.today().isoformat()
            },
            "trend_direction": trend_direction,
            "daily_data": trend_data,
            "filters": {
                "country": country
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        _logger.error(f"Failed to get validation trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


def _get_flag_reasons(vf: ValidationFlags) -> List[str]:
    """Extract rejection reasons from validation flags."""
    reasons = []
    if vf.insufficient_total_history:
        reasons.append("insufficient_total_history")
    if vf.insufficient_data_after_cleanup:
        reasons.append("insufficient_data_after_cleanup")
    if vf.backward_dates:
        reasons.append("backward_dates")
    if vf.zero_or_negative_prices:
        reasons.append("zero_or_negative_prices")
    if vf.extreme_price_jumps:
        reasons.append("extreme_price_jumps")
    if vf.critical_years:
        reasons.append("critical_years")
    if vf.multiple_violations_last252:
        reasons.append("multiple_violations_last252")
    if vf.multiple_weak_years:
        reasons.append("multiple_weak_years")
    if vf.robust_outliers:
        reasons.append("robust_outliers")
    if vf.price_discontinuities:
        reasons.append("price_discontinuities")
    return reasons


def _get_flag_warnings(vf: ValidationFlags) -> List[str]:
    """Extract warning reasons from validation flags."""
    warnings = []
    if vf.low_liquidity_warning:
        warnings.append("low_liquidity_warning")
    if vf.long_plateaus:
        warnings.append("long_plateaus")
    if vf.illiquid_spikes:
        warnings.append("illiquid_spikes")
    return warnings


@router.post("/validation/analytics/migrate")
def migrate_validation_flags(
    limit: int = Query(1000, description="Number of records to migrate"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Migrate existing insufficient_history flags to detailed ValidationFlags.
    
    This endpoint creates ValidationFlags entries for symbols that have
    insufficient_history set but no detailed validation data yet.
    """
    try:
        from services.validation_integration import sync_insufficient_history_flags
        
        migrated_count = sync_insufficient_history_flags(limit=limit)
        
        return JSONResponse(content={
            "status": "success",
            "migrated_records": migrated_count,
            "limit": limit,
            "message": f"Successfully migrated {migrated_count} records to ValidationFlags"
        })
        
    except Exception as e:
        _logger.error(f"Failed to migrate validation flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation/analytics/test/{symbol}")
def test_validation_processing(
    symbol: str,
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Test validation processing for a specific symbol.
    
    This endpoint processes a symbol through the complete validation pipeline
    and returns the detailed ValidationFlags that would be created.
    """
    try:
        from services.prices import load_prices
        from services.validation_integration import process_ticker_validation
        from core.db import get_db_session
        from core.models import PriceSeries
        
        # Load price data
        try:
            price_data = load_prices(symbol)
        except Exception as e:
            price_data = {
                "success": False,
                "error": str(e),
                "code": "load_error"
            }
        
        # Get symbol info
        session = get_db_session()
        price_series = session.query(PriceSeries).filter(
            PriceSeries.symbol == symbol
        ).first()
        
        country = price_series.country if price_series else None
        
        # Test validation processing
        try:
            validation_record = process_ticker_validation(
                symbol=symbol,
                validation_data=price_data,
                country=country,
                db_session=session
            )
            
            result = {
                "status": "success",
                "symbol": symbol,
                "country": country,
                "price_data": price_data,
                "validation_flags": {
                    "valid": bool(validation_record.valid) if validation_record else None,
                    "rejection_reasons": _get_flag_reasons(validation_record) if validation_record else [],
                    "warnings": _get_flag_warnings(validation_record) if validation_record else [],
                    "liquidity_metrics": validation_record.liquidity_metrics if validation_record else None,
                    "anomaly_details": validation_record.anomaly_details if validation_record else None,
                    "validation_summary": validation_record.validation_summary if validation_record else None
                }
            }
            
        except Exception as e:
            result = {
                "status": "error",
                "symbol": symbol,
                "country": country,
                "price_data": price_data,
                "validation_error": str(e)
            }
        
        session.close()
        return JSONResponse(content=result)
        
    except Exception as e:
        _logger.error(f"Failed to test validation processing for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation/analytics/test-batch")
def test_batch_validation_processing(
    limit: int = Query(10, description="Maximum number of symbols to process", ge=1, le=100),
    country: Optional[str] = Query(None, description="Filter symbols by country"),
    skip_existing: bool = Query(True, description="Skip symbols that already have validation_flags records"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Test validation processing for multiple symbols with limit.
    
    This endpoint processes multiple symbols through the validation pipeline
    and saves detailed ValidationFlags to the database for analytics testing.
    
    Can be called directly from browser with query parameters.
    
    Example: /api/validation/analytics/test-batch?limit=5&country=US&skip_existing=true
    
    Returns summary statistics and sample results.
    """
    from services.prices import load_prices
    from services.validation_integration import process_ticker_validation
    from core.db import get_db_session
    from core.models import PriceSeries, ValidationFlags
    
    session = get_db_session()
    
    try:
        # Build query for symbols to process
        query = session.query(PriceSeries).filter(PriceSeries.symbol.isnot(None))
        
        if country:
            query = query.filter(PriceSeries.country == country)
        
        if skip_existing:
            # Skip symbols that already have validation records
            existing_symbols = session.query(ValidationFlags.symbol).distinct()
            query = query.filter(~PriceSeries.symbol.in_(existing_symbols))
        
        # Get symbols to process
        symbols_to_process = query.limit(limit).all()
        
        if not symbols_to_process:
            return JSONResponse(content={
                "status": "no_symbols",
                "message": "No symbols found to process with given criteria",
                "filters": {
                    "country": country,
                    "skip_existing": skip_existing,
                    "limit": limit
                }
            })
        
        # Process symbols in parallel for better performance
        def _process_single_symbol(price_series) -> dict:
            """Process a single symbol through validation pipeline."""
            symbol = price_series.symbol
            country_code = price_series.country
            
            try:
                # Load price data for symbol
                _logger.info(f"Processing validation for {symbol} ({country_code})")
                
                # Check if this symbol has duplicates (exists in multiple countries)
                duplicate_count = session.query(PriceSeries).filter(
                    PriceSeries.symbol == symbol
                ).count()
                
                try:
                    if duplicate_count > 1:
                        # For duplicates, use country-specific suffix
                        from utils.common import _eodhd_suffix_for
                        suffix = _eodhd_suffix_for(price_series.exchange, country_code)
                        symbol_with_suffix = f"{symbol}{suffix}"
                        _logger.info(f"DUPLICATE DETECTED: Loading {symbol_with_suffix} for {country_code}")
                        price_data = load_prices(symbol_with_suffix)
                    else:
                        # For unique symbols, use symbol as-is
                        price_data = load_prices(symbol)
                    
                    price_load_success = True
                except Exception as e:
                    _logger.warning(f"Failed to load prices for {symbol}: {e}")
                    price_data = {
                        "success": False,
                        "error": str(e),
                        "code": "load_error"
                    }
                    price_load_success = False
                
                # Create new database session for this worker thread
                worker_session = get_db_session()
                try:
                    # Process validation regardless of price load success
                    # This allows us to test validation flags creation even with bad data
                    validation_record = process_ticker_validation(
                        symbol=symbol,
                        validation_data=price_data,
                        country=country_code,
                        db_session=worker_session
                    )
                    
                    # Collect result details
                    symbol_result = {
                        "symbol": symbol,
                        "country": country_code,
                        "status": "success",
                        "price_load_success": price_load_success,
                        "validation_flags_created": validation_record is not None,
                        "validation_record_id": validation_record.id if validation_record else None
                    }
                    
                    if validation_record:
                        symbol_result.update({
                            "valid": bool(validation_record.valid),
                            "rejection_reasons": _get_flag_reasons(validation_record),
                            "warnings": _get_flag_warnings(validation_record),
                            "as_of_date": validation_record.as_of_date.isoformat() if validation_record.as_of_date else None
                        })
                    
                    # Check if price_series was updated (MUST match both symbol AND country)
                    updated_ps = worker_session.query(PriceSeries).filter(
                        PriceSeries.symbol == symbol,
                        PriceSeries.country == country_code
                    ).first()
                    if updated_ps and updated_ps.valid is not None:
                        symbol_result["price_series_valid"] = bool(updated_ps.valid)
                        symbol_result["price_series_insufficient_history"] = bool(updated_ps.insufficient_history) if updated_ps.insufficient_history is not None else None
                        symbol_result["price_series_updated"] = True
                    else:
                        symbol_result["price_series_updated"] = False
                    
                    # Commit changes in this worker session
                    worker_session.commit()
                    return symbol_result
                    
                finally:
                    worker_session.close()
                
            except Exception as e:
                _logger.error(f"Failed to process {symbol}: {e}")
                return {
                    "symbol": symbol,
                    "country": country_code,
                    "status": "error",
                    "error": str(e),
                    "validation_flags_created": False,
                    "price_series_updated": False
                }
        
        # Get worker count from settings (respect user configuration)
        from config import get_config
        config = get_config()
        workers = max(2, getattr(config, 'exp_reprocess_workers', 4))
        
        _logger.info(
            f"Starting parallel validation with {workers} workers for "
            f"{len(symbols_to_process)} symbols"
        )
        
        # Initialize counters
        processed = 0
        successful = 0
        failed = 0
        details = []
        flags_created = 0
        price_series_updated = 0
        
        # Process symbols in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all jobs
            future_to_symbol = {
                executor.submit(_process_single_symbol, ps): ps.symbol
                for ps in symbols_to_process
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                processed += 1
                
                try:
                    result = future.result()
                    
                    if result["status"] == "success":
                        successful += 1
                        if result.get("validation_flags_created", False):
                            flags_created += 1
                        if result.get("price_series_updated", False):
                            price_series_updated += 1
                    else:
                        failed += 1
                    
                    details.append(result)
                    
                except Exception as exc:
                    failed += 1
                    _logger.error(f"Worker processing {symbol} generated an exception: {exc}")
                    details.append({
                        "symbol": symbol,
                        "status": "worker_error",
                        "error": str(exc),
                        "validation_flags_created": False,
                        "price_series_updated": False
                    })
        
        # Prepare results summary
        results = {
            "processed": processed,
            "successful": successful,
            "failed": failed,
            "details": details,
            "performance": {
                "workers_used": workers,
                "parallel_processing": True
            },
            "summary": {
                "total_requested": limit,
                "found_symbols": len(symbols_to_process),
                "processed": processed,
                "successful": successful,
                "failed": failed,
                "flags_created": flags_created,
                "price_series_updated": price_series_updated,
                "success_rate": f"{(successful / processed * 100):.1f}%" if processed > 0 else "0%"
            }
        }
        
        _logger.info(
            f"Parallel batch validation completed: {successful}/{processed} "
            f"symbols processed successfully using {workers} workers"
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        session.rollback()
        _logger.error(f"Batch validation processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    finally:
        session.close()


@router.get("/validation/analytics/instrument-types")
def get_instrument_types(
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """Get all unique instrument types from the database"""
    session = get_db_session()
    try:
        # Get all unique instrument types
        instrument_types = session.query(PriceSeries.instrument_type).distinct().filter(
            PriceSeries.instrument_type.isnot(None),
            PriceSeries.instrument_type != ''
        ).order_by(PriceSeries.instrument_type).all()
        
        # Extract the values from tuples
        types_list = [t[0] for t in instrument_types if t[0]]
        
        _logger.info(f"Found {len(types_list)} unique instrument types: {types_list}")
        
        return JSONResponse({
            "instrument_types": types_list,
            "count": len(types_list)
        })
        
    except Exception as e:
        _logger.error(f"Error fetching instrument types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching instrument types: {str(e)}")
    finally:
        session.close()


@router.get("/validation/analytics/db-status")
def get_validation_db_status(
    country: Optional[str] = Query(None, description="Filter by country"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated list of instrument types to filter by"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Get current status of validation data in the database.
    
    Quick overview of what validation data exists for testing and analysis.
    """
    session = get_db_session()
    
    try:
        # Parse instrument types filter
        types_list = []
        if instrument_types:
            types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
        
        # Base query for PriceSeries with filters
        ps_query = session.query(PriceSeries)
        if country:
            ps_query = ps_query.filter(PriceSeries.country == country)
        if types_list:
            ps_query = ps_query.filter(PriceSeries.instrument_type.in_(types_list))
        
        # Count symbols in price_series with different validation statuses (filtered)
        total_symbols = ps_query.count()
        
        # Price series flags status (filtered)
        valid_count = ps_query.filter(PriceSeries.valid == 1).count()
        invalid_count = ps_query.filter(PriceSeries.valid == 0).count()
        unprocessed_valid = ps_query.filter(PriceSeries.valid.is_(None)).count()
        
        insufficient_history_count = ps_query.filter(PriceSeries.insufficient_history == 1).count()
        sufficient_history_count = ps_query.filter(PriceSeries.insufficient_history == 0).count()
        unprocessed_history = ps_query.filter(PriceSeries.insufficient_history.is_(None)).count()
        
        # Base query for ValidationFlags with filters
        vf_query = session.query(ValidationFlags)
        if country:
            vf_query = vf_query.filter(ValidationFlags.country == country)
        if types_list:
            vf_query = vf_query.join(
                PriceSeries, 
                and_(
                    PriceSeries.symbol == ValidationFlags.symbol,
                    PriceSeries.country == ValidationFlags.country
                )
            ).filter(PriceSeries.instrument_type.in_(types_list))
        
        # ValidationFlags records (filtered)
        validation_flags_count = vf_query.count()
        
        # Recent validation flags (last 7 days, filtered)
        recent_date = (datetime.utcnow() - timedelta(days=7)).date()
        recent_validation_flags = vf_query.filter(
            ValidationFlags.created_at >= recent_date
        ).count()
        
        # Detailed flag statistics from ValidationFlags table
        flag_stats = {}
        if validation_flags_count > 0:
            # Get count for each flag type
            flag_columns = [
                'insufficient_total_history', 'insufficient_data_after_cleanup',
                'backward_dates', 'zero_or_negative_prices', 'extreme_price_jumps',
                'critical_years', 'multiple_violations_last252', 'multiple_weak_years', 'low_liquidity_warning',
                'robust_outliers', 'price_discontinuities', 'long_plateaus', 'illiquid_spikes'
            ]
            
            for flag in flag_columns:
                count = vf_query.filter(getattr(ValidationFlags, flag) == 1).count()
                if count > 0:
                    flag_stats[flag] = count
        
        # Countries breakdown (also filtered by instrument types if specified)
        countries_query = ps_query.with_entities(
            PriceSeries.country, 
            func.count(PriceSeries.symbol).label('count')
        ).group_by(PriceSeries.country).all()
        
        countries_breakdown = {country or 'Unknown': count for country, count in countries_query}
        
        # Get latest validation date (from filtered data)
        latest_validation = vf_query.with_entities(ValidationFlags.created_at).order_by(
            ValidationFlags.created_at.desc()
        ).first()
        latest_update = latest_validation[0].isoformat() if latest_validation else None
        
        # Determine filter description for display
        filter_desc = []
        if country:
            filter_desc.append(f"Country: {country}")
        if types_list:
            filter_desc.append(f"Types: {', '.join(types_list)}")
        
        filter_display = " | ".join(filter_desc) if filter_desc else "All Countries"
        
        # Build response matching frontend DatabaseStatus interface
        result = {
            "price_series_count": total_symbols,
            "validation_flags_count": validation_flags_count,
            "latest_update": latest_update,
            "countries_available": list(countries_breakdown.keys()),
            "symbols_with_flags": validation_flags_count,
            "symbols_without_flags": total_symbols - validation_flags_count,
            "filter_display": filter_display,
            "detailed_flags_breakdown": flag_stats
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        _logger.error(f"Failed to get validation DB status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


def _run_validation_job(
    job_id: str,
    limit: int,
    country: Optional[str],
    instrument_types: Optional[str], 
    skip_existing: bool
) -> None:
    """Background function to run validation job"""
    try:
        with _job_statuses_lock:
            _job_statuses[job_id]["status"] = "running"
            _job_statuses[job_id]["started_at"] = datetime.utcnow().isoformat()
            
        # Run the actual validation logic
        result = _execute_validation_job(limit, country, instrument_types, skip_existing, job_id)
        
        with _job_statuses_lock:
            _job_statuses[job_id]["status"] = "completed"
            _job_statuses[job_id]["completed_at"] = datetime.utcnow().isoformat()
            _job_statuses[job_id]["result"] = result
            
    except Exception as e:
        _logger.error(f"Background validation job {job_id} failed: {e}")
        with _job_statuses_lock:
            _job_statuses[job_id]["status"] = "failed"
            _job_statuses[job_id]["failed_at"] = datetime.utcnow().isoformat()
            _job_statuses[job_id]["error"] = str(e)


def _execute_validation_job(
    limit: int,
    country: Optional[str],
    instrument_types: Optional[str],
    skip_existing: bool,
    job_id: str
) -> Dict[str, Any]:
    """Execute the actual validation logic"""
    from services.prices import load_prices
    from services.validation_integration import process_ticker_validation
    from core.db import get_db_session
    from core.models import PriceSeries, ValidationFlags
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Get worker count from environment
    try:
        max_workers = int(os.getenv("EXP_REPROCESS_WORKERS", "8"))
    except (ValueError, TypeError):
        max_workers = 8
    
    session = get_db_session()
    
    try:
        # Parse instrument types filter
        types_list = []
        if instrument_types:
            types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
        
        # Build query for symbols to process
        query = session.query(PriceSeries).filter(PriceSeries.symbol.isnot(None))
        
        if country:
            query = query.filter(PriceSeries.country == country)
            
        if types_list:
            query = query.filter(PriceSeries.instrument_type.in_(types_list))
        
        if skip_existing:
            # Skip symbols that already have validation records
            existing_symbols = session.query(ValidationFlags.symbol).distinct()
            existing_set = {symbol for (symbol,) in existing_symbols.all()}
            if existing_set:
                query = query.filter(~PriceSeries.symbol.in_(existing_set))
        
        # Apply limit if specified
        if limit > 0:
            query = query.limit(limit)
        
        # Get all symbols to process
        symbols_to_process = [row.symbol for row in query.all()]
        total_symbols = len(symbols_to_process)
        
        if total_symbols == 0:
            return {
                "message": "No symbols to process",
                "total_symbols": 0,
                "filters_applied": {
                    "country": country,
                    "instrument_types": instrument_types,
                    "skip_existing": skip_existing,
                    "limit": limit
                },
                "processed": 0,
                "successful": 0,
                "failed": 0
            }
        
        _logger.info(f"Starting validation job {job_id} for {total_symbols} symbols with {max_workers} workers")
        
        # Update job status with progress info
        with _job_statuses_lock:
            _job_statuses[job_id]["total_symbols"] = total_symbols
            _job_statuses[job_id]["processed"] = 0
            _job_statuses[job_id]["successful"] = 0
            _job_statuses[job_id]["failed"] = 0
        
        # Process symbols with multiprocessing
        successful_count = 0
        failed_count = 0
        results = []
        
        def process_single_symbol(symbol_name):
            """Process a single symbol for validation"""
            local_session = get_db_session()
            try:
                _logger.info(f"Processing validation for {symbol_name}")
                
                # Get country info from database FIRST to determine correct data source
                price_series_record = local_session.query(PriceSeries).filter(
                    PriceSeries.symbol == symbol_name
                ).first()
                country = price_series_record.country if price_series_record else None
                
                # Check if this symbol has duplicates (exists in multiple countries)
                duplicate_count = local_session.query(PriceSeries).filter(
                    PriceSeries.symbol == symbol_name
                ).count()
                
                # Load price data with correct country-specific suffix for duplicates
                try:
                    if duplicate_count > 1 and price_series_record:
                        # For duplicates, use country-specific suffix
                        from utils.common import _eodhd_suffix_for
                        suffix = _eodhd_suffix_for(price_series_record.exchange, country)
                        symbol_with_suffix = f"{symbol_name}{suffix}"
                        _logger.info(f"DUPLICATE DETECTED: Loading {symbol_with_suffix} for {country}")
                        price_data = load_prices(symbol_with_suffix)
                    else:
                        # For unique symbols, use symbol as-is
                        price_data = load_prices(symbol_name)
                except Exception as e:
                    price_data = {
                        "success": False,
                        "error": str(e),
                        "code": "load_error"
                    }
                
                # Process validation
                result = process_ticker_validation(
                    symbol=symbol_name,
                    validation_data=price_data,
                    country=country,
                    db_session=local_session
                )
                
                local_session.commit()
                
                # Convert ValidationFlags to dict for JSON serialization
                result_summary = {
                    "valid": result.valid if result else None,
                    "has_critical_issues": result.has_critical_issues if result else None,
                    "has_warnings": result.has_warnings if result else None,
                } if result else None
                
                return {
                    "symbol": symbol_name,
                    "status": "success",
                    "validation_summary": result_summary
                }
                
            except Exception as e:
                if local_session:
                    local_session.rollback()
                error_msg = str(e)
                _logger.error(f"Failed to process validation for {symbol_name}: {error_msg}")
                return {
                    "symbol": symbol_name,
                    "status": "failed",
                    "error": error_msg
                }
            finally:
                if local_session:
                    local_session.close()
        
        # Execute with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_single_symbol, symbol): symbol 
                for symbol in symbols_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Update job progress
                processed = successful_count + failed_count
                with _job_statuses_lock:
                    _job_statuses[job_id]["processed"] = processed
                    _job_statuses[job_id]["successful"] = successful_count
                    _job_statuses[job_id]["failed"] = failed_count
                
                # Log progress more frequently
                if processed % 1 == 0 or processed == total_symbols:  # Every symbol
                    _logger.info(f"Job {job_id} progress: {processed}/{total_symbols} symbols processed ({successful_count} success, {failed_count} failed)")
        
        # Summary statistics
        success_rate = (successful_count / total_symbols * 100) if total_symbols > 0 else 0
        
        return {
            "message": f"Validation completed for {total_symbols} symbols",
            "filters_applied": {
                "country": country,
                "instrument_types": instrument_types,
                "skip_existing": skip_existing,
                "limit": limit
            },
            "processing_stats": {
                "total_symbols": total_symbols,
                "processed": successful_count + failed_count,
                "successful": successful_count, 
                "failed": failed_count,
                "success_rate": f"{success_rate:.1f}%",
                "workers_used": max_workers
            },
            "sample_results": results[:5] if results else [],  # Show first 5 results as sample
            "failed_symbols": [r["symbol"] for r in results if r["status"] == "failed"][:10]  # First 10 failed
        }
        
    finally:
        session.close()


@router.post("/validation/analytics/validate-all")
def start_validation_job(
    limit: int = Query(0, description="Maximum number of symbols to process (0 = all symbols)"),
    country: Optional[str] = Query(None, description="Filter symbols by country"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated list of instrument types to filter by"),
    skip_existing: bool = Query(True, description="Skip symbols that already have validation_flags records"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Start validation job for ALL symbols in database (or filtered subset).
    
    This endpoint returns immediately with job_id and runs validation in background.
    Use the /validation/analytics/job-status/<job_id> endpoint to check progress.
    
    WARNING: This processes potentially thousands of symbols. Use with caution.
    
    Examples:
    - POST /api/validation/analytics/validate-all (process ALL symbols)
    - POST /api/validation/analytics/validate-all?country=US&limit=1000
    - POST /api/validation/analytics/validate-all?instrument_types=ETF,Stock&skip_existing=false
    
    Returns job_id immediately (200 OK).
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    with _job_statuses_lock:
        _job_statuses[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "filters": {
                "country": country,
                "instrument_types": instrument_types,
                "skip_existing": skip_existing,
                "limit": limit
            },
            "total_symbols": None,
            "processed": 0,
            "successful": 0,
            "failed": 0
        }
    
    # Start background thread (logs will be visible in terminal)
    thread = threading.Thread(
        target=_run_validation_job,
        args=(job_id, limit, country, instrument_types, skip_existing),
        daemon=True  # Dies when main process dies
    )
    thread.start()
    
    _logger.info(f"Started validation job {job_id} with filters: country={country}, types={instrument_types}, limit={limit}")
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": "started",
        "message": "Validation job started successfully",
        "check_status_url": f"/api/validation/analytics/job-status/{job_id}",
        "filters_applied": {
            "country": country,
            "instrument_types": instrument_types,
            "skip_existing": skip_existing,
            "limit": limit
        }
    }, status_code=200)


@router.get("/validation/analytics/job-status/{job_id}")
def get_job_status(
    job_id: str,
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Get status of a validation job.
    
    Returns current progress, results, or error information for a running or completed job.
    
    Job statuses:
    - "pending": Job created but not started
    - "running": Job is currently processing
    - "completed": Job finished successfully  
    - "failed": Job encountered an error
    """
    with _job_statuses_lock:
        if job_id not in _job_statuses:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = _job_statuses[job_id].copy()
    
    return JSONResponse(content=job_info)


@router.get("/validation/analytics/jobs")
def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(20, description="Maximum number of jobs to return"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    List recent validation jobs.
    
    Useful for monitoring and debugging validation job history.
    """
    with _job_statuses_lock:
        jobs = list(_job_statuses.values())
    
    # Filter by status if specified
    if status:
        jobs = [job for job in jobs if job.get("status") == status]
    
    # Sort by creation time (newest first) and limit
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    jobs = jobs[:limit]
    
    return JSONResponse(content={
        "jobs": jobs,
        "total": len(jobs)
    })


@router.post("/validation/analytics/validate-missing")
def validate_missing_symbols(
    duplicates_only: bool = Query(False, description="Only validate symbols that have duplicates (same symbol name, different countries)"),
    revalidate_duplicates: bool = Query(True, description="Also revalidate existing symbols that have duplicates to fix data confusion"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Validate symbols that are missing from validation_flags table or have potential data confusion.
    
    This endpoint finds:
    1. Symbols in price_series that don't exist in validation_flags 
    2. If duplicates_only=True: only symbols that have duplicates (same name, different countries)
    3. If revalidate_duplicates=True: also revalidate existing symbols with same names to fix data confusion
    
    Example issue: AAPL Canada might be using US AAPL data but recorded as Canada validation.
    This endpoint will revalidate both US AAPL and Canada AAPL to ensure each uses correct country data.
    """
    import uuid
    
    job_id = str(uuid.uuid4())
    
    def _run_missing_validation_job():
        """Background job to validate missing/duplicate symbols"""
        from services.validation_integration import process_ticker_validation
        
        session = get_db_session()
        _logger = logging.getLogger(__name__)
        
        try:
            symbols_to_validate = []
            
            # Step 1: Find missing symbols
            if duplicates_only:
                # Find symbols with duplicates (same symbol name in different countries)
                duplicate_symbols_query = session.query(PriceSeries.symbol).group_by(PriceSeries.symbol).having(func.count(PriceSeries.symbol) > 1).subquery()
                
                missing_symbols_query = session.query(PriceSeries).filter(
                    PriceSeries.symbol.in_(select(duplicate_symbols_query.c.symbol))
                ).outerjoin(ValidationFlags, and_(
                    ValidationFlags.symbol == PriceSeries.symbol,
                    ValidationFlags.country == PriceSeries.country
                )).filter(ValidationFlags.id == None)
                
                _logger.info("Looking for duplicate symbols missing validation")
            else:
                # Find all symbols in price_series that don't have validation_flags
                missing_symbols_query = session.query(PriceSeries).outerjoin(ValidationFlags, and_(
                    ValidationFlags.symbol == PriceSeries.symbol,
                    ValidationFlags.country == PriceSeries.country
                )).filter(ValidationFlags.id == None)
                
                _logger.info(f"DEBUG: Looking for all missing validations...")
            
            missing_symbols = missing_symbols_query.all()
            symbols_to_validate.extend(missing_symbols)
            _logger.info(f"STATS: Found {len(missing_symbols)} symbols missing validation")
            
            # Step 2: Find duplicates for revalidation (if enabled)
            duplicate_symbols_for_revalidation = []
            if revalidate_duplicates and missing_symbols:
                # Get all symbol names that have missing validations
                missing_symbol_names = set([s.symbol for s in missing_symbols])
                
                # Find all symbols (including validated ones) that share these names
                duplicate_symbols_for_revalidation = session.query(PriceSeries).filter(
                    PriceSeries.symbol.in_(missing_symbol_names)
                ).all()
                
                # Remove missing symbols (already in symbols_to_validate) and keep only existing validated ones
                existing_duplicates = [s for s in duplicate_symbols_for_revalidation if s not in missing_symbols]
                symbols_to_validate.extend(existing_duplicates)
                
                _logger.info(f"PROCESSING: Found {len(existing_duplicates)} existing symbols to revalidate for data confusion fix")
                
                # Delete existing validation_flags for these symbols to force revalidation
                if existing_duplicates:
                    duplicate_symbol_country_pairs = [(s.symbol, s.country) for s in existing_duplicates]
                    for symbol, country in duplicate_symbol_country_pairs:
                        deleted_count = session.query(ValidationFlags).filter(
                            ValidationFlags.symbol == symbol,
                            ValidationFlags.country == country
                        ).delete()
                        if deleted_count > 0:
                            _logger.info(f"DELETED  Deleted existing validation for {symbol} ({country}) to force revalidation")
                    session.commit()
            
            total_to_validate = len(symbols_to_validate)
            
            if total_to_validate == 0:
                _logger.info("SUCCESS: No symbols found for validation")
                return
                
            # Log some examples
            for i, symbol in enumerate(symbols_to_validate[:5]):
                _logger.info(f"INFO Symbol to validate #{i+1}: {symbol.symbol} ({symbol.country}) - {symbol.name}")
            
            if total_to_validate > 5:
                _logger.info(f"... and {total_to_validate - 5} more")
                
            # Process symbols
            success_count = 0
            error_count = 0
            
            for i, price_series in enumerate(symbols_to_validate):
                try:
                    _logger.info(f"PROCESSING: Processing {i+1}/{total_to_validate}: {price_series.symbol} ({price_series.country})")
                    
                    # Prepare validation data in expected format
                    validation_data = {
                        'symbol': price_series.symbol,
                        'country': price_series.country,
                        'name': price_series.name,
                        'exchange': price_series.exchange,
                        'instrument_type': price_series.instrument_type or 'Unknown',
                        'currency': price_series.currency,
                        'isin': price_series.isin
                    }
                    
                    # Process validation
                    result = process_ticker_validation(
                        symbol=price_series.symbol,
                        validation_data=validation_data,
                        country=price_series.country,
                        db_session=session
                    )
                    
                    if result and result.valid is not None:
                        success_count += 1
                        status = "SUCCESS: Valid" if result.valid else "ERROR: Invalid"
                        _logger.info(f"   → {status}")
                    else:
                        error_count += 1
                        _logger.warning(f"   → ⚠️ No result returned")
                        
                except Exception as e:
                    error_count += 1
                    _logger.error(f"   → CRITICAL ERROR: Error: {str(e)}")
                    
            _logger.info(f"RESULT: Validation completed: {success_count} success, {error_count} errors, {total_to_validate} total")
            
        except Exception as e:
            _logger.error(f"CRITICAL ERROR: Missing validation job failed: {str(e)}")
        finally:
            session.close()
    
    # Start background job
    import threading
    thread = threading.Thread(target=_run_missing_validation_job, daemon=True)
    thread.start()
    
    return JSONResponse({
        "job_id": job_id,
        "status": "started",
        "message": f"Comprehensive validation job started. Check logs for progress.",
        "duplicates_only": duplicates_only,
        "revalidate_duplicates": revalidate_duplicates,
        "description": "Will validate missing symbols and revalidate duplicates to fix data confusion issues like AAPL Canada using US data"
    })


@router.get("/validation/analytics/diagnose-symbol/{symbol}")
def diagnose_symbol_data(
    symbol: str,
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Diagnose data confusion for a specific symbol across countries.
    
    Shows all price_series records for a symbol to identify data mismatches.
    """
    session = get_db_session()
    try:
        # Get all price_series records for this symbol
        price_records = session.query(PriceSeries).filter(
            PriceSeries.symbol == symbol
        ).all()
        
        # Get all validation_flags for this symbol  
        validation_records = session.query(ValidationFlags).filter(
            ValidationFlags.symbol == symbol
        ).all()
        
        price_data = []
        for record in price_records:
            price_data.append({
                "id": record.id,
                "symbol": record.symbol,
                "country": record.country,
                "name": record.name,
                "exchange": record.exchange,
                "instrument_type": record.instrument_type,
                "currency": record.currency,
                "isin": record.isin,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            })
            
        validation_data = []
        for record in validation_records:
            validation_data.append({
                "id": record.id,
                "symbol": record.symbol,
                "country": record.country,
                "valid": bool(record.valid),
                "as_of_date": record.as_of_date.isoformat() if record.as_of_date else None,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            })
        
        return JSONResponse({
            "symbol": symbol,
            "price_series_records": price_data,
            "validation_flags_records": validation_data,
            "analysis": {
                "total_countries_in_price_series": len(set([r.country for r in price_records])),
                "total_countries_in_validation": len(set([r.country for r in validation_records])),
                "countries_with_price_data": list(set([r.country for r in price_records])),
                "countries_with_validation": list(set([r.country for r in validation_records])),
                "potential_data_confusion": len(price_records) > 1 and len(set([r.name for r in price_records])) > 1
            }
        })
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error diagnosing symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error diagnosing symbol: {str(e)}")
    finally:
        session.close()


@router.post("/validation/analytics/force-revalidate")
def force_revalidate_specific_symbols(
    symbols: str = Query(..., description="Comma-separated list of symbols to force revalidate (e.g., 'AAPL,MSFT')"),
    countries: str = Query(None, description="Comma-separated list of countries to filter by (e.g., 'US,Canada'). If not specified, all countries for the symbols will be revalidated."),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Force revalidation of specific symbols to fix validation issues.
    
    This endpoint:
    1. Deletes existing validation_flags for specified symbols
    2. Forces fresh validation with correct country-specific data
    3. Returns immediate results (synchronous)
    
    Use this to fix issues like Canadian AAPL being incorrectly validated as US AAPL.
    """
    from services.validation_integration import process_ticker_validation
    
    session = get_db_session()
    _logger = logging.getLogger(__name__)
    
    try:
        # Parse input parameters
        symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        country_list = [c.strip() for c in countries.split(',') if c.strip()] if countries else None
        
        if not symbol_list:
            return JSONResponse({
                "error": "No symbols provided",
                "symbols_requested": symbols
            }, status_code=400)
        
        _logger.info(f"PROCESSING: Force revalidating symbols: {symbol_list}, countries: {country_list}")
        
        # Find symbols to revalidate
        query = session.query(PriceSeries).filter(PriceSeries.symbol.in_(symbol_list))
        if country_list:
            query = query.filter(PriceSeries.country.in_(country_list))
        
        symbols_to_revalidate = query.all()
        
        if not symbols_to_revalidate:
            return JSONResponse({
                "error": "No matching symbols found in price_series",
                "symbols_requested": symbol_list,
                "countries_requested": country_list
            }, status_code=404)
        
        _logger.info(f"STATS: Found {len(symbols_to_revalidate)} symbols to revalidate")
        
        # Delete existing validation_flags for these symbols
        deleted_count = 0
        for price_series in symbols_to_revalidate:
            deleted = session.query(ValidationFlags).filter(
                ValidationFlags.symbol == price_series.symbol,
                ValidationFlags.country == price_series.country
            ).delete()
            deleted_count += deleted
            if deleted > 0:
                _logger.info(f"DELETED  Deleted existing validation for {price_series.symbol} ({price_series.country})")
        
        session.commit()
        _logger.info(f"DELETED  Total deleted validation records: {deleted_count}")
        
        # Force revalidation
        results = []
        success_count = 0
        error_count = 0
        
        for price_series in symbols_to_revalidate:
            try:
                _logger.info(f"PROCESSING: Force validating {price_series.symbol} ({price_series.country})")
                
                # CRITICAL: Log what data we're using for validation
                _logger.info(f"   STATS: Using metadata: {price_series.name} - {price_series.exchange} - {price_series.isin}")
                
                # STEP 1: Load actual time series data with correct country-specific suffix
                from services.prices import load_prices
                from utils.common import _eodhd_suffix_for
                
                try:
                    # Build country-specific symbol (e.g., AAPL.TO for Canada, AAPL for US)
                    suffix = _eodhd_suffix_for(price_series.exchange, price_series.country)
                    symbol_with_suffix = f"{price_series.symbol}{suffix}"
                    
                    _logger.info(f"   LOADING Loading data for {symbol_with_suffix} (country: {price_series.country})")
                    
                    price_data = load_prices(symbol_with_suffix)
                    _logger.info(f"   SUCCESS: Loaded time series for {symbol_with_suffix}")
                    
                    # Log key info about loaded data
                    if isinstance(price_data, dict):
                        success = price_data.get("success", False)
                        years = price_data.get("summary", {}).get("years") if price_data.get("summary") else None
                        _logger.info(f"   DATA Price data: success={success}, years={years}")
                        
                        # CRITICAL: Log if insufficient history
                        if years is not None and years < 5.0:  # Example threshold
                            _logger.warning(f"   ⚠️  Insufficient history: {years} years < 5 year minimum")
                    
                except Exception as e:
                    _logger.error(f"   CRITICAL ERROR: Failed to load prices for {symbol_with_suffix}: {str(e)}")
                    price_data = {
                        "success": False,
                        "error": str(e),
                        "code": "load_error"
                    }
                
                # STEP 2: Process validation with real time series data
                result = process_ticker_validation(
                    symbol=price_series.symbol,
                    validation_data=price_data,  # SUCCESS: Now using real time series data!
                    country=price_series.country,
                    db_session=session
                )
                
                if result:
                    success_count += 1
                    status = "SUCCESS: Valid" if result.valid else "ERROR: Invalid"
                    _logger.info(f"   → {status}")
                    
                    # Parse validation summary to get actual validation details
                    summary_data = {}
                    if hasattr(result, 'validation_summary') and result.validation_summary:
                        try:
                            import json
                            summary_data = json.loads(result.validation_summary)
                        except:
                            pass
                    
                    results.append({
                        "symbol": price_series.symbol,
                        "country": price_series.country,
                        "valid": result.valid,
                        "years_actual": summary_data.get("years_actual"),
                        "has_history_error": summary_data.get("has_history_error"),
                        "validation_summary": summary_data
                    })
                else:
                    error_count += 1
                    _logger.error(f"   → CRITICAL ERROR: No result returned")
                    results.append({
                        "symbol": price_series.symbol,
                        "country": price_series.country,
                        "error": "No validation result returned"
                    })
                    
            except Exception as e:
                error_count += 1
                _logger.error(f"   → CRITICAL ERROR: Error: {str(e)}")
                results.append({
                    "symbol": price_series.symbol,
                    "country": price_series.country,
                    "error": str(e)
                })
        
        _logger.info(f"RESULT: Force revalidation completed: {success_count} success, {error_count} errors")
        
        return JSONResponse({
            "status": "completed",
            "summary": {
                "symbols_requested": symbol_list,
                "countries_requested": country_list,
                "symbols_found": len(symbols_to_revalidate),
                "deleted_validations": deleted_count,
                "success_count": success_count,
                "error_count": error_count
            },
            "results": results
        })
        
    except Exception as e:
        _logger.error(f"CRITICAL ERROR: Force revalidation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Force revalidation failed: {str(e)}")
    finally:
        session.close()


@router.get("/validation/analytics/whatif-years")
def whatif_years_analysis(
    years_threshold: float = Query(..., description="New years threshold to analyze (e.g., 5.0, 7.5, 15.0)"),
    country: str = Query("", description="Filter by country (optional)"),
    instrument_types: str = Query("", description="Filter by instrument types, comma-separated (optional)"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    What-if analysis: How many symbols would be valid with a different years threshold?
    
    Analyzes existing validation_summary data without re-running validation.
    Changes only the years_required criteria and recalculates validity.
    
    Example: If current threshold is 10 years, what would happen with 5 years?
    """
    session = get_db_session()
    _logger = logging.getLogger(__name__)
    
    try:
        _logger.info(f"ANALYSIS What-if analysis: years_threshold={years_threshold}, country='{country}', instrument_types='{instrument_types}'")
        
        # Parse instrument types
        instrument_types_list = []
        if instrument_types:
            instrument_types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
        
        # Base query for ValidationFlags with validation_summary
        vf_query = session.query(ValidationFlags).filter(
            ValidationFlags.validation_summary.isnot(None)
        )
        
        # Apply country filter
        if country:
            vf_query = vf_query.filter(ValidationFlags.country == country)
            
        # Apply instrument type filter by joining with PriceSeries
        if instrument_types_list:
            vf_query = vf_query.join(
                PriceSeries, 
                and_(
                    PriceSeries.symbol == ValidationFlags.symbol,
                    PriceSeries.country == ValidationFlags.country
                )
            ).filter(PriceSeries.instrument_type.in_(instrument_types_list))
        
        all_records = vf_query.all()
        _logger.info(f"STATS: Found {len(all_records)} records with validation_summary")
        
        if not all_records:
            return JSONResponse({
                "years_threshold": years_threshold,
                "total_analyzed": 0,
                "would_be_valid": 0,
                "would_be_invalid": 0,
                "would_be_valid_percentage": 0.0,
                "would_be_invalid_percentage": 0.0,
                "current_valid": 0,
                "current_invalid": 0,
                "improvement": {
                    "additional_valid": 0,
                    "percentage_change": 0.0
                },
                "message": "No records found with the specified filters"
            })
        
        # Analyze each record with new years threshold
        current_valid = 0
        current_invalid = 0
        would_be_valid = 0
        would_be_invalid = 0
        
        # Debug counters
        debug_stats = {
            'valid_with_null_years': 0,
            'invalid_with_years_data': 0,
            'would_improve': 0,
            'would_decline': 0,
            'sample_invalid_with_years': []
        }
        
        for record in all_records:
            validation_summary = record.validation_summary
            
            # Current validity
            if record.valid == 1:
                current_valid += 1
            else:
                current_invalid += 1
            
            # What-if validity calculation
            would_be_valid_flag = _calculate_whatif_validity(validation_summary, years_threshold, record.valid == 1)
            
            if would_be_valid_flag:
                would_be_valid += 1
            else:
                would_be_invalid += 1
                
            # Debug logging
            if validation_summary:
                try:
                    if isinstance(validation_summary, str):
                        import json
                        summary = json.loads(validation_summary)
                    else:
                        summary = validation_summary
                        
                    years_actual = summary.get('years_actual')
                    
                    if record.valid == 1 and years_actual is None:
                        debug_stats['valid_with_null_years'] += 1
                    elif record.valid == 0 and years_actual is not None:
                        debug_stats['invalid_with_years_data'] += 1
                        if len(debug_stats['sample_invalid_with_years']) < 5:
                            debug_stats['sample_invalid_with_years'].append({
                                'symbol': record.symbol,
                                'country': record.country, 
                                'years_actual': years_actual,
                                'has_history_error': summary.get('has_history_error', False),
                                'has_structural_issues': summary.get('has_structural_issues', False)
                            })
                        
                        # Check if this symbol would improve with new threshold (NEW LOGIC)
                        has_structural_issues = summary.get('has_structural_issues', False)
                        has_enough_years = years_actual >= years_threshold
                        
                        if has_enough_years and not has_structural_issues:
                            debug_stats['would_improve'] += 1
                            
                except:
                    pass
        
        _logger.info(f"DEBUG: Debug stats: valid_with_null_years={debug_stats['valid_with_null_years']}, invalid_with_years_data={debug_stats['invalid_with_years_data']}, would_improve={debug_stats['would_improve']}")
        if debug_stats['sample_invalid_with_years']:
            _logger.info(f"STATS: Sample invalid symbols with years data: {debug_stats['sample_invalid_with_years'][:3]}")
        
        total_analyzed = len(all_records)
        would_be_valid_percentage = (would_be_valid / total_analyzed * 100) if total_analyzed > 0 else 0.0
        would_be_invalid_percentage = (would_be_invalid / total_analyzed * 100) if total_analyzed > 0 else 0.0
        
        additional_valid = would_be_valid - current_valid
        percentage_change = ((would_be_valid - current_valid) / current_valid * 100) if current_valid > 0 else 0.0
        
        result = {
            "years_threshold": years_threshold,
            "total_analyzed": total_analyzed,
            "would_be_valid": would_be_valid,
            "would_be_invalid": would_be_invalid,
            "would_be_valid_percentage": round(would_be_valid_percentage, 2),
            "would_be_invalid_percentage": round(would_be_invalid_percentage, 2),
            "current_stats": {
                "current_valid": current_valid,
                "current_invalid": current_invalid,
                "current_valid_percentage": round((current_valid / total_analyzed * 100) if total_analyzed > 0 else 0.0, 2)
            },
            "improvement": {
                "additional_valid": additional_valid,
                "percentage_change": round(percentage_change, 2),
                "direction": "improvement" if additional_valid > 0 else "decline" if additional_valid < 0 else "no_change"
            },
            "filter_applied": {
                "country": country or "All countries",
                "instrument_types": instrument_types_list if instrument_types_list else "All types"
            }
        }
        
        _logger.info(f"RESULT: What-if result: {would_be_valid}/{total_analyzed} would be valid ({would_be_valid_percentage:.1f}%), improvement: {additional_valid:+d}")
        return JSONResponse(result)
        
    except Exception as e:
        _logger.error(f"CRITICAL ERROR: What-if analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"What-if analysis failed: {str(e)}")
    finally:
        session.close()


def _calculate_whatif_validity(validation_summary, new_years_threshold, current_valid_flag):
    """
    Calculate if a symbol would be valid with new years threshold.
    
    Logic:
    - Valid symbols (valid=1): years_actual=null, assume they have plenty of history (20+ years)
    - Invalid symbols (valid=0): years_actual is populated, use exact value for comparison
    """
    # Assume valid symbols have plenty of history (conservative estimate)
    ASSUMED_YEARS_FOR_VALID_SYMBOLS = 20.0
    
    if not validation_summary:
        # If no validation_summary but symbol is currently valid,
        # assume it has plenty of history and would remain valid for reasonable thresholds
        if current_valid_flag:
            return new_years_threshold <= ASSUMED_YEARS_FOR_VALID_SYMBOLS
        else:
            # Invalid symbol without validation_summary - can't determine, assume stays invalid
            return False
    
    # Parse validation_summary JSON if it's a string  
    if isinstance(validation_summary, str):
        import json
        try:
            summary = json.loads(validation_summary)
        except:
            return False
    else:
        summary = validation_summary
    
    years_actual = summary.get('years_actual')
    has_history_error = summary.get('has_history_error', False)
    has_structural_issues = summary.get('has_structural_issues', False)
    
    if years_actual is None:
        # Symbol has validation_summary but no years_actual
        if current_valid_flag:
            # Valid symbol - assume it has plenty of history
            # Would be valid unless new threshold is too high OR has structural issues
            # NOTE: We ignore has_history_error because we're changing the history requirement
            result = new_years_threshold <= ASSUMED_YEARS_FOR_VALID_SYMBOLS and not has_structural_issues
            return result
        else:
            # Invalid symbol without years_actual - probably has structural issues
            # Can only be fixed if no structural issues (history error doesn't matter for what-if)
            return not has_structural_issues
    else:
        # Symbol has recorded years_actual - use exact comparison
        # Symbol would be valid if: has enough years AND no structural issues
        # NOTE: We ignore has_history_error because that's exactly what we're testing
        has_enough_years = years_actual >= new_years_threshold
        result = has_enough_years and not has_structural_issues
        return result


@router.get("/validation/analytics/debug-duplicates")
def debug_duplicate_symbols(
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """Debug endpoint to check duplicate detection SQL"""
    session = get_db_session()
    _logger = logging.getLogger(__name__)
    
    try:
        _logger.info("DEBUG: Debugging duplicate symbol detection...")
        
        # Simple query to see what we have
        all_symbols = session.query(
            PriceSeries.symbol,
            PriceSeries.country
        ).order_by(PriceSeries.symbol).all()
        
        # Manual duplicate detection
        from collections import defaultdict
        symbol_countries = defaultdict(list)
        
        for symbol, country in all_symbols:
            symbol_countries[symbol].append(country)
        
        # Find duplicates manually
        manual_duplicates = []
        for symbol, countries in symbol_countries.items():
            if len(set(countries)) > 1:  # More than one unique country
                manual_duplicates.append({
                    "symbol": symbol,
                    "countries": list(set(countries)),
                    "count": len(set(countries))
                })
        
        _logger.info(f"STATS: Manual count found {len(manual_duplicates)} duplicates")
        
        # Now try the SQL query
        sql_duplicates = session.query(
            PriceSeries.symbol,
            func.count(func.distinct(PriceSeries.country)).label('country_count')
        ).group_by(PriceSeries.symbol).having(
            func.count(func.distinct(PriceSeries.country)) > 1
        ).all()
        
        _logger.info(f"STATS: SQL query found {len(sql_duplicates)} duplicates")
        
        sql_results = [(row.symbol, row.country_count) for row in sql_duplicates]
        
        return JSONResponse({
            "manual_duplicates_count": len(manual_duplicates),
            "sql_duplicates_count": len(sql_results),
            "manual_duplicates": manual_duplicates[:20],  # First 20
            "sql_duplicates": sql_results[:20],  # First 20
            "sample_aapl_check": {
                "aapl_countries": symbol_countries.get("AAPL", []),
                "aapl_unique_countries": list(set(symbol_countries.get("AAPL", []))),
                "aapl_country_count": len(set(symbol_countries.get("AAPL", [])))
            }
        })
        
    except Exception as e:
        _logger.error(f"CRITICAL ERROR: Debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
    finally:
        session.close()


@router.post("/validation/analytics/fix-all-duplicates")
def fix_all_duplicate_symbols(
    dry_run: bool = Query(False, description="If true, only shows what would be fixed without making changes"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    Find and fix ALL duplicate symbols (symbols that exist in multiple countries).
    
    This endpoint automatically:
    1. Finds all symbols that exist in multiple countries (e.g., AAPL in US+Canada)
    2. Deletes existing incorrect validation_flags for these symbols
    3. Revalidates each with correct country-specific data (AAPL.TO vs AAPL)
    4. Returns comprehensive statistics
    
    Much more efficient than revalidating the entire universe!
    """
    from services.validation_integration import process_ticker_validation
    from services.prices import load_prices
    from utils.common import _eodhd_suffix_for
    
    session = get_db_session()
    _logger = logging.getLogger(__name__)
    
    try:
        # STEP 1: Find all symbols that appear in multiple countries
        _logger.info("DEBUG: Finding all duplicate symbols (symbols in multiple countries)...")
        
        # Find symbols that appear in multiple countries
        duplicate_symbols_query = session.query(
            PriceSeries.symbol,
            func.count(func.distinct(PriceSeries.country)).label('country_count')
        ).group_by(PriceSeries.symbol).having(
            func.count(func.distinct(PriceSeries.country)) > 1
        ).all()
        
        # Get detailed info for each duplicate symbol
        duplicate_symbols = []
        for row in duplicate_symbols_query:
            # Get all countries for this symbol
            countries = session.query(func.distinct(PriceSeries.country)).filter(
                PriceSeries.symbol == row.symbol
            ).all()
            countries_list = [c[0] for c in countries]
            duplicate_symbols.append((row.symbol, row.country_count, ','.join(countries_list)))
        
        _logger.info(f"STATS: Found {len(duplicate_symbols)} duplicate symbols")
        
        # Log some examples
        for symbol, count, countries in duplicate_symbols[:10]:
            _logger.info(f"   INFO {symbol}: {count} countries ({countries})")
        
        if len(duplicate_symbols) > 10:
            _logger.info(f"   ... and {len(duplicate_symbols) - 10} more")
        
        # STEP 2: Get all price_series records for duplicate symbols
        duplicate_symbol_names = [symbol for symbol, _, _ in duplicate_symbols]
        
        all_duplicate_records = session.query(PriceSeries).filter(
            PriceSeries.symbol.in_(duplicate_symbol_names)
        ).order_by(PriceSeries.symbol, PriceSeries.country).all()
        
        _logger.info(f"STATS: Found {len(all_duplicate_records)} total records to process")
        
        if dry_run:
            # Just return what would be processed
            results = []
            for record in all_duplicate_records:
                suffix = _eodhd_suffix_for(record.exchange, record.country)
                symbol_with_suffix = f"{record.symbol}{suffix}"
                
                results.append({
                    "symbol": record.symbol,
                    "country": record.country,
                    "name": record.name,
                    "exchange": record.exchange,
                    "would_load_symbol": symbol_with_suffix,
                    "action": "would_revalidate"
                })
            
            return JSONResponse({
                "status": "dry_run_completed",
                "duplicate_symbols_found": len(duplicate_symbols),
                "total_records_to_process": len(all_duplicate_records),
                "duplicate_symbols": duplicate_symbols,
                "detailed_plan": results[:50],  # Limit for readability
                "message": f"Would process {len(all_duplicate_records)} duplicate records. Run with dry_run=false to execute."
            })
        
        # STEP 3: Delete existing validation_flags for all duplicate symbols
        _logger.info("DELETED  Deleting existing validation flags for all duplicate symbols...")
        
        deleted_count = 0
        for record in all_duplicate_records:
            deleted = session.query(ValidationFlags).filter(
                ValidationFlags.symbol == record.symbol,
                ValidationFlags.country == record.country
            ).delete()
            deleted_count += deleted
            
        session.commit()
        _logger.info(f"DELETED  Deleted {deleted_count} existing validation records")
        
        # STEP 4: Revalidate all duplicate symbols with multiprocessing
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        workers = int(os.getenv('EXP_REPROCESS_WORKERS', 8))
        _logger.info(f"PROCESSING: Revalidating all duplicate symbols with {workers} workers...")
        
        def _process_duplicate_symbol(record_data):
            """Process a single duplicate symbol with correct country-specific data"""
            record_symbol, record_country, record_name, record_exchange = record_data
            
            # Create new session for this thread
            thread_session = get_db_session()
            thread_logger = logging.getLogger(f"{__name__}.worker")
            
            try:
                # Build correct symbol with country suffix
                suffix = _eodhd_suffix_for(record_exchange, record_country)
                symbol_with_suffix = f"{record_symbol}{suffix}"
                
                thread_logger.info(f"LOADING [{record_symbol}] Loading {symbol_with_suffix} for {record_country}")
                
                # Load correct time series data
                try:
                    price_data = load_prices(symbol_with_suffix)
                    
                    if isinstance(price_data, dict):
                        success = price_data.get("success", False)
                        years = price_data.get("summary", {}).get("years") if price_data.get("summary") else None
                        
                        if not success:
                            thread_logger.warning(f"   ⚠️  [{record_symbol}] Price load failed: {price_data.get('error', 'unknown error')}")
                        elif years and years < 5.0:
                            thread_logger.info(f"   ⚠️  [{record_symbol}] Insufficient history: {years} years")
                        
                except Exception as e:
                    thread_logger.error(f"   CRITICAL ERROR: [{record_symbol}] Price load error: {str(e)}")
                    price_data = {
                        "success": False,
                        "error": str(e),
                        "code": "load_error"
                    }
                
                # Process validation
                result = process_ticker_validation(
                    symbol=record_symbol,
                    validation_data=price_data,
                    country=record_country,
                    db_session=thread_session
                )
                
                if result:
                    status = "SUCCESS: Valid" if result.valid else "ERROR: Invalid"
                    thread_logger.info(f"   → [{record_symbol}] {status}")
                    
                    # Parse validation summary 
                    summary_data = {}
                    if hasattr(result, 'validation_summary') and result.validation_summary:
                        try:
                            import json
                            summary_data = json.loads(result.validation_summary)
                        except:
                            pass
                    
                    return {
                        "symbol": record_symbol,
                        "country": record_country,
                        "name": record_name,
                        "exchange": record_exchange,
                        "loaded_symbol": symbol_with_suffix,
                        "valid": result.valid,
                        "years_actual": summary_data.get("years_actual"),
                        "has_history_error": summary_data.get("has_history_error"),
                        "status": "success"
                    }
                else:
                    thread_logger.error(f"   → CRITICAL ERROR: [{record_symbol}] No result returned")
                    return {
                        "symbol": record_symbol,
                        "country": record_country,
                        "error": "No validation result returned",
                        "status": "error"
                    }
                    
            except Exception as e:
                thread_logger.error(f"   → CRITICAL ERROR: [{record_symbol}] Processing error: {str(e)}")
                return {
                    "symbol": record_symbol,
                    "country": record_country,
                    "error": str(e),
                    "status": "error"
                }
            finally:
                thread_session.close()
        
        # Prepare data for multiprocessing
        records_data = [
            (record.symbol, record.country, record.name, record.exchange)
            for record in all_duplicate_records
        ]
        
        results = []
        success_count = 0
        error_count = 0
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(_process_duplicate_symbol, record_data): record_data
                for record_data in records_data
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_record)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        error_count += 1
                    
                    # Progress update every 50 completions
                    if (i + 1) % 50 == 0 or (i + 1) == len(records_data):
                        _logger.info(f"STATS: Progress: {i+1}/{len(records_data)} completed ({success_count} success, {error_count} errors)")
                        
                except Exception as e:
                    error_count += 1
                    record_data = future_to_record[future]
                    _logger.error(f"CRITICAL ERROR: Future processing error for {record_data[0]}: {str(e)}")
                    results.append({
                        "symbol": record_data[0],
                        "country": record_data[1],
                        "error": f"Future error: {str(e)}",
                        "status": "error"
                    })
        
        _logger.info(f"RESULT: Duplicate symbols fix completed: {success_count} success, {error_count} errors")
        
        return JSONResponse({
            "status": "completed",
            "summary": {
                "duplicate_symbols_found": len(duplicate_symbols),
                "total_records_processed": len(all_duplicate_records),
                "validation_flags_deleted": deleted_count,
                "success_count": success_count,
                "error_count": error_count
            },
            "duplicate_symbols": duplicate_symbols,
            "results": results,
            "message": f"Successfully fixed {success_count} duplicate symbol validations!"
        })
        
    except Exception as e:
        _logger.error(f"CRITICAL ERROR: Duplicate symbols fix failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Duplicate symbols fix failed: {str(e)}")
    finally:
        session.close()


@router.get("/validation/analytics/validate-all-sync")
def validate_all_symbols_sync(
    limit: int = Query(0, description="Maximum number of symbols to process (0 = all symbols)"),
    country: Optional[str] = Query(None, description="Filter symbols by country"),
    instrument_types: Optional[str] = Query(None, description="Comma-separated list of instrument types to filter by"),
    skip_existing: bool = Query(True, description="Skip symbols that already have validation_flags records"),
    _auth: None = Depends(_basic_auth_if_configured),
) -> JSONResponse:
    """
    SYNCHRONOUS validation with REAL-TIME LOGS visible in terminal.
    
    ⚠️  This endpoint BLOCKS until completion but shows all progress logs.
    ⚠️  Use this if you need to see logs in real-time in terminal.
    ⚠️  For large datasets, use the async version instead.
    
    Examples:
    - GET /api/validation/analytics/validate-all-sync?limit=100&country=US
    - GET /api/validation/analytics/validate-all-sync?country=Canada
    
    You will see ALL logs in terminal in real-time.
    """
    from services.prices import load_prices
    from services.validation_integration import process_ticker_validation
    from core.db import get_db_session
    from core.models import PriceSeries, ValidationFlags
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Get worker count from environment
    try:
        max_workers = int(os.getenv("EXP_REPROCESS_WORKERS", "8"))
    except (ValueError, TypeError):
        max_workers = 8
    
    session = get_db_session()
    
    try:
        _logger.info("=" * 60)
        _logger.info("STARTING: STARTING SYNCHRONOUS VALIDATION WITH REAL-TIME LOGS")
        _logger.info("=" * 60)
        
        # Parse instrument types filter
        types_list = []
        if instrument_types:
            types_list = [t.strip() for t in instrument_types.split(',') if t.strip()]
        
        # Build query for symbols to process
        query = session.query(PriceSeries).filter(PriceSeries.symbol.isnot(None))
        
        if country:
            query = query.filter(PriceSeries.country == country)
            _logger.info(f"LOADING Filtering by country: {country}")
            
        if types_list:
            query = query.filter(PriceSeries.instrument_type.in_(types_list))
            _logger.info(f"DATA Filtering by instrument types: {', '.join(types_list)}")
        
        if skip_existing:
            # Skip symbols that already have validation records
            existing_symbols = session.query(ValidationFlags.symbol).distinct()
            existing_set = {symbol for (symbol,) in existing_symbols.all()}
            if existing_set:
                query = query.filter(~PriceSeries.symbol.in_(existing_set))
                _logger.info(f"⏭️  Skipping {len(existing_set)} symbols with existing validation")
        
        # Apply limit if specified
        if limit > 0:
            query = query.limit(limit)
            _logger.info(f"LIMIT: Limiting to {limit} symbols")
        
        # Get all symbols to process
        symbols_to_process = [row.symbol for row in query.all()]
        total_symbols = len(symbols_to_process)
        
        if total_symbols == 0:
            _logger.warning("ERROR: No symbols to process with current filters!")
            return JSONResponse(content={
                "message": "No symbols to process",
                "total_symbols": 0,
                "filters_applied": {
                    "country": country,
                    "instrument_types": instrument_types,
                    "skip_existing": skip_existing,
                    "limit": limit
                },
                "processed": 0,
                "successful": 0,
                "failed": 0
            })
        
        _logger.info(f"RESULT: Found {total_symbols} symbols to validate")
        _logger.info(f"⚙️  Using {max_workers} workers")
        _logger.info("=" * 60)
        
        # Process symbols with multiprocessing
        successful_count = 0
        failed_count = 0
        results = []
        
        def process_single_symbol(symbol_name):
            """Process a single symbol for validation"""
            local_session = get_db_session()
            try:
                _logger.info(f"PROCESSING: Processing: {symbol_name}")
                
                # Get country info from database FIRST to determine correct data source
                price_series_record = local_session.query(PriceSeries).filter(
                    PriceSeries.symbol == symbol_name
                ).first()
                symbol_country = price_series_record.country if price_series_record else None
                
                # Check if this symbol has duplicates (exists in multiple countries)
                duplicate_count = local_session.query(PriceSeries).filter(
                    PriceSeries.symbol == symbol_name
                ).count()
                
                # Load price data with correct country-specific suffix for duplicates
                try:
                    if duplicate_count > 1 and price_series_record:
                        # For duplicates, use country-specific suffix
                        from utils.common import _eodhd_suffix_for
                        suffix = _eodhd_suffix_for(price_series_record.exchange, symbol_country)
                        symbol_with_suffix = f"{symbol_name}{suffix}"
                        _logger.info(f"DUPLICATE DETECTED: Loading {symbol_with_suffix} for {symbol_country}")
                        price_data = load_prices(symbol_with_suffix)
                    else:
                        # For unique symbols, use symbol as-is
                        price_data = load_prices(symbol_name)
                    
                    _logger.info(f"SUCCESS: Loaded price data for {symbol_name}")
                except Exception as e:
                    price_data = {
                        "success": False,
                        "error": str(e),
                        "code": "load_error"
                    }
                    _logger.warning(f"⚠️  Price data load failed for {symbol_name}: {str(e)[:50]}...")
                
                # Process validation
                result = process_ticker_validation(
                    symbol=symbol_name,
                    validation_data=price_data,
                    country=symbol_country,
                    db_session=local_session
                )
                
                local_session.commit()
                
                # Log result
                if result and result.valid:
                    _logger.info(f"SUCCESS: VALID: {symbol_name} - validation successful")
                else:
                    _logger.warning(f"ERROR: INVALID: {symbol_name} - validation failed")
                
                # Convert ValidationFlags to dict for JSON serialization
                result_summary = {
                    "valid": result.valid if result else None,
                    "has_critical_issues": result.has_critical_issues if result else None,
                    "has_warnings": result.has_warnings if result else None,
                } if result else None
                
                return {
                    "symbol": symbol_name,
                    "status": "success",
                    "validation_summary": result_summary
                }
                
            except Exception as e:
                if local_session:
                    local_session.rollback()
                error_msg = str(e)
                _logger.error(f"CRITICAL ERROR: ERROR processing {symbol_name}: {error_msg}")
                return {
                    "symbol": symbol_name,
                    "status": "failed",
                    "error": error_msg
                }
            finally:
                if local_session:
                    local_session.close()
        
        # Execute with ThreadPoolExecutor
        _logger.info("STARTING: Starting parallel processing...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_single_symbol, symbol): symbol 
                for symbol in symbols_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Log progress after each symbol
                processed = successful_count + failed_count
                progress_pct = (processed / total_symbols) * 100
                _logger.info(f"STATS: Progress: {processed}/{total_symbols} ({progress_pct:.1f}%) | SUCCESS: {successful_count} success | ERROR: {failed_count} failed")
        
        # Summary statistics
        success_rate = (successful_count / total_symbols * 100) if total_symbols > 0 else 0
        
        _logger.info("=" * 60)
        _logger.info("SUCCESS: VALIDATION COMPLETED!")
        _logger.info(f"STATS: FINAL STATS: {total_symbols} total | {successful_count} success | {failed_count} failed")
        _logger.info(f"RESULT: SUCCESS RATE: {success_rate:.1f}%")
        _logger.info("=" * 60)
        
        response_data = {
            "message": f"Validation completed for {total_symbols} symbols",
            "filters_applied": {
                "country": country,
                "instrument_types": instrument_types,
                "skip_existing": skip_existing,
                "limit": limit
            },
            "processing_stats": {
                "total_symbols": total_symbols,
                "processed": successful_count + failed_count,
                "successful": successful_count, 
                "failed": failed_count,
                "success_rate": f"{success_rate:.1f}%",
                "workers_used": max_workers
            },
            "sample_results": results[:5] if results else [],
            "failed_symbols": [r["symbol"] for r in results if r["status"] == "failed"][:10]
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        _logger.error(f"CRITICAL ERROR: FATAL ERROR in validation: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    finally:
        session.close()
