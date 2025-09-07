"""
Validation Repository for handling validation-related database operations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from sqlalchemy.orm import Session

from repositories.base_repository import BaseRepository
from core.models import ValidationFlags, AnomalyReport, InsufficientDataEvent
from services.infrastructure.redis_cache_service import (
    get_cache_service, 
    CacheKeyType, 
    StandardCachePolicies
)
import logging

logger = logging.getLogger(__name__)


class ValidationRepository(BaseRepository[ValidationFlags]):
    """Repository for validation operations."""
    
    def __init__(self):
        super().__init__(ValidationFlags)
        self.cache_service = get_cache_service()
    
    def get_by_symbol(self, symbol: str) -> Optional[ValidationFlags]:
        """Get validation flags by symbol."""
        def query_func(session: Session) -> Optional[ValidationFlags]:
            return (
                session.query(ValidationFlags)
                .filter(ValidationFlags.symbol == symbol)
                .first()
            )
        
        return self.execute_query(query_func)
    
    def upsert_validation_flags(
        self,
        symbol: str,
        country: Optional[str] = None,
        validation_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert or update validation flags with cache invalidation."""
        def query_func(session: Session) -> bool:
            existing = (
                session.query(ValidationFlags)
                .filter(ValidationFlags.symbol == symbol)
                .first()
            )
            
            if existing:
                # Update existing
                if country:
                    existing.country = country
                if validation_data:
                    existing.valid = 1 if validation_data.get("success", False) else 0
                    existing.validation_summary = validation_data
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                from datetime import date
                new_flags = ValidationFlags(
                    symbol=symbol,
                    country=country,
                    as_of_date=date.today(),
                    valid=1 if (validation_data and validation_data.get("success", False)) else 0,
                    validation_summary=validation_data or {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(new_flags)
            
            session.commit()
            return True
        
        result = self.execute_query(query_func) or False
        
        # Invalidate related caches when validation flags change
        if result:
            self._invalidate_validation_caches()
        
        return result
    
    def _invalidate_validation_caches(self) -> None:
        """Invalidate all validation-related caches."""
        try:
            # Invalidate all validation flags caches
            pattern = f"{CacheKeyType.VALIDATION_FLAGS.value}:*"
            deleted_count = self.cache_service.invalidate_pattern(pattern)
            logger.debug(f"Invalidated {deleted_count} validation cache keys")
        except Exception as e:
            logger.warning(f"Failed to invalidate validation caches: {e}")
    
    def get_symbols_by_validation_status(
        self,
        valid: Optional[bool] = None,
        country: Optional[str] = None
    ) -> List[str]:
        """Get symbols filtered by validation status with Redis caching."""
        # Create cache key from parameters
        cache_key_parts = []
        if valid is not None:
            cache_key_parts.append(f"valid_{valid}")
        if country:
            cache_key_parts.append(f"country_{country}")
        cache_key_parts.append("validation_symbols")
        
        # Try cache first
        cached_result = self.cache_service.get(
            CacheKeyType.VALIDATION_FLAGS, 
            *cache_key_parts
        )
        if cached_result is not None:
            logger.debug(f"Cache hit for validation symbols: {cache_key_parts}")
            return cached_result
        
        # Execute database query
        def query_func(session: Session) -> List[str]:
            query = session.query(ValidationFlags.symbol)
            
            if valid is not None:
                query = query.filter(ValidationFlags.valid == (1 if valid else 0))
            
            if country:
                query = query.filter(ValidationFlags.country == country)
            
            return [s[0] for s in query.all()]
        
        result = self.execute_query(query_func) or []
        
        # Cache the result
        if result:
            success = self.cache_service.set(
                CacheKeyType.VALIDATION_FLAGS,
                result,
                StandardCachePolicies.VALIDATION_FLAGS,
                *cache_key_parts
            )
            if success:
                logger.debug(f"Cached {len(result)} validation symbols")
        
        return result
    
    def create_anomaly_report(
        self,
        symbol: str,
        anomaly_type: str,
        description: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create anomaly report."""
        def query_func(session: Session) -> bool:
            report = AnomalyReport(
                symbol=symbol,
                anomaly_type=anomaly_type,
                description=description,
                severity=severity,
                metadata=metadata or {},
                detected_at=datetime.utcnow(),
                is_resolved=False
            )
            session.add(report)
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def get_anomaly_reports(
        self,
        symbol: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        severity: Optional[str] = None
    ) -> List[AnomalyReport]:
        """Get anomaly reports with filters."""
        def query_func(session: Session) -> List[AnomalyReport]:
            query = session.query(AnomalyReport)
            
            if symbol:
                query = query.filter(AnomalyReport.symbol == symbol)
            
            if is_resolved is not None:
                query = query.filter(AnomalyReport.is_resolved == is_resolved)
            
            if severity:
                query = query.filter(AnomalyReport.severity == severity)
            
            return query.order_by(AnomalyReport.detected_at.desc()).all()
        
        return self.execute_query(query_func) or []
    
    def resolve_anomaly(self, anomaly_id: int, resolution_notes: Optional[str] = None) -> bool:
        """Mark anomaly as resolved."""
        def query_func(session: Session) -> bool:
            anomaly = session.query(AnomalyReport).filter(AnomalyReport.id == anomaly_id).first()
            
            if anomaly:
                anomaly.is_resolved = True
                anomaly.resolved_at = datetime.utcnow()
                if resolution_notes:
                    anomaly.resolution_notes = resolution_notes
                session.commit()
                return True
            return False
        
        return self.execute_query(query_func) or False
    
    def create_insufficient_data_event(
        self,
        symbol: str,
        reason: str,
        data_points_count: Optional[int] = None,
        required_count: Optional[int] = None
    ) -> bool:
        """Log insufficient data event."""
        def query_func(session: Session) -> bool:
            event = InsufficientDataEvent(
                symbol=symbol,
                reason=reason,
                data_points_count=data_points_count,
                required_count=required_count,
                detected_at=datetime.utcnow()
            )
            session.add(event)
            session.commit()
            return True
        
        return self.execute_query(query_func) or False
    
    def get_insufficient_data_events(
        self,
        symbol: Optional[str] = None,
        days_back: Optional[int] = 30
    ) -> List[InsufficientDataEvent]:
        """Get recent insufficient data events."""
        def query_func(session: Session) -> List[InsufficientDataEvent]:
            query = session.query(InsufficientDataEvent)
            
            if symbol:
                query = query.filter(InsufficientDataEvent.symbol == symbol)
            
            if days_back:
                from datetime import timedelta
                cutoff = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(InsufficientDataEvent.detected_at >= cutoff)
            
            return query.order_by(InsufficientDataEvent.detected_at.desc()).all()
        
        return self.execute_query(query_func) or []
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary with Redis caching."""
        # Try cache first
        cached_summary = self.cache_service.get(
            CacheKeyType.VALIDATION_FLAGS, 
            "summary"
        )
        if cached_summary is not None:
            logger.debug("Cache hit for validation summary")
            return cached_summary
        
        # Execute database query
        def query_func(session: Session) -> Dict[str, Any]:
            total_symbols = session.query(ValidationFlags).count()
            
            valid_symbols = (
                session.query(ValidationFlags)
                .filter(ValidationFlags.valid == 1)
                .count()
            )
            
            invalid_symbols = total_symbols - valid_symbols
            
            # Get recent anomalies count - simplified for testing
            recent_anomalies = 0  # TODO: Fix AnomalyReport queries
            
            return {
                "total_symbols": total_symbols,
                "valid_symbols": valid_symbols,
                "invalid_symbols": invalid_symbols,
                "recent_anomalies": recent_anomalies,
                "data_quality_percentage": round(
                    (valid_symbols / total_symbols) * 100, 2
                ) if total_symbols > 0 else 0.0
            }
        
        result = self.execute_query(query_func) or {}
        
        # Cache the result
        if result:
            success = self.cache_service.set(
                CacheKeyType.VALIDATION_FLAGS,
                result,
                StandardCachePolicies.VALIDATION_FLAGS,
                "summary"
            )
            if success:
                logger.debug(f"Cached validation summary: {result.get('total_symbols', 0)} symbols")
        
        return result
