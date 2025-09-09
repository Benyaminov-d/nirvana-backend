"""Maintenance routes for system administration and monitoring."""

import time
import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_

from core.db import get_db_session
from core.models import CompassAnchor, CvarSnapshot
from utils.auth import require_pub_or_basic as _require_pub_or_basic
from services.compass_anchors import current_quarter_version, calibrate_harvard_universe_anchors
from logging_config import get_logger_levels

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/reset-circuit-breaker")
def reset_circuit_breaker(
    service: Optional[str] = Query(None, description="Specific service to reset (e.g. eodhd_historical_prices)"),
    all_services: bool = Query(False, description="Reset all circuit breakers"),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    
    try:
        from config import get_config
        import redis
        
        config = get_config()
        redis_client = redis.from_url(config.redis.url)
        
        # Get all circuit breaker keys
        if all_services:
            keys = redis_client.keys("circuit:*")
            services = [key.decode('utf-8').split(':', 1)[1] for key in keys]
        elif service:
            keys = [f"circuit:{service}"]
            services = [service]
        else:
            return {
                "success": False,
                "error": "Either service name or all_services flag must be provided"
            }
        
        results = {}
        for svc in services:
            key = f"circuit:{svc}"
            # Get current state before resetting
            state_data = redis_client.hgetall(key)
            old_state = state_data.get(b'state', b'unknown').decode('utf-8')
            
            # Reset circuit breaker
            redis_client.hset(key, mapping={
                'state': 'closed',
                'failures': 0,
                'last_success': int(time.time())
            })
            
            results[svc] = {
                "previous_state": old_state,
                "current_state": "closed"
            }
        
        return {
            "success": True,
            "reset_services": results
        }
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker: {e}")
        raise HTTPException(500, f"Failed to reset circuit breaker: {str(e)}")


@router.get("/verify-anchors")
def verify_anchors(
    calibrate_if_invalid: bool = Query(False, description="Automatically recalibrate invalid anchors"),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Verify that anchors are properly calibrated and within expected ranges."""
    session = get_db_session()
    if not session:
        return {"success": False, "error": "Database not available"}
        
    issues = {}
    missing_critical = []
    anchors_data = {}
    current_quarter = None
    
    try:
        current_quarter = current_quarter_version()
        
        # Get all anchors
        anchors = session.query(CompassAnchor).all()
        
        # No anchors at all is a critical issue
        if not anchors:
            result = {
                "success": False, 
                "error": "No anchors found in database",
                "action_required": "Run calibration"
            }
            
            # Recalibrate if requested
            if calibrate_if_invalid:
                try:
                    from startup.business_bootstrap import calibrate_compass_anchors
                    calibration_result = calibrate_compass_anchors("validated")
                    result["calibration_result"] = calibration_result
                    result["action_taken"] = "Calibration performed"
                except Exception as e:
                    result["calibration_error"] = str(e)
            
            return result
        
        # Check critical anchors exist
        critical_categories = ["GLOBAL:US", "GLOBAL:UK", "HARVARD-US", "GLOBAL-HARVARD"]
        
        for category in critical_categories:
            anchor = session.query(CompassAnchor).filter(
                CompassAnchor.category == category,
                CompassAnchor.version == current_quarter
            ).one_or_none()
            
            if not anchor:
                missing_critical.append(category)
                issues[f"missing_{category}"] = f"Missing critical anchor {category} for {current_quarter}"
                
        # Check anchor values are in reasonable ranges
        for anchor in anchors:
            # Create data object for this anchor
            anchors_data[anchor.category] = {
                "mu_low": float(anchor.mu_low),
                "mu_high": float(anchor.mu_high),
                "median_mu": float(anchor.median_mu),
                "version": anchor.version,
                "current": anchor.version == current_quarter,
                "issues": []
            }
            
            # Check if outdated
            if anchor.version != current_quarter:
                issue = f"Outdated anchor: {anchor.version} should be {current_quarter}"
                issues[f"outdated_{anchor.category}"] = issue
                anchors_data[anchor.category]["issues"].append(issue)
                
            # Check mu_low
            if anchor.mu_low < -0.10:
                issue = f"Suspiciously low mu_low: {anchor.mu_low:.4f}"
                issues[f"low_mu_low_{anchor.category}"] = issue
                anchors_data[anchor.category]["issues"].append(issue)
                
            # Check mu_high
            if anchor.mu_high > 0.50:
                issue = f"Suspiciously high mu_high: {anchor.mu_high:.4f}"
                issues[f"high_mu_high_{anchor.category}"] = issue
                anchors_data[anchor.category]["issues"].append(issue)
                
            # Check spread
            spread = anchor.mu_high - anchor.mu_low
            if spread < 0.05:
                issue = f"Spread too small: {spread:.4f}"
                issues[f"small_spread_{anchor.category}"] = issue
                anchors_data[anchor.category]["issues"].append(issue)
            elif spread > 0.40:
                issue = f"Spread too large: {spread:.4f}"
                issues[f"large_spread_{anchor.category}"] = issue
                anchors_data[anchor.category]["issues"].append(issue)
                
            # Check median
            if not (anchor.mu_low <= anchor.median_mu <= anchor.mu_high):
                issue = f"Median {anchor.median_mu:.4f} outside range [{anchor.mu_low:.4f}, {anchor.mu_high:.4f}]"
                issues[f"invalid_median_{anchor.category}"] = issue
                anchors_data[anchor.category]["issues"].append(issue)
        
        # Critical issue if any GLOBAL: or HARVARD- category has problems
        critical_issue = any(k for k in issues.keys() if any(c in k for c in ["GLOBAL:", "HARVARD-"]))
        
        result = {
            "success": len(issues) == 0,
            "critical_issue": critical_issue,
            "issues": issues,
            "missing_critical": missing_critical,
            "anchors_count": len(anchors),
            "current_quarter": current_quarter,
            "anchors": anchors_data
        }
        
        # Recalibrate if requested and issues found
        if calibrate_if_invalid and (critical_issue or missing_critical):
            try:
                from startup.business_bootstrap import calibrate_compass_anchors
                calibration_result = calibrate_compass_anchors("validated")
                result["calibration_result"] = calibration_result
                result["action_taken"] = "Calibration performed"
            except Exception as e:
                result["calibration_error"] = str(e)
                
        return result
        
    except Exception as e:
        logger.exception("Failed to verify anchors")
        return {"success": False, "error": str(e)}
    finally:
        try:
            session.close()
        except Exception:
            pass


@router.get("/logging-levels")
def get_logging_levels(_auth: None = Depends(_require_pub_or_basic)) -> Dict[str, str]:
    """Get current logging levels for all configured loggers."""
    return get_logger_levels()


@router.post("/set-logging-level")
def set_logging_level(
    logger_name: str = Query(..., description="Logger name to update"),
    level: str = Query(..., description="New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    _auth: None = Depends(_require_pub_or_basic)
) -> Dict[str, Any]:
    """Set logging level for a specific logger."""
    try:
        # Validate level
        level_upper = level.upper()
        if level_upper not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            return {
                "success": False,
                "error": f"Invalid log level: {level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
            }
            
        # Get numeric level
        numeric_level = getattr(logging, level_upper)
        
        # Set level for the specified logger
        logger = logging.getLogger(logger_name)
        old_level = logging.getLevelName(logger.level)
        logger.setLevel(numeric_level)
        
        return {
            "success": True,
            "logger": logger_name,
            "old_level": old_level,
            "new_level": level_upper
        }
        
    except Exception as e:
        logger.error(f"Failed to set logging level: {e}")
        return {"success": False, "error": str(e)}