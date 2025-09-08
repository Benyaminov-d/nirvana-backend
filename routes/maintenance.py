"""Maintenance API endpoints for server administration.

This module contains endpoints that are used for server maintenance.
These endpoints are not part of the public API and should be accessed
only by administrators.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from core.db import get_db_session
from core.models import AuthAttempt
from utils.auth import require_pub_or_basic as _require_auth
from datetime import datetime, timedelta

router = APIRouter(prefix="/maintenance", tags=["maintenance"])


@router.get("/cleanup-auth-attempts", dependencies=[Depends(_require_auth)])
def cleanup_auth_attempts(
    hours: Optional[int] = 24,
    email: Optional[str] = None,
    include_successful: bool = False
) -> dict:
    """Clean up authentication attempts.
    
    Args:
        hours: Delete attempts older than this many hours
        email: Filter by email (if provided)
        include_successful: If True, also delete successful attempts
    
    Returns:
        Dict with count of deleted records
    """
    if hours < 1:
        raise HTTPException(400, "hours must be at least 1")
    
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "database unavailable")
    
    try:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Build query based on parameters
        query = sess.query(AuthAttempt).filter(AuthAttempt.timestamp_utc < cutoff)
        
        if email:
            query = query.filter(AuthAttempt.email == email.lower().strip())
            
        if not include_successful:
            query = query.filter(AuthAttempt.success == 0)
            
        deleted_count = query.delete(synchronize_session=False)
        sess.commit()
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} authentication attempts"
        }
    except Exception as e:
        sess.rollback()
        raise HTTPException(500, f"Error cleaning up auth attempts: {str(e)}")
    finally:
        sess.close()


@router.get("/system-info", dependencies=[Depends(_require_auth)])
def system_info() -> dict:
    """Get system information.
    
    Returns:
        Dict with system information
    """
    from datetime import datetime
    import platform
    import psutil
    import os
    
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "processor": platform.processor()
            },
            "resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting system info: {str(e)}"
        }
