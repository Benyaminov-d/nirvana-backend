"""
Refactoring Tools Routes - Helper endpoints for refactoring process.

This module provides utility endpoints to help with the refactoring process,
including finding Russian text, emojis, and tracking architectural progress.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List
import re
import os
import logging
from pathlib import Path

from utils.auth import require_pub_or_basic as _require_pub_or_basic

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/refactoring/i18n-scan")
def internationalization_scan(
    scan_path: str = Query("backend", description="Path to scan relative to project root"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Scan for Russian text and emojis that need to be replaced.
    
    This endpoint helps with the internationalization task by finding:
    - Russian comments
    - Russian log messages  
    - Russian strings in code
    - Emojis that should be removed
    """
    
    try:
        base_path = Path(__file__).parent.parent.parent  # nirvana-app/
        scan_dir = base_path / scan_path
        
        if not scan_dir.exists():
            raise HTTPException(404, f"Scan path does not exist: {scan_dir}")
        
        # Patterns for detection
        cyrillic_pattern = re.compile(r'[а-яёА-ЯЁ]', re.UNICODE)
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|'
            r'[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|'
            r'[\U00002600-\U000026FF]|[\U00002700-\U000027BF]',
            re.UNICODE
        )
        
        findings = {
            "russian_comments": [],
            "russian_strings": [],
            "russian_logs": [],
            "emojis": [],
            "summary": {},
            "scan_info": {
                "path": str(scan_dir),
                "files_scanned": 0,
                "total_issues": 0
            }
        }
        
        # Scan Python files
        for py_file in scan_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            findings["scan_info"]["files_scanned"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # Skip if line is in REFACTORING_PLAN.md style files (allowed Russian)
                    if "REFACTORING_PLAN" in str(py_file) or "README" in str(py_file):
                        continue
                    
                    # Check for Russian text
                    if cyrillic_pattern.search(line):
                        relative_path = py_file.relative_to(base_path)
                        
                        if line.strip().startswith('#'):
                            # Russian comment
                            findings["russian_comments"].append({
                                "file": str(relative_path),
                                "line": line_num,
                                "content": line.strip(),
                                "type": "comment"
                            })
                        elif 'logger' in line or 'logging' in line:
                            # Russian log message
                            findings["russian_logs"].append({
                                "file": str(relative_path),
                                "line": line_num,
                                "content": line.strip(),
                                "type": "log"
                            })
                        else:
                            # Russian string/code
                            findings["russian_strings"].append({
                                "file": str(relative_path),
                                "line": line_num,
                                "content": line.strip(),
                                "type": "string"
                            })
                    
                    # Check for emojis
                    if emoji_pattern.search(line):
                        relative_path = py_file.relative_to(base_path)
                        findings["emojis"].append({
                            "file": str(relative_path),
                            "line": line_num,
                            "content": line.strip(),
                            "type": "emoji"
                        })
                        
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
                continue
        
        # Calculate summary
        findings["summary"] = {
            "total_russian_comments": len(findings["russian_comments"]),
            "total_russian_strings": len(findings["russian_strings"]),
            "total_russian_logs": len(findings["russian_logs"]),
            "total_emojis": len(findings["emojis"]),
            "files_with_issues": len(set(
                item["file"] for category in ["russian_comments", "russian_strings", "russian_logs", "emojis"]
                for item in findings[category]
            )),
            "priority_fixes": []
        }
        
        findings["scan_info"]["total_issues"] = sum([
            len(findings["russian_comments"]),
            len(findings["russian_strings"]),
            len(findings["russian_logs"]),
            len(findings["emojis"])
        ])
        
        # Add priority recommendations
        if findings["summary"]["total_russian_logs"] > 0:
            findings["summary"]["priority_fixes"].append("Replace Russian log messages first - affects runtime")
        
        if findings["summary"]["total_russian_strings"] > 0:
            findings["summary"]["priority_fixes"].append("Replace Russian strings in code - affects functionality")
        
        if findings["summary"]["total_emojis"] > 0:
            findings["summary"]["priority_fixes"].append("Remove emojis from code - professional style requirement")
        
        return findings
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"I18n scan failed: {e}")
        raise HTTPException(500, f"Scan failed: {str(e)}")


@router.get("/refactoring/architecture-progress")
def architecture_progress(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Show current architectural refactoring progress.
    
    This endpoint tracks the migration from old to new architecture.
    """
    
    try:
        # Count files in new architecture
        base_path = Path(__file__).parent.parent
        
        repositories_count = len(list((base_path / "repositories").glob("*.py"))) - 1  # Exclude __init__.py
        domain_services_count = len(list((base_path / "services" / "domain").glob("*.py"))) - 1
        refactored_routes_count = len([
            f for f in (base_path / "routes").glob("*refactored*.py")
        ])
        
        # Estimate remaining work by counting get_db_session usage
        old_db_access_count = 0
        total_py_files = 0
        
        for py_file in base_path.rglob("*.py"):
            if "__pycache__" in str(py_file) or "repositories" in str(py_file):
                continue
            
            total_py_files += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "get_db_session()" in content:
                        old_db_access_count += content.count("get_db_session()")
            except Exception:
                continue
        
        progress = {
            "phase_1_repositories": {
                "status": "Completed",
                "repositories_created": repositories_count,
                "expected_repositories": 6,
                "completion_percentage": min(100, (repositories_count / 6) * 100)
            },
            "phase_1_domain_services": {
                "status": "In Progress", 
                "domain_services_created": domain_services_count,
                "expected_domain_services": 5,
                "completion_percentage": min(100, (domain_services_count / 5) * 100)
            },
            "route_migration": {
                "status": "In Progress",
                "refactored_routes": refactored_routes_count,
                "legacy_db_access_points": old_db_access_count,
                "total_py_files": total_py_files,
                "migration_notes": [
                    "/cvar/clear-cache migrated to new architecture",
                    "/cvar/snapshot migrated to new architecture", 
                    "Demo routes created showing new patterns",
                    f"{old_db_access_count} legacy DB access points remaining"
                ]
            },
            "architecture_quality": {
                "repositories_with_base_class": repositories_count > 0,
                "domain_services_isolated": domain_services_count > 0,
                "clean_separation_achieved": refactored_routes_count > 0,
                "testing_ready": True,  # New architecture is testable
                "performance_improved": "Session management centralized"
            },
            "next_milestones": [
                "Complete CVaR route migration",
                "Create application services layer",
                "Migrate ticker routes",
                "Create infrastructure services",
                "Remove legacy cvar_calculator.py"
            ],
            "benefits_realized": [
                "No SQL in new route handlers",
                "Centralized database session management",
                "Type-safe repository interfaces",
                "Easy testing with dependency injection",
                "Consistent error handling",
                "Clear separation of concerns"
            ]
        }
        
        return progress
        
    except Exception as e:
        logger.error(f"Architecture progress check failed: {e}")
        raise HTTPException(500, f"Progress check failed: {str(e)}")


@router.get("/refactoring/migration-status")
def migration_status(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Detailed status of ongoing migration efforts.
    """
    
    try:
        base_path = Path(__file__).parent.parent
        
        # Analyze route files
        route_files = list((base_path / "routes").glob("*.py"))
        route_analysis = {
            "total_route_files": len(route_files),
            "refactored_files": [],
            "legacy_files": [],
            "migration_candidates": []
        }
        
        for route_file in route_files:
            if "__init__" in route_file.name:
                continue
                
            try:
                with open(route_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_info = {
                    "name": route_file.name,
                    "has_db_session": "get_db_session()" in content,
                    "db_session_count": content.count("get_db_session()"),
                    "has_repositories": "Repository" in content,
                    "has_domain_services": "domain" in content.lower(),
                    "line_count": len(content.split('\n'))
                }
                
                if "refactored" in route_file.name or file_info["has_repositories"]:
                    route_analysis["refactored_files"].append(file_info)
                elif file_info["has_db_session"]:
                    route_analysis["legacy_files"].append(file_info)
                else:
                    route_analysis["migration_candidates"].append(file_info)
                    
            except Exception:
                continue
        
        return {
            "migration_summary": {
                "refactored_routes": len(route_analysis["refactored_files"]),
                "legacy_routes": len(route_analysis["legacy_files"]),
                "migration_candidates": len(route_analysis["migration_candidates"]),
                "total_routes": route_analysis["total_route_files"]
            },
            "detailed_analysis": route_analysis,
            "recommendations": {
                "high_priority": [
                    f for f in route_analysis["legacy_files"]
                    if f["db_session_count"] > 10
                ],
                "quick_wins": [
                    f for f in route_analysis["legacy_files"] 
                    if f["db_session_count"] <= 3
                ],
                "complex_migrations": [
                    f for f in route_analysis["legacy_files"]
                    if f["line_count"] > 500
                ]
            },
            "architecture_notes": {
                "pattern_established": len(route_analysis["refactored_files"]) > 0,
                "repositories_ready": True,
                "domain_services_ready": True,
                "migration_approach": "Incremental endpoint-by-endpoint migration"
            }
        }
        
    except Exception as e:
        logger.error(f"Migration status check failed: {e}")
        raise HTTPException(500, f"Status check failed: {str(e)}")
