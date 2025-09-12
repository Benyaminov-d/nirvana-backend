"""Business logic bootstrap - CVaR, caching, Compass, reconciliation."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, Union, List, Tuple

from core.db import get_db_session
from core.models import CvarSnapshot, CompassAnchor, ValidationFlags
from core.models import Symbols

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure detailed bootstrap logs are visible


def warm_cache_from_db(db_ready: bool) -> None:
    """Warm cache from existing CVaR snapshots."""
    if not db_ready:
        return
    
    from services.domain.cvar_unified_service import CvarUnifiedService
    
    try:
        sess = get_db_session()
        if sess is None:
            return
        
        try:
            rows = (
                sess.query(CvarSnapshot)
                .order_by(
                    CvarSnapshot.symbol.asc(),
                    CvarSnapshot.as_of_date.desc(),
                )
                .all()
            )
            by_symbol: dict[str, dict[int, CvarSnapshot]] = {}
            for r in rows:
                slot = by_symbol.setdefault(r.symbol, {})
                if r.alpha_label not in slot:
                    slot[r.alpha_label] = r
            
            calc = CvarUnifiedService()
            for sym, parts in by_symbol.items():
                any_block = parts.get(95) or parts.get(99) or parts.get(50)
                if any_block is None:
                    continue
                as_of = any_block.as_of_date.isoformat()

                def _triple(opt: CvarSnapshot | None) -> dict:
                    if opt is None:
                        return {"nig": None, "ghst": None, "evar": None}
                    return {"nig": opt.cvar_nig, "ghst": opt.cvar_ghst, "evar": opt.cvar_evar}

                payload = {
                    "success": True,
                    "as_of_date": as_of,
                    "start_date": None,
                    "data_summary": {},
                    "cached": True,
                    "cvar50": {
                    "annual": _triple(parts.get(50)),
                    "snapshot": _triple(parts.get(50)),
                    "alpha": 0.5,
                },
                    "cvar95": {
                    "annual": _triple(parts.get(95)),
                    "snapshot": _triple(parts.get(95)),
                    "alpha": 0.05,
                },
                    "cvar99": {
                    "annual": _triple(parts.get(99)),
                    "snapshot": _triple(parts.get(99)),
                    "alpha": 0.01,
                },
                }
                try:
                    calc.set_cached(sym, payload)
                except Exception:
                    continue
        finally:
            try:
                sess.close()
            except Exception:
                pass
    except Exception:
        logger.exception("Cache warming failed")


def get_valid_symbols() -> List[str]:
    """Get all symbols marked as valid in validation_flags."""
    session = get_db_session()
    if not session:
        logger.error("Failed to create database session")
        return []

    try:
        query = (
            session.query(ValidationFlags.symbol)
            .filter(ValidationFlags.valid == 1)
            .distinct()
        )
        symbols = [row[0] for row in query.all()]
        logger.info("Found %d valid symbols", len(symbols))
        return symbols
    except Exception as e:
        logger.error("Error retrieving valid symbols: %s", e)
        return []
    finally:
        session.close()


def check_cvar_processing_status() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if all valid symbols have been processed for CVaR.

    Returns:
        Tuple of (is_complete, status_dict)
    """
    valid_symbols = get_valid_symbols()
    if not valid_symbols:
        logger.error("No valid symbols found")
        return False, {"error": "No valid symbols found"}

    session = get_db_session()
    if not session:
        logger.error("Failed to create database session")
        return False, {"error": "Database session failed"}

    try:
        # Create a set of valid symbols for faster lookups
        valid_set = set(valid_symbols)
        total = len(valid_set)

        # Get processed symbols (with alpha_label=99 for consistency)
        processed_query = (
            session.query(CvarSnapshot.symbol)
            .filter(CvarSnapshot.alpha_label == 99)
            .distinct()
        )
        processed_symbols = set(row[0] for row in processed_query.all())

        # Filter to only include valid symbols
        processed_valid = processed_symbols.intersection(valid_set)
        processed_count = len(processed_valid)

        # Calculate remaining symbols
        remaining = valid_set - processed_valid
        remaining_count = len(remaining)

        # Calculate completion percentage
        completion_pct = (processed_count / total) * 100 if total > 0 else 0

        logger.info(
            "CVaR processing status: %d/%d symbols completed (%.1f%%)",
            processed_count, total, completion_pct
        )

        # Print first 10 remaining symbols as a sample if not complete
        if remaining:
            # Grace period before first retry to allow workers to start consuming
            try:
                import time as _t
                grace_min = int(os.getenv("CVAR_RETRY_GRACE_MINUTES", "60"))
                start_ts = int(os.getenv("CVAR_BOOTSTRAP_START_TS", "0"))
                now_ts = int(_t.time())
                within_grace = start_ts > 0 and (now_ts - start_ts) < (grace_min * 60)
            except Exception:
                within_grace = False

            sample = list(remaining)[:10]
            logger.info("Sample of remaining symbols: %s", ", ".join(sample))
            
            # Check if we should retry missing symbols
            retry_enabled = os.getenv("CVAR_RETRY_MISSING", "1").lower() in ("1", "true", "yes")

            # Do not retry during grace window when nothing processed yet
            if within_grace and processed_count == 0:
                logger.info(
                    "Skipping retry during grace period (%d min) to allow workers to start",
                    int(os.getenv("CVAR_RETRY_GRACE_MINUTES", "60"))
                )
                return False, {
                    "completed": False,
                    "processed": processed_count,
                    "total": total,
                    "remaining": remaining_count,
                    "percentage": completion_pct,
                    "grace": True,
                }
            retry_count = int(os.getenv("CVAR_RETRY_COUNT", "3"))
            retry_batch_size = int(os.getenv("CVAR_RETRY_BATCH_SIZE", "100"))
            
            if retry_enabled and retry_count > 0:
                # Get current retry attempt from environment
                current_retry = int(os.getenv("CVAR_CURRENT_RETRY", "0"))
                
                if current_retry < retry_count:
                    logger.info(
                        "Retrying missing symbols (attempt %d/%d): %d symbols", 
                        current_retry + 1, retry_count, len(remaining)
                    )
                    
                    # Increment retry counter
                    os.environ["CVAR_CURRENT_RETRY"] = str(current_retry + 1)
                    
                    # Retry in batches to avoid overwhelming the system
                    remaining_list = list(remaining)
                    for i in range(0, len(remaining_list), retry_batch_size):
                        batch = remaining_list[i:i+retry_batch_size]
                        logger.info("Retrying batch %d: %d symbols", i//retry_batch_size + 1, len(batch))
                        
                        try:
                            from startup.cvar_bootstrap import _enqueue_via_servicebus
                            result = _enqueue_via_servicebus(batch)
                            logger.info("Retry batch result: %s", result)
                        except Exception as e:
                            logger.error("Error retrying symbols: %s", e)

        status = {
            "completed": remaining_count == 0,
            "processed": processed_count,
            "total": total,
            "remaining": remaining_count,
            "percentage": completion_pct
        }

        return remaining_count == 0, status
    except Exception as e:
        logger.error("Error checking CVaR processing status: %s", e)
        return False, {"error": str(e)}
    finally:
        session.close()

def enqueue_all_if_snapshots_empty(db_ready: bool) -> Union[dict, None]:
    """
    If snapshots are empty, enqueue all symbols for CVaR calculation.
    Then wait for processing to complete before returning.
    """
    if not db_ready:
        return None
    
    try:
        # Use the existing function to enqueue symbols
        from startup.cvar_bootstrap import enqueue_all_if_snapshots_empty as _enqueue_all
        result = _enqueue_all(db_ready)

        # If nothing was enqueued (snapshots not empty or no symbols), return immediately
        if result is None or result.get("mode") == "none" or result.get("symbols", 0) == 0:
            return result

        # If symbols were enqueued, wait for processing to complete
        logger.info(
            "Waiting for CVaR processing to complete for %d symbols...",
            result.get('symbols', 0)
        )

        # Check if wait is enabled
        wait_enabled = os.getenv("WAIT_FOR_CVAR_PROCESSING", "1").lower() in ("1", "true", "yes")
        if not wait_enabled:
            logger.info("Skipping wait for CVaR processing (WAIT_FOR_CVAR_PROCESSING=0)")
            return result

        # Wait parameters
        check_interval_minutes = int(os.getenv("CVAR_CHECK_INTERVAL_MINUTES", "15"))
        max_wait_hours = int(os.getenv("CVAR_MAX_WAIT_HOURS", "24"))
        max_checks = (max_wait_hours * 60) // check_interval_minutes

        logger.info(
            "Will check every %d minutes, max wait time: %d hours",
            check_interval_minutes, max_wait_hours
        )

        # Initial check
        is_complete, status = check_cvar_processing_status()
        if is_complete:
            logger.info("All symbols already processed")
            result["processing_status"] = status
            return result

        # Wait and check periodically
        for check_num in range(1, max_checks + 1):
            # Wait for the specified interval
            wait_seconds = check_interval_minutes * 60
            logger.info("Waiting %d minutes before next check...", check_interval_minutes)

            # Print progress to stdout for visibility
            print(
                "  - CVaR processing: %d/%d symbols (%.1f%%), next check in %d minutes" % (
                    status.get('processed', 0),
                    status.get('total', 0),
                    status.get('percentage', 0),
                    check_interval_minutes
                )
            )

            time.sleep(wait_seconds)

            # Check status
            is_complete, status = check_cvar_processing_status()

            # If all symbols processed, break the loop
            if is_complete:
                logger.info("All symbols processed successfully")
                print(
                    "  - CVaR processing completed: %d/%d symbols",
                    status.get('processed', 0),
                    status.get('total', 0)
                )
                break

            # Log progress
            logger.info(
                "Check %d/%d: %d/%d symbols processed (%.1f%%)",
                check_num, max_checks,
                status.get('processed', 0), status.get('total', 0),
                status.get('percentage', 0)
            )

            # If reached max checks, log warning but continue
            if check_num >= max_checks:
                logger.warning(
                    "Reached maximum wait time of %d hours, but only %d/%d symbols processed",
                    max_wait_hours,
                    status.get('processed', 0), status.get('total', 0)
                )
                print(
                    "  - WARNING: CVaR processing incomplete after %d hours: %d/%d symbols (%.1f%%)" % (
                        max_wait_hours,
                        status.get('processed', 0),
                        status.get('total', 0),
                        status.get('percentage', 0)
                    )
                )
                break

        # Add processing status to result
        result["processing_status"] = status
        return result
    except Exception as e:
        logger.error("CVaR bootstrap failed: %s", str(e))
        return {"error": str(e)}


def reconcile_cvar_snapshots() -> Dict[str, Union[int, str]]:
    """Reconcile CVaR snapshots to bind them to instruments."""
    try:
        from startup.reconcile_snapshots import reconcile_cvar_snapshots as _reconcile
        return _reconcile()
    except Exception as e:
        logger.error("Snapshot reconciliation failed: %s", str(e))
        return {"error": str(e)}


def setup_compass_parameters() -> None:
    """Setup Compass parameters for validated universe."""
    try:
        from services.compass_parameters_service import CompassParametersService
        from core.universe_config import ACTIVE_UNIVERSE
        
        logger.info("Creating compass parameters for validated universe")
        
        # Create service and process parameters (μ from EODHD, L from CVaR snapshots)
        service = CompassParametersService()
        service.create_parameters_for_validated_universe(universe_type=ACTIVE_UNIVERSE)
        
        logger.info("Compass parameters processing completed")
    except Exception as e:
        logger.error("Compass parameters creation failed: %s", str(e))


def verify_anchors_quality() -> Dict[str, Any]:
    """Verify that anchors are properly calibrated and within expected ranges."""
    session = get_db_session()
    if not session:
        return {"success": False, "error": "Database not available"}
        
    issues = {}
    current_quarter = None
    
    try:
        from services.compass_anchors import current_quarter_version
        current_quarter = current_quarter_version()
    except Exception as e:
        logger.error(f"Failed to get current quarter: {e}")
        current_quarter = "Unknown"
    
    # Get all anchors
    anchors = session.query(CompassAnchor).all()
    
    # No anchors at all is a critical issue
    if not anchors:
        return {
            "success": False, 
            "error": "No anchors found in database",
            "action_required": "Run calibration"
        }
    
    # Check critical anchors exist
    critical_categories = ["GLOBAL:US", "GLOBAL:UK", "HARVARD-US", "GLOBAL-HARVARD"]
    for category in critical_categories:
        anchor = session.query(CompassAnchor).filter(
            CompassAnchor.category == category,
            CompassAnchor.version == current_quarter
        ).one_or_none()
        
        if not anchor:
            issues[f"missing_{category}"] = f"Missing critical anchor {category} for {current_quarter}"
            
    # Check anchor values are in reasonable ranges
    for anchor in anchors:
        # Check if outdated
        if anchor.version != current_quarter:
            issues[f"outdated_{anchor.category}"] = f"Outdated anchor: {anchor.version} should be {current_quarter}"
            
        # Check mu_low
        if anchor.mu_low < -0.10:
            issues[f"low_mu_low_{anchor.category}"] = f"Suspiciously low mu_low: {anchor.mu_low:.4f}"
            
        # Check mu_high
        if anchor.mu_high > 0.50:
            issues[f"high_mu_high_{anchor.category}"] = f"Suspiciously high mu_high: {anchor.mu_high:.4f}"
            
        # Check spread
        spread = anchor.mu_high - anchor.mu_low
        if spread < 0.05:
            issues[f"small_spread_{anchor.category}"] = f"Spread too small: {spread:.4f}"
        elif spread > 0.40:
            issues[f"large_spread_{anchor.category}"] = f"Spread too large: {spread:.4f}"
            
        # Check median
        if not (anchor.mu_low <= anchor.median_mu <= anchor.mu_high):
            issues[f"invalid_median_{anchor.category}"] = f"Median {anchor.median_mu:.4f} outside range [{anchor.mu_low:.4f}, {anchor.mu_high:.4f}]"
    
    # Critical issue if any GLOBAL: or HARVARD- category has problems
    critical_issue = any(k for k in issues.keys() if any(c in k for c in ["GLOBAL:", "HARVARD-"]))
    
    return {
        "success": len(issues) == 0,
        "critical_issue": critical_issue,
        "issues": issues,
        "anchors_count": len(anchors),
        "current_quarter": current_quarter
    }

def calibrate_compass_anchors(scope: str = "default") -> Dict[str, Any]:
    """Auto-calibrate Compass anchors based on scope."""
    try:
        from services.compass_anchors import (
            auto_calibrate_from_db,
            auto_calibrate_global_per_country_from_db,
            auto_calibrate_by_type_country_from_db,
            calibrate_validated_universe_anchors,
            calibrate_harvard_universe_anchors,
        )
        
        results = {}
        
        # Check if we need calibration
        anchor_check = verify_anchors_quality()
        if anchor_check.get("success", False) and scope == "auto":
            logger.info("Anchors quality check passed, skipping calibration")
            results["status"] = "skipped_not_needed"
            results["check"] = anchor_check
            return results
        
        if scope == "validated" or scope == "full" or scope == "auto":
            # Validated universe anchors (filters: US/UK/CA)
            logger.info("Calibrating validated universe anchors")
            res_validated = calibrate_validated_universe_anchors()
            results['validated'] = res_validated
            logger.info("Validated anchors summary: %s", res_validated)
            
            # Also calibrate Harvard anchors for full/validated scope
            logger.info("Calibrating Harvard universe anchors")
            res_harvard = calibrate_harvard_universe_anchors()
            results['harvard'] = res_harvard
            logger.info("Harvard anchors summary: %s", res_harvard)
            
        elif scope == "global" or scope == "full":
            # Per-country GLOBAL (discover from DB)
            res1 = auto_calibrate_global_per_country_from_db()
            results['per_country'] = res1
            logger.info("Anchors per-country summary: %s", res1)
            
            # Per (type,country)
            res2 = auto_calibrate_by_type_country_from_db()
            results['by_type_country'] = res2
            logger.info("Anchors by-type-country summary: %s", res2)
            
        elif scope == "harvard_only":
            # Only calibrate Harvard universe anchors
            logger.info("Calibrating Harvard universe anchors")
            res_harvard = calibrate_harvard_universe_anchors()
            results['harvard'] = res_harvard
            logger.info("Harvard anchors summary: %s", res_harvard)
            
        else:
            # Default US-Equity-Returns calibration
            if auto_calibrate_from_db("US-Equity-Returns"):
                results['default'] = "calibrated"
                logger.info("Compass anchors calibrated (US-Equity-Returns)")
            else:
                results['default'] = "present_or_skipped"
                logger.info("Compass anchors present or skipped (US-Equity-Returns)")
        
        # Re-check anchor quality after calibration
        if scope in ["full", "validated", "harvard_only", "auto"]:
            post_check = verify_anchors_quality()
            results["post_calibration_check"] = post_check
            
            if not post_check.get("success", False):
                logger.warning("Anchor quality issues remain after calibration: %s", post_check.get("issues", {}))
        
        return results
    except Exception as e:
        logger.error("Compass anchors calibration failed: %s", str(e))
        return {"error": str(e)}


def calibrate_experiment_anchors() -> Dict[str, Any]:
    """Calibrate experimental anchors for /score-experiment page."""
    try:
        from services.compass_anchors import calibrate_special_sets
        
        res_exp = calibrate_special_sets()
        logger.info("Experiment anchors summary: %s", res_exp)
        return res_exp
    except Exception as e:
        logger.error("Experiment anchors calibration failed: %s", str(e))
        return {"error": str(e)}


def run_business_bootstrap(db_ready: bool) -> Dict[str, Union[bool, str, Dict[str, Any]]]:
    """Run all business logic bootstrap tasks with environment controls."""
    if not db_ready:
        return {"error": "Database not ready"}
    
    results = {}

    # Unified symbols pipeline enqueue (optional) - controlled via STARTUP_SYMBOLS_PIPELINE
    # If enabled, the backend will enqueue symbols to the Service Bus 'symbols-q'
    # based on env-provided symbols or universe filters. This powers the end-to-end
    # pipeline: validation → CVaR → Compass params → series write.
    try:
        pipeline_flag = os.getenv("STARTUP_SYMBOLS_PIPELINE", "0").lower()
        test_mode = pipeline_flag == "test"
        if pipeline_flag in ("1", "true", "yes", "test"):
            print("  - Enqueuing symbols pipeline%s..." % (" (TEST mode)" if test_mode else ""))  # Force visibility
            from services.application.queue_orchestration_service import (
                QueueOrchestrationService,
            )

            svc = QueueOrchestrationService()

            # Determine symbols or filters from env
            raw_syms = (os.getenv("PIPELINE_SYMBOLS") or "").strip()
            symbols: list[str] = []
            if raw_syms:
                symbols = [s.strip().upper() for s in raw_syms.split(",") if s.strip()]
            elif os.getenv("PIPELINE_SYMBOLS_FILE"):
                fpath = os.getenv("PIPELINE_SYMBOLS_FILE", "").strip()
                try:
                    import pathlib
                    p = pathlib.Path(fpath)
                    if p.exists():
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        for line in text.splitlines():
                            for part in line.split(","):
                                tok = part.strip().upper()
                                if tok:
                                    symbols.append(tok)
                except Exception as _e:
                    logger.warning("Failed to read PIPELINE_SYMBOLS_FILE=%s: %s", fpath, _e)

            source = (os.getenv("PIPELINE_SOURCE") or "eodhd").strip()
            as_of = (os.getenv("PIPELINE_AS_OF") or "").strip() or None

            # Option to include ALL symbols (valid and non-valid)
            include_all_symbols = (
                (os.getenv("STARTUP_SYMBOLS_PIPELINE_VALIDANDNONVALID", "0") or "")
                .lower()
                in ("1", "true", "yes")
            )

            if not symbols:
                # If explicitly requested, include ALL symbols (valid and non-valid)
                if include_all_symbols:
                    try:
                        sess = get_db_session()
                        if sess:
                            db_syms = [row[0] for row in sess.query(Symbols.symbol).distinct().all()]
                            symbols = [s for s in db_syms if isinstance(s, str) and s.strip()]
                            try:
                                sess.close()
                            except Exception:
                                pass
                            if symbols:
                                logger.info(
                                    "Symbols pipeline (ALL): using %d symbols from DB "
                                    "(valid and non-valid)",
                                    len(symbols),
                                )
                    except Exception as _all_err:
                        logger.warning("Symbols pipeline (ALL) DB load failed: %s", _all_err)
                else:
                    # Default: build from existing UniverseManager (validated only)
                    try:
                        from services.universe_manager import get_universe_manager
                        from core.universe_config import ACTIVE_UNIVERSE
                        um = get_universe_manager(ACTIVE_UNIVERSE)
                        products = um.get_universe_products()
                        symbols = [p.symbol for p in products if getattr(p, "symbol", None)]
                    except Exception as _ue:
                        logger.warning("UniverseManager fallback failed: %s", _ue)

            if not symbols:
                # Universe is empty (likely no valid symbols yet) → fallback to DB symbols
                try:
                    sess = get_db_session()
                    if sess:
                        db_syms = [row[0] for row in sess.query(Symbols.symbol).distinct().all()]
                        symbols = [s for s in db_syms if isinstance(s, str) and s.strip()]
                        try:
                            sess.close()
                        except Exception:
                            pass
                        if symbols:
                            logger.info("Symbols pipeline fallback: using %d symbols from DB", len(symbols))
                except Exception as _db_fallback_err:
                    logger.warning("DB fallback for symbols failed: %s", _db_fallback_err)

            # In TEST mode, limit the number of symbols enqueued (default 200)
            if test_mode and symbols:
                try:
                    cap = int(os.getenv("STARTUP_SYMBOLS_TEST_LIMIT", "200"))
                except Exception:
                    cap = 200
                if cap > 0:
                    symbols = symbols[:cap]
                    logger.info("STARTUP_SYMBOLS_PIPELINE=TEST → limiting symbols to %d", len(symbols))

            if symbols:
                res = svc.enqueue_symbol_batch(symbols, source=source, as_of=as_of)
                results["symbols_pipeline"] = res
            else:
                results["symbols_pipeline"] = {"success": False, "error": "no symbols resolved"}
            print(
                "  - Symbols pipeline enqueued%s"
                % (" (TEST mode, limited)" if test_mode else "")
            )  # Force visibility
        else:
            results["symbols_pipeline"] = "skipped"
    except Exception:
        logger.exception("Symbols pipeline enqueue failed")
        results["symbols_pipeline"] = "error"
    
    # CVaR bootstrap - controlled via STARTUP_CVAR_BOOTSTRAP
    # This must run first to ensure we have CVaR data for compass parameters
    try:
        cvar_bootstrap_flag = os.getenv("STARTUP_CVAR_BOOTSTRAP", "1").lower()
        if cvar_bootstrap_flag in ("1", "true", "yes"):
            print("  - Running CVaR bootstrap...")  # Force visibility
            res_boot = enqueue_all_if_snapshots_empty(db_ready)
            print(f"  - CVaR bootstrap completed: {res_boot}")  # Force visibility
            results['cvar_bootstrap'] = res_boot
            if res_boot is not None:
                logger.info("CVaR bootstrap summary: %s", res_boot)
        else:
            print(f"  - CVaR bootstrap skipped (STARTUP_CVAR_BOOTSTRAP={cvar_bootstrap_flag})")  # Force visibility
            logger.info(
                "CVaR bootstrap skipped (STARTUP_CVAR_BOOTSTRAP=%s)",
                cvar_bootstrap_flag,
            )
            results['cvar_bootstrap'] = 'skipped'
    except Exception:
        logger.exception("CVaR bootstrap enqueue failed")
        results['cvar_bootstrap'] = 'error'
    
    # CVaR enqueue for all valid symbols - controlled via STARTUP_CVAR_ENQUEUE_VALID
    # Enqueue all ValidationFlags.valid=1 directly to CVaR calculations queue
    try:
        cvar_enq_valid_flag = os.getenv("STARTUP_CVAR_ENQUEUE_VALID", "0").lower()
        if cvar_enq_valid_flag in ("1", "true", "yes"):
            print("  - Enqueuing CVaR for all valid symbols...")  # Force visibility
            try:
                from startup.cvar_bootstrap import _build_symbol_list, _enqueue_via_servicebus
                syms_valid = _build_symbol_list()
                if syms_valid:
                    res_enq = _enqueue_via_servicebus(syms_valid)
                else:
                    res_enq = {"mode": "none", "symbols": 0}
                results['cvar_enqueue_valid'] = res_enq
                logger.info("CVaR enqueue (all valid) summary: %s", res_enq)
            except Exception as _cv_e:
                logger.exception("CVaR enqueue (all valid) failed: %s", _cv_e)
                results['cvar_enqueue_valid'] = 'error'
            print("  - CVaR enqueue for valid symbols completed")  # Force visibility
        else:
            results['cvar_enqueue_valid'] = 'skipped'
    except Exception:
        logger.exception("CVaR enqueue (all valid) step failed")
        results['cvar_enqueue_valid'] = 'error'
    
    # Compass parameters - controlled via STARTUP_COMPASS_PARAMETERS
    # This must run before anchors calibration
    try:
        compass_params_flag = os.getenv("STARTUP_COMPASS_PARAMETERS", "1").lower()  # Default to 1 (enabled)
        if compass_params_flag in ("1", "true", "yes"):
            print("  - Creating Compass parameters...")  # Force visibility
            setup_compass_parameters()
            print("  - Compass parameters created")  # Force visibility
            results['compass_parameters'] = True
        else:
            logger.info(
                "Compass parameters creation skipped (STARTUP_COMPASS_PARAMETERS=%s)",
                compass_params_flag,
            )
            results['compass_parameters'] = 'skipped'
    except Exception:
        logger.exception("Compass parameters creation failed")
        results['compass_parameters'] = 'error'
    
    # Compass anchors - controlled via STARTUP_COMPASS_ANCHORS
    # This must run after compass parameters are created
    try:
        compass_anchors_flag = os.getenv("STARTUP_COMPASS_ANCHORS", "1").lower()
        if compass_anchors_flag in ("1", "true", "yes"):
            print("  - Calibrating Compass anchors...")  # Force visibility
            scope = os.getenv("COMPASS_ANCHOR_SCOPE", "auto").lower()
            
            # Check anchor quality first
            print("  - Checking anchor quality...")  # Force visibility
            anchor_check = verify_anchors_quality()
            if anchor_check.get("issues"):
                print(f"  - Anchor quality issues detected: {len(anchor_check.get('issues', {}))}")  # Force visibility
                
            # Calibrate anchors - if scope is "auto", it will only calibrate if needed
            anchor_results = calibrate_compass_anchors(scope)
            print(f"  - Compass anchors calibration complete: {anchor_results.get('status', 'completed')}")  # Force visibility
            results['compass_anchors'] = anchor_results
            
            # Optional experiment anchors - controlled via COMPASS_EXPERIMENT_ANCHORS
            exp_flag = os.getenv("COMPASS_EXPERIMENT_ANCHORS", "0").lower()
            if exp_flag in ("1", "true", "yes"):
                exp_results = calibrate_experiment_anchors()
                results['experiment_anchors'] = exp_results
            else:
                logger.info("Experiment anchors skipped (COMPASS_EXPERIMENT_ANCHORS=%s)", exp_flag)
                results['experiment_anchors'] = 'skipped'
        else:
            logger.info("Compass anchors skipped (STARTUP_COMPASS_ANCHORS=0)")
            results['compass_anchors'] = 'skipped'
            results['experiment_anchors'] = 'skipped'
    except Exception:
        logger.exception("Compass anchors auto-calibration failed")
        results['compass_anchors'] = 'error'
        results['experiment_anchors'] = 'error'
    
    
    # Reconcile CVaR snapshots - controlled via NIR_RECONCILE_SNAPSHOTS
    try:
        reconcile_flag = os.getenv("NIR_RECONCILE_SNAPSHOTS", "0").lower()
        if reconcile_flag in ("1", "true", "yes"):
            res_rec = reconcile_cvar_snapshots()
            results['reconcile_snapshots'] = res_rec
            logger.info("Reconcile snapshots summary: %s", res_rec)
        else:
            logger.info("Reconcile snapshots skipped (NIR_RECONCILE_SNAPSHOTS=0)")
            results['reconcile_snapshots'] = 'skipped'
    except Exception:
        logger.exception("Reconcile snapshots failed")
        results['reconcile_snapshots'] = 'error'
    
    # Cache warming - controlled via STARTUP_CACHE_WARMING
    try:
        cache_warming_flag = os.getenv("STARTUP_CACHE_WARMING", "1").lower()
        if cache_warming_flag in ("1", "true", "yes"):
            print("  - Warming CVaR cache from database...")  # Force visibility
            warm_cache_from_db(db_ready)
            print("  - Cache warming completed")  # Force visibility
            results['cache_warming'] = True
        else:
            print(f"  - Cache warming skipped (STARTUP_CACHE_WARMING={cache_warming_flag})")  # Force visibility
            logger.info(
                "Cache warming skipped (STARTUP_CACHE_WARMING=%s)",
                cache_warming_flag,
            )
            results['cache_warming'] = 'skipped'
    except Exception:
        logger.exception("Cache warming failed")
        results['cache_warming'] = 'error'
    
    return results
