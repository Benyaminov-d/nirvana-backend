from __future__ import annotations
import math
import os
import logging
from typing import Any, Dict, Tuple

import numpy as np

from services.lambert import (
    lambert_nig_lookup,
    lambert_ghst_lookup,
    lambert_evar_lookup,
    lambert_nig_params,
)
from services.azure_client import call_cvar_calculate


logger = logging.getLogger(__name__)


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _extract_from(
    data_in: dict[str, Any],
) -> Tuple[float | None, float | None, float | None]:
    try:
        n = _to_float(data_in.get("cvar_nig"))
    except Exception:
        n = None
    try:
        g = _to_float(data_in.get("cvar_ghst"))
    except Exception:
        g = None
    try:
        e = _to_float(data_in.get("cvar_evar"))
    except Exception:
        e = None
    return n, g, e


def compute_cvar_blocks(
    returns_log: np.ndarray,
    *,
    symbol: str | None,
    historical: bool,
    prefer_local: bool,
    sims: int,
    trading_days: int,
    log_phase: int,
    load_from_db_cb,
) -> Dict[str, Any]:
    r_log = np.ascontiguousarray(returns_log, np.float64)
    r_simple_diag = np.expm1(r_log)
    r_simple_diag = r_simple_diag[np.isfinite(r_simple_diag)]

    # dynamic alpha for 50%
    try:
        total_days = int(r_simple_diag.size)
        losing_days = int((r_simple_diag < 0.0).sum())
        dyn_alpha_50 = (
            1.0 - (losing_days / total_days)
            if total_days > 0
            else 0.5
        )
    except Exception:
        dyn_alpha_50 = 0.5
    if not math.isfinite(dyn_alpha_50):
        dyn_alpha_50 = 0.5
    dyn_alpha_50 = max(0.001, min(0.999, float(dyn_alpha_50)))

    alpha_map = {50: dyn_alpha_50, 95: 0.95, 99: 0.99}

    lam_nig = lambert_nig_lookup(symbol) if symbol else None
    if log_phase >= 3 and symbol:
        try:
            _ = lambert_nig_params(symbol)
        except Exception as _exc:
            logger.warning(
                "lambert NIG params lookup failed for %s: %s",
                symbol,
                _exc,
            )
    lam_ghst = lambert_ghst_lookup(symbol) if symbol else None
    lam_evar = lambert_evar_lookup(symbol) if symbol else None

    disable_local_env = (
        os.getenv("NIR_DISABLE_LOCAL_COMPUTE", "0").lower()
        in ("1", "true", "yes")
    )
    skip_remote_when_local = (
        os.getenv("NIR_SKIP_REMOTE_WHEN_LOCAL", "0").lower()
        in ("1", "true", "yes")
    )
    use_local_first = True if historical else bool(prefer_local)
    local_allowed = (
        not disable_local_env
    ) and (
        historical or bool(prefer_local)
    )

    def _do_local(
        alpha_conf: float,
    ) -> Tuple[float | None, float | None, float | None]:
        n_l: float | None = None
        g_l: float | None = None
        e_l: float | None = None
        if not local_allowed:
            return n_l, g_l, e_l
        try:
            import nirvana_risk as nr  # type: ignore
            # Use log returns for core engines (NIG/GHST expect log space)
            tail_alpha = (
                (1.0 - float(alpha_conf))
                if float(alpha_conf) > 0.5
                else float(alpha_conf)
            )
            try:
                dec = int(os.getenv("NVAR_RET_DECIMALS", "12"))
            except Exception:
                dec = 12
            r_for_core = r_log
            if isinstance(r_for_core, np.ndarray) and dec >= 0:
                r_for_core = np.round(r_for_core, dec)

            # Optional winsorization in log space (best-effort)
            try:
                import importlib as _il  # type: ignore
                _ts = _il.import_module("nirvana_risk.timeseries")
                _wins = getattr(_ts, "winsorize_returns", None)
                from nirvana_risk.timeseries.config import (  # type: ignore
                    get_settings as _ts_get_settings,
                )
                _ts_cfg = _ts_get_settings()
                if getattr(_ts_cfg, "winsor_enabled", True) and callable(_wins):
                    r_for_core = np.asarray(
                        _wins(r_for_core, _ts_cfg.winsor_p_low, _ts_cfg.winsor_p_high),
                        np.float64,
                    )
            except Exception:
                pass

            years_auto = (
                os.getenv("NVAR_YEARS_AUTO", "0").lower()
                in ("1", "true", "yes")
            )
            if years_auto:
                try:
                    years_observed = float(simple.size) / 252.0
                except Exception:
                    years_observed = 1.0
                try:
                    years_min = float(os.getenv("NVAR_MIN_YEARS", "10.0"))
                except Exception:
                    years_min = 10.0
                years_local = max(
                    years_min,
                    years_observed,
                )
            else:
                try:
                    years_local = float(os.getenv("NVAR_YEARS", "1.0"))
                except Exception:
                    years_local = 1.0

            ghst_skew_env = os.getenv("NVAR_GHST_SKEW")
            ghst_skew_val = None
            if ghst_skew_env not in (None, ""):
                try:
                    ghst_skew_val = float(str(ghst_skew_env))
                except Exception:
                    ghst_skew_val = None
            ghst_auto = (
                os.getenv("NVAR_GHST_AUTO_SKEW", "0").lower()
                in ("1", "true", "yes")
            )
            ghst_family = int(os.getenv("NVAR_GHST_FAMILY", "0"))

            snap = nr.cvar.snapshot(
                r_for_core,
                alpha=tail_alpha,
                years=years_local,
                n_sims=sims,
                seed_nig=int(os.getenv("NVAR_SEED_NIG", "500")),
                seed_ghst=int(os.getenv("NVAR_SEED_GHST", "501")),
                ghst_df=float(os.getenv("NVAR_GHST_DF", "8.02")),
                ghst_skew=ghst_skew_val,
                ghst_auto_skew=ghst_auto,
                ghst_family=ghst_family,
            )
            n_l = _to_float(snap.get("NIG"))
            g_l = _to_float(snap.get("GHST"))
            e_l = _to_float(snap.get("EVaR"))
        except Exception as _exc:
            logger.warning("Local compute failed: %s", _exc)
        return n_l, g_l, e_l

    insufficient_error: dict[str, Any] | None = None
    out: Dict[str, Any] = {}

    # Sanity guard for pathological outputs from engines
    def _sanitize(v: float | None) -> float | None:
        try:
            if v is None:
                return None
            fv = float(v)
            if not np.isfinite(fv):
                return None
            if fv < 0:
                fv = abs(fv)
            try:
                cap = float(os.getenv("NVAR_CVAR_SANITY_MAX", "5.0"))
            except Exception:
                cap = 5.0
            if fv > cap:
                return None
            return fv
        except Exception:
            return None

    for label, alpha in alpha_map.items():
        nig = None
        ghst = None
        evar_val = None

        def _do_func_wrap() -> Tuple[float | None, float | None, float | None]:
            nonlocal insufficient_error
            data_func = (
                call_cvar_calculate(symbol, alpha) if symbol else {}
            )
            try:
                if (
                    isinstance(data_func, dict)
                    and data_func.get("_error")
                    and str(data_func.get("code")) == "insufficient_history"
                ):
                    insufficient_error = {
                        "success": False,
                        "code": "insufficient_history",
                        "error": data_func.get(
                            "error",
                            "insufficient history",
                        ),
                    }
                    return None, None, None
            except Exception:
                pass
            return _extract_from(data_func)

        if use_local_first:
            n_l, g_l, e_l = _do_local(alpha)
            nig, ghst, evar_val = n_l, g_l, e_l
            if not historical and not (
                local_allowed and skip_remote_when_local
            ):
                if nig is None and ghst is None and evar_val is None:
                    n_f, g_f, e_f = _do_func_wrap()
                    nig, ghst, evar_val = n_f, g_f, e_f
        else:
            n_f, g_f, e_f = _do_func_wrap()
            nig, ghst, evar_val = n_f, g_f, e_f
            if nig is None and ghst is None and evar_val is None:
                n_l, g_l, e_l = _do_local(alpha)
                nig, ghst, evar_val = n_l, g_l, e_l
                if (
                    (nig is None)
                    and (ghst is None)
                    and (evar_val is None)
                    and symbol
                ):
                    try:
                        payload_db = load_from_db_cb(symbol)
                        if isinstance(payload_db, dict):
                            key = f"cvar{label}"
                            blk = payload_db.get(key) or {}
                            ann = (
                                blk.get("annual", {})
                                if isinstance(blk, dict)
                                else {}
                            )
                            nig = ann.get("nig")
                            ghst = ann.get("ghst")
                            evar_val = ann.get("evar")
                    except Exception:
                        pass

        block = {
            "annual": {
                "nig": _sanitize(nig),
                "ghst": _sanitize(ghst),
                "evar": _sanitize(evar_val),
            },
            "snapshot": {"nig": nig, "ghst": ghst, "evar": evar_val},
            "alpha": float(alpha),
        }
        if lam_nig is not None and label in (95, 99):
            lam_block: dict[str, Any] = {"nig": lam_nig}
            if lam_ghst is not None and ghst is not None:
                lam_block["ghst"] = lam_ghst
            if (
                lam_evar is not None
                and label == 99
                and evar_val is not None
            ):
                lam_block["evar"] = lam_evar
            block["lambert"] = lam_block
        out[f"cvar{label}"] = block

    if insufficient_error is not None:
        return insufficient_error
    return out
