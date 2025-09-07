from __future__ import annotations
import math
import os
from datetime import datetime as _dt, timedelta as _td
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import requests  # type: ignore

from utils.common import (
    resolve_eodhd_endpoint_symbol,
    canonical_instrument_type,
)
import importlib as _il
import logging as _log
from config import get_config as _get_config
try:
    from nirvana_risk.pipeline.metrics import measure_years_observations  # type: ignore
    from nirvana_risk.pipeline.gate import enforce_min_years  # type: ignore
    from nirvana_risk.pipeline.errors import InsufficientHistoryError  # type: ignore
    try:
        from nirvana_risk.pipeline.io_eodhd import get_price_series as _lib_get_prices  # type: ignore
        from nirvana_risk.pipeline.config import get_config as _lib_get_config  # type: ignore
    except Exception:  # pragma: no cover
        _lib_get_prices = None  # type: ignore
        _lib_get_config = None  # type: ignore
except Exception:  # pragma: no cover
    measure_years_observations = None  # type: ignore
    enforce_min_years = None  # type: ignore
    InsufficientHistoryError = Exception  # type: ignore
    _lib_get_prices = None  # type: ignore
    _lib_get_config = None  # type: ignore

_CONFIG = _get_config()

# Optional reusable timeseries pipeline (library) via dynamic import
_ts_cov_last252 = None  # type: ignore
_ts_cov_per_year = None  # type: ignore
_ts_liq_decide = None  # type: ignore
_ts_min_days_12m = None  # type: ignore
_an_long_flat = None  # type: ignore
_an_illiquid = None  # type: ignore
_an_incon_splits = None  # type: ignore
_an_price_field = None  # type: ignore
try:  # pragma: no cover - optional import
    _ts_mod = _il.import_module("nirvana_risk.timeseries")
    _ts_cov_last252 = getattr(_ts_mod, "coverage_last252", None)
    _ts_cov_per_year = getattr(_ts_mod, "coverage_per_year", None)
    _ts_liq_decide = getattr(_ts_mod, "decide_liquidity_status", None)
    _ts_min_days_12m = getattr(
        _ts_mod, "coverage_min_trading_days_last_12m", None
    )
    _an_long_flat = getattr(_ts_mod, "detect_long_flatline", None)
    _an_illiquid = getattr(_ts_mod, "detect_illiquid_spikes", None)
    _an_incon_splits = getattr(_ts_mod, "detect_inconsistent_splits", None)
    _an_price_field = getattr(
        _ts_mod, "detect_price_field_inconsistent", None
    )
    _ts_interpolate = getattr(_ts_mod, "interpolate_nav_daily", None)
except Exception:  # noqa: BLE001
    pass

TRADING_DAYS = 252

_logger = _log.getLogger("prices")
if not _logger.handlers:
    _h = _log.StreamHandler()
    _fmt = "%(asctime)s prices %(levelname)s: %(message)s"
    _h.setFormatter(_log.Formatter(_fmt))
    _logger.addHandler(_h)
_logger.setLevel(_log.INFO)
_logger.propagate = False


def _annual_from_prices(prices: np.ndarray) -> float:
    try:
        if prices is None or len(prices) < 2:
            return 0.0
        return float((prices[-1] / prices[0] - 1.0) * 100.0)
    except Exception:  # noqa: BLE001
        return 0.0


def load_prices(
    symbol: str,
    to_date: str | None = None,
    *,
    from_years: int | None = None,
) -> Dict[str, Any]:
    """
    Returns dict(success, returns, as_of_date, summary)
    summary: obs / years / price_range / total_return
    """
    csv_root = Path(__file__).parents[1] / "data-snapshots"
    use_csv = os.getenv("LOCAL_CSV_DATA", "False").lower() in (
        "true",
        "1",
        "yes",
    )
    try:
        _logger.info(
            "load start: sym=%s to_date=%s from_years=%s",
            symbol,
            str(to_date),
            str(from_years),
        )
    except Exception:
        pass

    # Resolve endpoint symbol early to ensure availability for later logic
    # (e.g., US equity split adjustments), regardless of loader branch used.
    try:
        endpoint_symbol = resolve_eodhd_endpoint_symbol(symbol)
    except Exception:  # noqa: BLE001
        endpoint_symbol = symbol

    def _csv_path(sym: str) -> Path:
        return csv_root / (
            "SP500TR.csv"
            if sym == "SP500TR"
            else "Fidelity_mutual_funds20240430.csv"
        )

    used_csv_branch = False

    if use_csv:
        path = _csv_path(symbol)
        if path.exists():
            try:
                df = pd.read_csv(path)
                if symbol == "SP500TR":
                    cols = set(df.columns)
                    if not {"Date", "SP500TR"} <= cols:
                        return {"success": False, "error": "bad CSV format"}
                    df = df.sort_values("Date")
                    prices = df["SP500TR"].to_numpy(float)
                    dates = df["Date"].to_numpy(str)
                    used_csv_branch = True
                else:
                    if "Date" in df.columns and symbol in df.columns:
                        df = df.sort_values("Date")
                        prices = df[symbol].to_numpy(float)
                        dates = df["Date"].to_numpy(str)
                        used_csv_branch = True
            except Exception:  # noqa: BLE001
                pass

    if not used_csv_branch:
        # Prefer library loader when available for parity with function
        if callable(_lib_get_prices):
            try:
                cfg = _lib_get_config() if callable(_lib_get_config) else {}
                days = int(cfg.get("years_days", 0))
                raw = _lib_get_prices(symbol, days=days, suffix=None)
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": f"API error {exc}"}
        else:
            api_key = os.getenv("EODHD_API_KEY")
            if not api_key:
                return {"success": False, "error": "EODHD_API_KEY not set"}
            endpoint_symbol = resolve_eodhd_endpoint_symbol(symbol)
            if not endpoint_symbol:
                return {"success": False, "error": f"unsupported symbol {symbol}"}
            url = f"https://eodhistoricaldata.com/api/eod/{endpoint_symbol}"
            _req = requests.Session()
            try:
                params = {"api_token": api_key, "fmt": "json", "order": "d"}
                yrs_env = os.getenv("NVAR_FETCH_YEARS")
                yrs = (
                    from_years
                    if from_years is not None
                    else (int(yrs_env) if yrs_env else None)
                )
            except Exception:  # noqa: BLE001
                yrs = None
            if yrs is not None and yrs > 0:
                since = (
                    _dt.utcnow().date() - _td(days=int(yrs * 365.25))
                ).isoformat()
                params["from"] = since
            try:
                resp = _req.get(url, params=params, timeout=60)
                resp.raise_for_status()
                raw = resp.json()
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": f"API error {exc}"}
        try:
            _rows = int(len(raw) if isinstance(raw, list) else 0)
            _logger.info("api branch: sym=%s rows=%d", symbol, _rows)
        except Exception:
            pass

        records: List[Tuple[str, float, float]] = []
        adj_present = 0
        adj_positive = 0
        zero_price_flags = 0
        price_field = _CONFIG.price_field
        alt_field = (
            "close" if price_field == "adjusted_close" else "adjusted_close"
        )
        allow_close_fallback = bool(_CONFIG.allow_close_fallback)
        for row in raw:
            try:
                d = str(row.get("date") if isinstance(row, dict) else row["date"])  # type: ignore[index]
                # Row may be from lib (adj_close/close) or direct EODHD (adjusted_close/close)
                pv = None
                for k in (price_field, "adj_close", "adjusted_close", "close"):
                    if isinstance(row, dict) and k in row and row[k] not in (None, 0, "", "0"):
                        pv = row[k]
                        break
                if pv in (None, 0, "", "0") and allow_close_fallback:
                    pv = row.get(alt_field) if isinstance(row, dict) else None
                p = float(pv) if pv is not None else float("nan")
                if (isinstance(row, dict) and ("adjusted_close" in row or "adj_close" in row)):
                    adj_present += 1
                    try:
                        adj_val = row.get("adjusted_close") or row.get("adj_close") or 0.0
                        if float(adj_val) > 0:
                            adj_positive += 1
                    except Exception:
                        pass
                if p == 0.0:
                    zero_price_flags += 1
                vol_val = row.get("volume") if isinstance(row, dict) else None
                try:
                    v = float(vol_val) if vol_val is not None else float("nan")
                except Exception:  # noqa: BLE001
                    v = float("nan")
                if (p == p) and p > 0:
                    records.append((d, p, v))
            except (KeyError, ValueError, TypeError):
                continue

        try:
            _logger.info("parsed valid price rows: %d", int(len(records)))
        except Exception:
            pass
        if not records:
            return {"success": False, "error": "no valid API price rows"}

        records.sort(key=lambda t: t[0])
        dates = [d for d, _, _ in records]
        prices = [p for _, p, _ in records]
        volumes = [v for _, _, v in records]

        use_splits = os.getenv("NVAR_USE_SPLITS", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        is_us_equity = endpoint_symbol.endswith(".US") and (
            symbol not in {"SP500TR"}
        )
        apply_splits = bool(
            use_splits and is_us_equity and (adj_positive == 0)
        )
        split_events: list[tuple[str, float]] = []
        if apply_splits:
            splits_url = (
                "https://eodhistoricaldata.com/api/splits/"
                f"{endpoint_symbol}"
            )
            try:
                s_resp = _req.get(
                    splits_url,
                    params={
                        "api_token": api_key,
                        "from": dates[0],
                        "to": dates[-1],
                        "fmt": "json",
                    },
                    timeout=60,
                )
                s_resp.raise_for_status()
                s_raw = s_resp.json()
                for ev in s_raw:
                    try:
                        sd = str(ev.get("date") or ev.get("Date"))
                        ratio = ev.get("ratio") or ev.get("split")
                        if isinstance(ratio, str):
                            if "/" in ratio:
                                a, b = ratio.split("/")
                            elif ":" in ratio:
                                a, b = ratio.split(":")
                            else:
                                a, b = ratio, "1"
                            r = float(a) / float(b)
                        else:
                            r = float(ratio)
                        if r > 0 and math.isfinite(r):
                            split_events.append((sd, r))
                    except Exception:  # noqa: BLE001
                        continue
                if split_events:
                    split_events.sort(key=lambda t: t[0])

                    def _factor_for_date(d: str) -> float:
                        f = 1.0
                        for sd, r in split_events:
                            if sd > d:
                                f *= r
                        return f

                    factors = [_factor_for_date(d) for d in dates]
                    prices = [p / f for p, f in zip(prices, factors)]
            except Exception:  # noqa: BLE001
                pass

        if len(prices) < 2:
            return {"success": False, "error": "not enough price data"}

    # Cut to date if requested
    if to_date:
        try:
            td = str(to_date)
            if len(td) == 8 and td.isdigit():
                try:
                    dt_obj = _dt.strptime(td, "%d%m%Y")
                except Exception:
                    dt_obj = _dt.strptime(td, "%Y%m%d")
                td_iso = dt_obj.strftime("%Y-%m-%d")
            else:
                try:
                    dt_obj = _dt.strptime(td, "%Y-%m-%d")
                except Exception:
                    dt_obj = _dt.fromisoformat(td)
                td_iso = dt_obj.date().isoformat()
            while dates and str(dates[-1]) > td_iso:
                dates.pop()
                prices.pop()
        except Exception:  # noqa: BLE001
            return {"success": False, "error": f"bad to_date: {to_date}"}

    prices_np = np.asarray(prices, float)
    volumes_np = None
    try:
        volumes_np = (
            np.asarray(volumes, float)
            if "volumes" in locals()
            else None
        )
    except Exception:  # noqa: BLE001
        volumes_np = None

    anomalies: dict[str, Any] | None = None
    try:
        import importlib as _il
        _mod = _il.import_module("nirvana_risk_core.src_core")
        validate_and_clean = getattr(_mod, "validate_and_clean", None)
    except Exception:
        validate_and_clean = None

    # Read validation flag from shared config for parity with function
    try:
        _cfg_local = _lib_get_config() if callable(_lib_get_config) else {}
    except Exception:
        _cfg_local = {}
    _von = str(_cfg_local.get("validation_enabled", True)).lower() not in (
        "0",
        "off",
        "false",
        "no",
    )

    _orig_dates_for_drop_stats: list[str] | None = list(dates)
    _orig_prices_for_ar = prices_np.copy()

    if _von and validate_and_clean is not None:
        try:
            asset_class = None
            try:
                asset_class = canonical_instrument_type(symbol)
            except Exception:
                asset_class = None
            res = validate_and_clean(
                list(dates),
                list(map(float, prices_np.tolist())),
                symbol=symbol,
                policy=os.getenv("NVAR_VALIDATION_POLICY", "autofix"),
                asset_class=(asset_class or ""),
            )
            if isinstance(res, dict):
                nd = res.get("dates")
                np_prices = res.get("prices")
                rep = res.get("report")
                summ = res.get("summary")
                if (
                    isinstance(nd, list)
                    and isinstance(np_prices, list)
                    and len(nd) == len(np_prices)
                    and len(np_prices) >= 2
                ):
                    dates = nd
                    prices_np = np.asarray(np_prices, float)
                    anomalies = {
                        "report": rep if isinstance(rep, list) else [],
                        "summary": summ if isinstance(summ, dict) else {},
                        "policy": os.getenv(
                            "NVAR_VALIDATION_POLICY",
                            "autofix",
                        ),
                        "asset_class": asset_class,
                    }
        except Exception:  # noqa: BLE001
            anomalies = None

    # Optional NAV interpolation to daily grid (feature-flag)
    try:
        _nav_interp_on = (
            str(os.getenv("NVAR_NAV_INTERPOLATE", "0")).strip().lower()
            in ("1", "true", "yes", "on")
        )
    except Exception:
        _nav_interp_on = False
    if _nav_interp_on and callable(_ts_interpolate):
        try:
            d_list = list(map(str, list(dates)))
            p_list = list(map(float, prices_np.tolist()))
            d_new, p_new = _ts_interpolate(d_list, p_list)
            if (
                isinstance(d_new, list)
                and isinstance(p_new, list)
                and len(d_new) == len(p_new)
                and len(p_new) >= 2
            ):
                dates = d_new
                prices_np = np.asarray(p_new, float)
        except Exception:
            pass

    with np.errstate(divide="ignore", invalid="ignore"):
        rets_raw = np.diff(np.log(prices_np))
    mask = np.isfinite(rets_raw)
    rets = rets_raw[mask]

    dates0 = np.asarray(dates[:-1])[mask]
    dates1 = np.asarray(dates[1:])[mask]
    prices0 = prices_np[:-1][mask]
    prices1 = prices_np[1:][mask]

    try:
        eq_lb_env = int(_CONFIG.eq_lookback_days)
    except Exception:
        eq_lb_env = 0
    if (
        eq_lb_env > 0
        and not used_csv_branch
        and symbol not in {"BTC", "ETH", "SP500TR"}
    ):
        keep = min(len(prices_np), eq_lb_env + 1)
        if keep >= 2:
            prices_np = prices_np[-keep:]
            dates = dates[-keep:]
            with np.errstate(divide="ignore", invalid="ignore"):
                rets_raw = np.diff(np.log(prices_np))
            mask = np.isfinite(rets_raw)
            rets = rets_raw[mask]
            dates0 = np.asarray(dates[:-1])[mask]
            dates1 = np.asarray(dates[1:])[mask]
            prices0 = prices_np[:-1][mask]
            prices1 = prices_np[1:][mask]

    try:
        thr = float(os.getenv("NVAR_FATAL_LOGRET", "1.0"))
    except Exception:
        thr = 1.0
    try:
        min_lr = float(np.min(rets)) if rets.size else 0.0
        max_lr = float(np.max(rets)) if rets.size else 0.0
        if (abs(min_lr) >= thr) or (abs(max_lr) >= thr):
            if anomalies is None:
                anomalies = {
                    "report": [],
                    "summary": {},
                    "policy": os.getenv("NVAR_VALIDATION_POLICY", "autofix"),
                    "asset_class": None,
                }
            rep_obj = anomalies.get("report")
            rep = rep_obj if isinstance(rep_obj, list) else []
            rep = [x for x in rep if isinstance(x, dict)]
            rep.append(
                {
                    "type": "extreme_log_return",
                    "severity": "fatal",
                    "min": float(min_lr),
                    "max": float(max_lr),
                    "threshold": float(thr),
                }
            )
            anomalies["report"] = rep
    except Exception:  # noqa: BLE001
        pass

    # Merge helper for anomaly dicts
    def _merge_anomaly_payload(anm: dict | None) -> None:
        nonlocal anomalies
        if not isinstance(anm, dict):
            return
        if anomalies is None:
            anomalies = {
                "report": [],
                "summary": {},
                "policy": os.getenv("NVAR_VALIDATION_POLICY", "autofix"),
                "asset_class": None,
            }
        try:
            rep_dst = anomalies.get("report") if isinstance(anomalies, dict) else []
            if not isinstance(rep_dst, list):
                rep_dst = []
            rep_src = anm.get("report") if isinstance(anm.get("report"), list) else []
            for x in rep_src:
                if isinstance(x, dict):
                    rep_dst.append(x)
            anomalies["report"] = rep_dst
        except Exception:
            pass
        try:
            sm_dst = anomalies.get("summary") if isinstance(anomalies, dict) else {}
            if not isinstance(sm_dst, dict):
                sm_dst = {}
            sm_src = anm.get("summary") if isinstance(anm.get("summary"), dict) else {}
            if isinstance(sm_src, dict):
                sm_dst.update(sm_src)
            anomalies["summary"] = sm_dst
        except Exception:
            pass

    # Best-effort anomalies from shared lib (optional)
    try:
        # Long flatline on raw prices
        if callable(_an_long_flat):
            _merge_anomaly_payload(_an_long_flat(prices_np.tolist(), threshold_days=int(os.getenv("NVAR_FLATLINE_DAYS", "20"))))
    except Exception:
        pass
    try:
        # Illiquid spikes using log-returns and end-date volumes
        if callable(_an_illiquid):
            _merge_anomaly_payload(
                _an_illiquid(
                    rets.tolist(),
                    list(dates[1:]) if isinstance(dates, list) else dates1.tolist(),
                    (volumes_np.tolist() if volumes_np is not None else None),
                    ret_threshold=float(os.getenv("NVAR_ILLIQ_RET", "0.25")),
                    vol_threshold=float(os.getenv("NVAR_ILLIQ_VOL", "0.0")),
                )
            )
    except Exception:
        pass
    try:
        # Inconsistent splits: extreme returns without declared splits while adjusted unavailable
        if callable(_an_incon_splits):
            _merge_anomaly_payload(
                _an_incon_splits(
                    rets.tolist(),
                    split_events if isinstance(split_events, list) else [],
                    threshold=float(os.getenv("NVAR_SPLIT_RET", "1.0")),
                )
            )
    except Exception:
        pass
    try:
        # Price field consistency: adjusted availability vs selected field and fallback usage
        if callable(_an_price_field):
            adjusted_available = bool(adj_positive > 0)
            _merge_anomaly_payload(
                _an_price_field(
                    adjusted_available,
                    price_field,
                    allow_close_fallback,
                    rets.tolist(),
                    split_events if isinstance(split_events, list) else [],
                )
            )
    except Exception:
        pass

    summary = {
        "observations": int(len(rets)),
        "years": round(len(rets) / TRADING_DAYS, 1),
        "price_range": {
            "min": float(prices_np.min()),
            "max": float(prices_np.max()),
            "start": float(prices_np[0]),
            "end": float(prices_np[-1]),
        },
        "total_return": float((prices_np[-1] / prices_np[0] - 1) * 100),
    }

    try:
        summary["annual_return"] = {
            "before_fix": _annual_from_prices(_orig_prices_for_ar),
            "after_fix": _annual_from_prices(prices_np),
        }
    except Exception:  # noqa: BLE001
        pass

    try:
        if anomalies is not None:
            s_obj = (
                anomalies.get("summary")
                if isinstance(anomalies.get("summary"), dict)
                else {}
            )
            if isinstance(s_obj, dict):
                s_obj.update(
                    {
                        "annual_return": {
                            "before_fix": _annual_from_prices(
                                _orig_prices_for_ar
                            ),
                            "after_fix": _annual_from_prices(prices_np),
                        }
                    }
                )
                anomalies["summary"] = s_obj
    except Exception:  # noqa: BLE001
        pass

    # Enforce minimum years using library gate (when enabled)
    try:
        min_years = float(_CONFIG.min_years)
    except Exception:
        min_years = 10.0
    enforce_years = bool(_CONFIG.enforce_min_years)
    if enforce_years:
        try:
            years_val = (
                measure_years_observations(rets.tolist())
                if callable(measure_years_observations)
                else float(summary.get("years", 0.0))
            )
        except Exception:
            years_val = float(summary.get("years", 0.0))
        try:
            if callable(enforce_min_years):
                enforce_min_years(years_val, min_years=min_years, eps=0.08)
            else:
                if (years_val + 0.08) < float(min_years):
                    raise InsufficientHistoryError(years_val, float(min_years))
        except InsufficientHistoryError as _e:  # type: ignore
            out_fail2: Dict[str, Any] = {
                "success": False,
                "error": str(_e),
                "code": "insufficient_history",
                "years": years_val,
                "min_years": float(min_years),
            }
            if anomalies is not None:
                try:
                    out_fail2["anomalies_report"] = anomalies
                except Exception:
                    pass
            try:
                _logger.warning(
                    "%s: loader error: insufficient history: %.1fy (<%.1fy)",
                    symbol,
                    years_val,
                    float(min_years),
                )
            except Exception:
                pass
            return out_fail2

    # Attach coverage metrics and liquidity decision (best-effort)
    try:
        if _ts_cov_last252 and _ts_cov_per_year and _ts_liq_decide:
            dates_np = np.asarray(dates, dtype="datetime64[D]")
            last252 = _ts_cov_last252(rets)
            per_year = _ts_cov_per_year(dates_np, rets)
            # Extended diagnostics
            min_days = None
            try:
                if _ts_min_days_12m:
                    min_days = _ts_min_days_12m(dates_np)
            except Exception:
                min_days = None
            liq = _ts_liq_decide(
                last252,
                per_year,
                min_trading_days_last_12m=min_days,
            )
            summary["last_252"] = {
                "n_obs": int(last252.n_obs),
                "n_nonzero": int(last252.n_nonzero),
                "zero_share": float(last252.zero_share),
                "dropped_points_recent_total": int(
                    last252.dropped_points_recent_total
                ),
                "dropped_points_recent_with_flag": int(
                    last252.dropped_points_recent_with_flag
                ),
                "resampled": bool(last252.resampled),
                "unique_prices": getattr(last252, "unique_prices", None),
                "plateau_share": getattr(last252, "plateau_share", None),
            }
            summary["coverage_by_year"] = {
                int(k): int(v) for k, v in per_year.items()
            }
            summary["liquidity_decision"] = {
                "status": liq.status,
                "reasons": list(liq.reasons),
                "min_trading_days_last_12m": min_days,
            }
    except Exception:  # noqa: BLE001
        pass

    out: Dict[str, Any] = {
        "success": True,
        "returns": rets,
        "as_of_date": str(dates[-1]),
        "start_date": str(dates[0]) if len(dates) else None,
        "summary": summary,
    }
    if anomalies is not None:
        out["anomalies_report"] = anomalies
    try:
        _logger.info(
            "load ok: sym=%s n_returns=%d years=%.2f as_of=%s",
            symbol,
            int(rets.size),
            float(rets.size) / 252.0,
            str(out.get("as_of_date")),
        )
    except Exception:
        pass
    return out
