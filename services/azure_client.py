from __future__ import annotations
import os
import time
from typing import Any, Dict

import requests  # type: ignore


def call_cvar_calculate(
    symbol: str,
    alpha: float,
) -> Dict[str, Any]:
    """Call Azure Function /cvar/calculate with best-effort auth.

    Returns a dict parsed from JSON or an error dict on failure.
    """
    # Prefer unified env used across backend: NVAR_FUNC_URL
    base = (
        os.getenv("NVAR_FUNC_URL")
        or os.getenv("AZ_FUNC_BASE_URL")
        or os.getenv("NIR_FUNC_URL")
    )
    if not base:
        return {"_error": True, "error": "no_func_url"}
    b = str(base).strip()
    if not (b.startswith("http://") or b.startswith("https://")):
        b = "https://" + b
    # Azure Functions HTTP endpoints are under /api
    url = b.rstrip("/") + "/api/cvar/calculate"

    # Support SYMBOL:SUFFIX format
    payload_symbol = symbol
    payload_suffix = None
    if ":" in symbol:
        try:
            s_base, s_suf = symbol.split(":", 1)
            payload_symbol = s_base
            payload_suffix = s_suf
        except Exception:
            payload_symbol = symbol
            payload_suffix = None

    payload: Dict[str, Any] = {
        "symbol": payload_symbol,
        "alpha": float(alpha),
        "force": True,
    }
    if payload_suffix:
        payload["suffix"] = payload_suffix

    headers: dict[str, str] = {}
    params: dict[str, str] | None = None
    bearer = os.getenv("AZ_FUNC_BEARER") or os.getenv("NIR_FUNC_BEARER")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    else:
        # Prefer unified env key first
        key = (
            os.getenv("NVAR_FUNC_KEY")
            or os.getenv("AZ_FUNC_KEY")
            or os.getenv("NIR_FUNC_KEY")
            or os.getenv("AZ_FUNC_CODE")
            or os.getenv("NIR_FUNC_CODE")
        )
        if key:
            headers["x-functions-key"] = key
            params = {"code": key}

    t0 = time.time()
    try:
        resp = requests.post(
            url,
            json=payload,
            params=params,
            headers=headers,
            timeout=max(30, int(os.getenv("NIR_HTTP_TIMEOUT", "60"))),
        )
        if resp.status_code == 422:
            try:
                data_e = resp.json()
            except Exception:
                data_e = {"error": "insufficient_history"}
            return {
                "_error": True,
                "code": data_e.get("code", "insufficient_history"),
                "error": data_e.get("error", "insufficient history"),
            }
        resp.raise_for_status()
        data = resp.json()
        _ = time.time() - t0  # keep for potential diagnostics
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        return {"_error": True, "error": str(exc)}


