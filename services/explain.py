from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI  # type: ignore


_DISCLAIMER = (
    "Nirvana is not a licensed or certified investment advisor. "
    "All information is provided for educational and informational "
    "purposes only and does not constitute financial advice."
)


def _client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("NIR_OPENAI_KEY")
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("NIR_OPENAI_BASE")
    if not key:
        return None
    if base:
        return OpenAI(api_key=key, base_url=base)
    return OpenAI(api_key=key)


def generate_explanation(
    symbol: str,
    name: str | None,
    cvar: dict | None,
    meta: dict | None = None,
) -> dict:
    """Return dict with short_info, interpretation, disclaimer.

    - symbol: e.g., BTC
    - name:   e.g., Bitcoin
    - cvar:   payload from /cvar/curve-all (subset is fine)
    """
    # Fallback text if no LLM configured
    fallback_short = (
        f"{symbol}{' (' + name + ')' if name else ''}: no summary available."
    )

    # Build a compact context for interpretation
    def _worst(ann: dict | None) -> float | None:
        try:
            if not isinstance(ann, dict):
                return None
            vals = [ann.get("nig"), ann.get("ghst"), ann.get("evar")]
            floats = []
            for v in vals:
                try:
                    f = float(v) if v is not None else None
                except Exception:
                    f = None
                if f is not None and f == f:
                    floats.append(f)
            return max(floats) if floats else None
        except Exception:
            return None

    # Accept either full payload with annual blocks, or flat dict with c50/c95/c99
    if isinstance(cvar, dict) and all(k in cvar for k in ("c50", "c95", "c99")):
        c50 = cvar.get("c50")
        c95 = cvar.get("c95")
        c99 = cvar.get("c99")
    else:
        c50 = _worst((cvar or {}).get("cvar50", {}).get("annual"))
        c95 = _worst((cvar or {}).get("cvar95", {}).get("annual"))
        c99 = _worst((cvar or {}).get("cvar99", {}).get("annual"))

    prompt = (
        "You are a concise explainer. STRICT ORDER AND CONTENT RULES: "
        "1) short_info: 1-2 sentences describing the specific instrument "
        "(use symbol and name if present). Do NOT mention CVaR or any risk "
        "metrics here. 2) summary: one sentence about 99% CVaR if c99 is "
        "provided, phrased as '~1-in-100 years you could lose about X%'. "
        "3) interpretation: 2-3 sentences explaining how to read THESE CVaR "
        "values (c50/c95/c99) for this instrument. Do NOT define CVaR in "
        "the abstract; relate to the provided numbers. If meta has country, "
        "exchange, currency, type, ISIN, years, or total_return_pct, weave 1-2 "
        "useful bits of that context into short_info (no speculation). If "
        "meta.as_of is present, you may reference it as the data date. "
        "Keep neutral and educational; no advice; no headings; no bullets."
    )
    context = {
        "symbol": symbol,
        "name": name,
        "cvar50": c50,
        "cvar95": c95,
        "cvar99": c99,
        "meta": (meta or {}),
    }

    client = _client()
    if client is None:
        return {
            "short_info": fallback_short,
            "interpretation": (
                "CVaR shows expected loss in adverse scenarios. The 95% CVaR "
                "represents an average loss across the worst ~1-in-20 years; "
                "99% CVaR across the worst ~1-in-100 years. Values are shown "
                "as positive fractions (e.g., 0.25 = 25%)."
            ),
            "disclaimer": _DISCLAIMER,
            "from_llm": False,
        }

    try:
        sys = (
            "Return only valid JSON with keys exactly: short_info, summary, interpretation. "
            "short_info MUST NOT mention CVaR; summary MUST be one sentence; "
            "interpretation MUST refer to the provided numbers and MUST NOT start with a CVaR definition."
        )
        msg = [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
            {"role": "user", "content": str(context)},
        ]
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
        import json as _json

        parsed = _json.loads(txt)
        short_info = str(parsed.get("short_info") or fallback_short)
        interpretation = str(
            parsed.get("interpretation")
            or (
                "CVaR summarizes expected losses under adverse scenarios. "
                "Values are positive loss fractions."
            )
        )
        summary = str(parsed.get("summary") or "")
        return {
            "short_info": short_info,
            "interpretation": interpretation,
            "summary": summary,
            "disclaimer": _DISCLAIMER,
            "from_llm": True,
        }
    except Exception:
        return {
            "short_info": fallback_short,
            "interpretation": (
                "CVaR shows expected loss in adverse scenarios. The 95% CVaR "
                "represents an average loss across the worst ~1-in-20 years; "
                "99% CVaR across the worst ~1-in-100 years. Values are shown "
                "as positive fractions (e.g., 0.25 = 25%)."
            ),
            "summary": (
                f"99% CVaR suggests that in a ~1-in-100 year scenario, loss "
                f"could be about {int(round((float(c99) if c99 else 0)*100))}% of capital."
            ) if c99 is not None else "",
            "disclaimer": _DISCLAIMER,
            "from_llm": False,
        }

