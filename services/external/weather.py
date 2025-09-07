from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import time
import requests  # type: ignore


_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_TTL_SEC = 600.0  # 10 minutes


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    try:
        exp, data = _CACHE.get(key, (0.0, {}))
        if exp >= time.time():
            return dict(data)
    except Exception:
        return None
    return None


def _cache_put(key: str, data: Dict[str, Any]) -> None:
    try:
        _CACHE[key] = (time.time() + _TTL_SEC, dict(data))
    except Exception:
        pass


def _geocode(name: str) -> Optional[Dict[str, Any]]:
    q = (name or "").strip()
    if not q:
        return None
    key = f"geo::{q.lower()}"
    hit = _cache_get(key)
    if hit is not None:
        return hit
    try:
        url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={requests.utils.quote(q)}&count=1&language=en&format=json"
        )
        resp = requests.get(url, timeout=(3, 8))
        data = resp.json() if resp.ok else {}
        arr = data.get("results") or []
        if not arr:
            return None
        r0 = arr[0]
        out = {
            "name": r0.get("name"),
            "country": r0.get("country"),
            "lat": r0.get("latitude"),
            "lon": r0.get("longitude"),
        }
        _cache_put(key, out)
        return out
    except Exception:
        return None


def _weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    key = f"wx::{lat:.4f},{lon:.4f}"
    hit = _cache_get(key)
    if hit is not None:
        return hit
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        resp = requests.get(url, timeout=(3, 8))
        data = resp.json() if resp.ok else {}
        cur = data.get("current_weather") or {}
        if not cur:
            return None
        out = {
            "temperature_c": cur.get("temperature"),
            "windspeed_ms": cur.get("windspeed"),
            "weathercode": cur.get("weathercode"),
            "time": cur.get("time"),
        }
        _cache_put(key, out)
        return out
    except Exception:
        return None


_WX_DESC = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    95: "thunderstorm",
}


def get_current_by_text(text: str) -> Optional[Dict[str, Any]]:
    g = _geocode(text)
    if not g or g.get("lat") is None or g.get("lon") is None:
        return None
    w = _weather(float(g["lat"]), float(g["lon"]))
    if not w:
        return None
    code = w.get("weathercode")
    desc = _WX_DESC.get(int(code)) if isinstance(code, int) else None
    return {
        "city": g.get("name"),
        "country": g.get("country"),
        "lat": g.get("lat"),
        "lon": g.get("lon"),
        "temperature_c": w.get("temperature_c"),
        "windspeed_ms": w.get("windspeed_ms"),
        "description": desc,
        "time": w.get("time"),
    }


