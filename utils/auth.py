from __future__ import annotations

import os
import time as _time
from typing import Optional

from fastapi import HTTPException, Request  # type: ignore
from datetime import datetime, timedelta

import jwt  # type: ignore


def jwt_secret() -> str | None:
    val = os.getenv("NIR_JWT_SECRET")
    return val if val else None


def jwt_algorithm() -> str:
    return os.getenv("NIR_JWT_ALG", "HS256")


def mint_jwt_token(
    subject: str,
    *,
    ttl_minutes: int = 60 * 24,
) -> str | None:
    secret = jwt_secret()
    if not secret:
        return None
    try:
        now = datetime.utcnow()
        payload = {
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int(
                (now + timedelta(minutes=ttl_minutes)).timestamp()
            ),
        }
        token = jwt.encode(payload, secret, algorithm=jwt_algorithm())
        return (
            token if isinstance(token, str)
            else token.decode("utf-8")  # type: ignore[attr-defined]
        )
    except Exception:
        return None


def verify_jwt_token(token: str) -> dict | None:
    secret = jwt_secret()
    if not secret:
        return None
    try:
        data = jwt.decode(
            token,
            secret,
            algorithms=[jwt_algorithm()],
        )
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def public_secret() -> Optional[str]:
    val = os.getenv("NIR_PUBLIC_TOKEN_SECRET")
    return val if val else None


def mint_public_token() -> Optional[str]:
    secret = public_secret()
    if not secret:
        return None
    try:
        import hmac
        import hashlib

        ts = str(int(_time.time()))
        sig = hmac.new(
            secret.encode("utf-8"), ts.encode("utf-8"), hashlib.sha256
        )
        return f"{ts}:{sig.hexdigest()}"
    except Exception:
        return None


def check_public_token(raw: Optional[str]) -> bool:
    secret = public_secret()
    if not secret or not raw:
        return False
    try:
        import hmac
        import hashlib

        parts = raw.split(":", 1)
        if len(parts) != 2:
            return False
        ts_s, sig_hex = parts
        ts = int(ts_s)
        if abs(int(_time.time()) - ts) > 600:
            return False
        exp = hmac.new(
            secret.encode("utf-8"), ts_s.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        import secrets as _secrets
        return _secrets.compare_digest(sig_hex, exp)
    except Exception:
        return False


def basic_auth_if_configured(request: Request) -> None:
    user = os.getenv("BASIC_AUTH_USER", "")
    pwd = os.getenv("BASIC_AUTH_PASS", "")
    if not user or not pwd:
        return
    auth = request.headers.get("authorization")
    challenge = {"WWW-Authenticate": 'Basic realm="Nirvana"'}
    if not auth or not auth.lower().startswith("basic "):
        raise HTTPException(401, "Unauthorized", headers=challenge)
    try:
        import base64  # type: ignore
        import secrets  # type: ignore
        decoded = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
        if ":" not in decoded:
            raise ValueError("malformed")
        u, p = decoded.split(":", 1)
        if not (
            secrets.compare_digest(u, user)
            and secrets.compare_digest(p, pwd)
        ):
            raise ValueError("badcreds")
    except Exception:
        raise HTTPException(401, "Unauthorized", headers=challenge)


def require_pub_or_basic(request: Request) -> None:
    token = request.cookies.get("nir_pub")
    if check_public_token(token):
        return
    # Accept user JWT in cookie if present
    try:
        user_jwt = request.cookies.get("nir_user")
        if user_jwt and verify_jwt_token(user_jwt):
            return
    except Exception:
        pass
    try:
        allowed = [
            h.strip().lower()
            for h in (os.getenv("NIR_ALLOWED_ORIGINS", "localhost").split(","))
            if h.strip()
        ]
        if allowed:
            import urllib.parse as _url
            origin = request.headers.get("origin") or ""
            referer = request.headers.get("referer") or ""

            def _host(u: str) -> str:
                try:
                    return (_url.urlparse(u).hostname or "").lower()
                except Exception:
                    return ""
            if _host(origin) in allowed or _host(referer) in allowed:
                return
    except Exception:
        pass
    basic_auth_if_configured(request)
