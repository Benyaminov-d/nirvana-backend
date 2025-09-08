from __future__ import annotations
from datetime import datetime

from fastapi import APIRouter, HTTPException, Response, Request  # type: ignore
from pydantic import BaseModel, EmailStr  # type: ignore
from passlib.context import CryptContext  # type: ignore

from core.db import get_db_session
from core.models import User, AuthAttempt
from utils.auth import mint_jwt_token, verify_jwt_token

router = APIRouter(prefix="/auth", tags=["auth"])

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class SigninRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    ok: bool
    email: EmailStr
    token_set: bool


def _hash_password(raw: str) -> str:
    return _pwd_ctx.hash(raw)


def _verify_password(raw: str, hashed: str | None) -> bool:
    try:
        return bool(hashed) and _pwd_ctx.verify(raw, str(hashed))
    except Exception:
        return False


@router.post("/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, resp: Response) -> AuthResponse:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "database unavailable")
    try:
        email = payload.email.lower().strip()
        existing = (
            sess.query(User)
            .filter(User.email == email)
            .one_or_none()  # type: ignore
        )
        if existing is not None:
            raise HTTPException(409, "email already registered")
        user = User(
            email=email,
            password_hash=_hash_password(payload.password),
            email_verified=0,
        )
        sess.add(user)
        sess.commit()
        token = mint_jwt_token(f"u:{user.id}")
        if not token:
            raise HTTPException(500, "auth not configured")
        # Set HTTP-only cookie for user JWT
        resp.set_cookie(
            key="nir_user",
            value=token,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
        )
        return AuthResponse(ok=True, email=email, token_set=True)
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.post("/signin", response_model=AuthResponse)
def signin(payload: SigninRequest, resp: Response) -> AuthResponse:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "database unavailable")
    try:
        email = payload.email.lower().strip()
        # Rate limit: block after 5 fails in last 10 minutes by email or IP
        try:
            from datetime import datetime, timedelta
            ip = ""  # optional: extract from headers if behind proxy
            ten_min_ago = datetime.utcnow() - timedelta(minutes=10)
            q = (
                sess.query(AuthAttempt)
                .filter(AuthAttempt.timestamp_utc >= ten_min_ago)
                .filter(AuthAttempt.purpose == "signin")
            )
            if email:
                q = q.filter(AuthAttempt.email == email)  # type: ignore
            recent = q.all()
            fails = [a for a in recent if int(getattr(a, "success", 0)) == 0]
            if len(fails) >= 5:
                raise HTTPException(429, "too many attempts, try later")
        except HTTPException:
            raise
        except Exception:
            pass
        user = (
            sess.query(User)
            .filter(User.email == email)
            .one_or_none()  # type: ignore
        )
        ok = bool(
            user is not None
            and _verify_password(payload.password, user.password_hash)
        )
        try:
            att = AuthAttempt(
                email=email,
                ip=None,
                purpose="signin",
                success=1 if ok else 0,
            )
            sess.add(att)
            sess.commit()
        except Exception:
            pass
        if not ok:
            raise HTTPException(401, "invalid credentials")
        try:
            user.last_login_utc = datetime.utcnow()
            sess.merge(user)
            sess.commit()
        except Exception:
            pass
        token = mint_jwt_token(f"u:{user.id}")
        if not token:
            raise HTTPException(500, "auth not configured")
        resp.set_cookie(
            key="nir_user",
            value=token,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/",
        )
        return AuthResponse(ok=True, email=email, token_set=True)
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.post("/logout")
def logout(resp: Response) -> dict:
    # Expire the cookie immediately
    resp.delete_cookie("nir_user", path="/")
    return {"ok": True}


# New endpoint to check authentication status
@router.get("/status")
def auth_status(request: Request) -> dict:
    token = request.cookies.get("nir_user")
    if not token:
        return {"authenticated": False}
    
    jwt_data = verify_jwt_token(token)
    if not jwt_data or not jwt_data.get("sub"):
        return {"authenticated": False}
    
    try:
        # Parse user ID from JWT subject
        user_id_str = jwt_data.get("sub", "").split(":", 1)[1]
        user_id = int(user_id_str)
        
        # Fetch user from database to verify it exists
        sess = get_db_session()
        if sess is None:
            raise HTTPException(503, "database unavailable")
        
        try:
            user = sess.query(User).filter(User.id == user_id).one_or_none()
            if user is None:
                return {"authenticated": False}
                
            return {"authenticated": True, "email": user.email}
        finally:
            sess.close()
    except Exception:
        return {"authenticated": False}


class ResetRequest(BaseModel):
    email: EmailStr


class ResetConfirmRequest(BaseModel):
    token: str
    new_password: str


@router.post("/request-password-reset")
def request_password_reset(payload: ResetRequest) -> dict:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "database unavailable")
    try:
        email = payload.email.lower().strip()
        user = (
            sess.query(User)
            .filter(User.email == email)
            .one_or_none()  # type: ignore
        )
        if user is None:
            return {"ok": True}
        from secrets import token_urlsafe
        from datetime import datetime, timedelta
        user.password_reset_token = token_urlsafe(32)
        user.password_reset_expires_utc = datetime.utcnow() + timedelta(minutes=30)
        sess.merge(user)
        sess.commit()
        # Email sending is out-of-scope here; token stored on user
        return {"ok": True}
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.post("/reset-password")
def reset_password(payload: ResetConfirmRequest, resp: Response) -> dict:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "database unavailable")
    try:
        from datetime import datetime
        token = (payload.token or "").strip()
        if not token:
            raise HTTPException(400, "invalid token")
        user = (
            sess.query(User)
            .filter(User.password_reset_token == token)
            .one_or_none()  # type: ignore
        )
        if user is None or not user.password_reset_expires_utc or user.password_reset_expires_utc < datetime.utcnow():
            raise HTTPException(400, "invalid or expired token")
        user.password_hash = _hash_password(payload.new_password)
        user.password_reset_token = None
        user.password_reset_expires_utc = None
        sess.merge(user)
        sess.commit()
        # Optionally log in the user after reset
        jwt_token = mint_jwt_token(f"u:{user.id}")
        if jwt_token:
            resp.set_cookie(
                key="nir_user",
                value=jwt_token,
                httponly=True,
                secure=False,
                samesite="lax",
                path="/",
            )
        return {"ok": True}
    finally:
        try:
            sess.close()
        except Exception:
            pass


@router.get("/verify")
def verify_email(token: str, resp: Response) -> dict:
    sess = get_db_session()
    if sess is None:
        raise HTTPException(503, "database unavailable")
    try:
        t = (token or "").strip()
        if not t:
            raise HTTPException(400, "invalid token")
        user = (
            sess.query(User)
            .filter(User.verification_token == t)
            .one_or_none()  # type: ignore
        )
        if user is None:
            raise HTTPException(400, "invalid token")
        user.email_verified = 1
        user.verification_token = None
        sess.merge(user)
        sess.commit()
        jwt_token = mint_jwt_token(f"u:{user.id}")
        if jwt_token:
            resp.set_cookie(
                key="nir_user",
                value=jwt_token,
                httponly=True,
                secure=False,
                samesite="lax",
                path="/",
            )
        return {"ok": True}
    finally:
        try:
            sess.close()
        except Exception:
            pass