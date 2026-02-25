"""JWT authentication — token creation, verification, and password hashing.

Security decisions:
- HS256 with a strong secret key (minimum 32 chars enforced in config)
- Separate access token (short-lived) and refresh token (long-lived)
- Passwords hashed with bcrypt (cost factor 12)
- Token type claim prevents refresh tokens being used as access tokens
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from haiip.api.config import get_settings

settings = get_settings()

# bcrypt with cost factor 12 — ~0.3s per hash, brute-force resistant
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return bcrypt hash of plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches hashed. Constant-time comparison."""
    return _pwd_context.verify(plain, hashed)


# ── Token creation ────────────────────────────────────────────────────────────

def _create_token(
    subject: str,
    token_type: str,
    extra_claims: dict[str, Any],
    expires_delta: timedelta,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        **extra_claims,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(
    user_id: str,
    tenant_id: str,
    role: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create short-lived access token."""
    return _create_token(
        subject=user_id,
        token_type=TOKEN_TYPE_ACCESS,
        extra_claims={"tenant_id": tenant_id, "role": role, **(extra or {})},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(user_id: str, tenant_id: str) -> str:
    """Create long-lived refresh token."""
    return _create_token(
        subject=user_id,
        token_type=TOKEN_TYPE_REFRESH,
        extra_claims={"tenant_id": tenant_id},
        expires_delta=timedelta(days=settings.refresh_token_expire_days),
    )


# ── Token verification ────────────────────────────────────────────────────────

class TokenError(Exception):
    """Raised when a token is invalid, expired, or of the wrong type."""


def decode_token(token: str, expected_type: str = TOKEN_TYPE_ACCESS) -> dict[str, Any]:
    """Decode and validate a JWT.

    Raises:
        TokenError: if the token is invalid, expired, or wrong type.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise TokenError(str(exc)) from exc

    if payload.get("type") != expected_type:
        raise TokenError(f"Expected token type '{expected_type}'")

    return payload


def decode_access_token(token: str) -> dict[str, Any]:
    return decode_token(token, TOKEN_TYPE_ACCESS)


def decode_refresh_token(token: str) -> dict[str, Any]:
    return decode_token(token, TOKEN_TYPE_REFRESH)
