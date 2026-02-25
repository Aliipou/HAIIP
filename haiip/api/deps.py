"""FastAPI dependency injection — all Depends() live here.

Rules:
- Never import from routes inside this file (circular import risk)
- Every dependency must be testable in isolation
- CurrentUser is the primary security gate — use it on every protected route
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from haiip.api.auth import TokenError, decode_access_token
from haiip.api.database import get_db
from haiip.api.models import User

_bearer = HTTPBearer(auto_error=True)


# ── Token extraction ──────────────────────────────────────────────────────────

def _extract_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
) -> str:
    return credentials.credentials


# ── Current user ──────────────────────────────────────────────────────────────

async def get_current_user(
    token: Annotated[str, Depends(_extract_token)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Decode JWT and return the matching User row.

    Raises 401 on any auth failure — never leaks why the token is invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str = payload["sub"]
        tenant_id: str = payload["tenant_id"]
    except (TokenError, KeyError):
        raise credentials_exception

    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.tenant_id == tenant_id,
            User.is_active.is_(True),
        )
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception

    return user


# ── Role-based access ─────────────────────────────────────────────────────────

def require_role(*roles: str):
    """Dependency factory — ensures current user has one of the given roles."""

    async def _check(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return _check


def require_admin() -> User:
    return Depends(require_role("admin"))


# ── Typed aliases (use these in route signatures) ─────────────────────────────

CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_role("admin"))]
EngineerUser = Annotated[User, Depends(require_role("admin", "engineer"))]
DB = Annotated[AsyncSession, Depends(get_db)]
