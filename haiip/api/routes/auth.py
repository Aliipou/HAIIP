"""Auth routes — login, refresh, register, logout."""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from haiip.api.auth import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from haiip.api.config import get_settings
from haiip.api.deps import DB, AdminUser, CurrentUser
from haiip.api.models import Tenant, User
from haiip.api.schemas import (
    LoginRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)

router = APIRouter()
logger = structlog.get_logger(__name__)
settings = get_settings()


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DB) -> TokenResponse:
    """Authenticate a user and return access + refresh tokens."""
    # Look up tenant
    tenant_result = await db.execute(
        select(Tenant).where(Tenant.slug == body.tenant_slug, Tenant.is_active.is_(True))
    )
    tenant = tenant_result.scalar_one_or_none()
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Look up user — generic error to prevent user enumeration
    user_result = await db.execute(
        select(User).where(
            User.email == body.email,
            User.tenant_id == tenant.id,
            User.is_active.is_(True),
        )
    )
    user = user_result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Update last_login
    user.last_login = datetime.now(UTC)
    await db.flush()

    logger.info("auth.login", user_id=user.id, tenant_id=tenant.id)

    return TokenResponse(
        access_token=create_access_token(user.id, tenant.id, user.role),
        refresh_token=create_refresh_token(user.id, tenant.id),
        token_type="bearer",  # noqa: S106
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, db: DB) -> TokenResponse:
    """Exchange a refresh token for a new access token."""
    try:
        payload = decode_refresh_token(body.refresh_token)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from exc

    user_result = await db.execute(
        select(User).where(
            User.id == payload["sub"],
            User.is_active.is_(True),
        )
    )
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return TokenResponse(
        access_token=create_access_token(user.id, user.tenant_id, user.role),
        refresh_token=create_refresh_token(user.id, user.tenant_id),
        token_type="bearer",  # noqa: S106
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, current_user: AdminUser, db: DB) -> User:
    """Register a new user in the admin's tenant. Admin-only."""
    existing = await db.execute(
        select(User).where(
            User.email == body.email,
            User.tenant_id == current_user.tenant_id,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered in this tenant",
        )

    user = User(
        tenant_id=current_user.tenant_id,
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    logger.info("auth.register", user_id=user.id, role=user.role, by=current_user.id)
    return user


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser) -> User:
    """Return the currently authenticated user."""
    return current_user
