"""Zero-trust service-to-service authentication.

Every internal microservice call must carry a signed service token.
Even internal APIs (worker → API, dashboard → API) require verification.

Architecture:
- Service tokens are short-lived HS256 JWTs (5 minutes)
- Signed with a separate SERVICE_SECRET_KEY (not the user JWT key)
- Claims: service name, issued-at, expiry, allowed scopes
- FastAPI dependency: require_service_token() — use on any internal route
- Middleware: ServiceAuthMiddleware — auto-verify on /internal/ prefix

Usage in routes:
    @router.post("/internal/retrain")
    async def retrain(svc: ServiceToken = Depends(require_service_token())):
        ...

Usage for outgoing calls (e.g., Celery worker → API):
    token = create_service_token("worker", scopes=["retrain", "predict"])
    headers = service_auth_headers(token)
    httpx.post("http://api:8000/internal/retrain", headers=headers)

References:
    - NIST SP 800-204B: Zero Trust for Microservices
    - IETF RFC 7519: JWT
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SERVICE_TOKEN_EXPIRE_MINUTES = 5
SERVICE_TOKEN_TYPE = "service"

# Known service identities — expand as new services are added
KNOWN_SERVICES = frozenset({"api", "worker", "dashboard", "scheduler", "test-harness"})


# ── Token creation ─────────────────────────────────────────────────────────────


def create_service_token(
    service_name: str,
    scopes: list[str] | None = None,
    expires_minutes: int = SERVICE_TOKEN_EXPIRE_MINUTES,
) -> str:
    """Create a short-lived service-to-service JWT.

    Args:
        service_name: Identifier of the calling service (must be in KNOWN_SERVICES).
        scopes:       List of allowed operations, e.g. ["retrain", "predict"].
        expires_minutes: Token TTL in minutes (default 5).

    Returns:
        Signed JWT string.

    Raises:
        ValueError: if service_name is not registered.
    """
    from jose import jwt

    from haiip.api.config import get_settings

    if service_name not in KNOWN_SERVICES:
        raise ValueError(
            f"Unknown service {service_name!r}. Registered services: {sorted(KNOWN_SERVICES)}"
        )

    settings = get_settings()
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": service_name,
        "type": SERVICE_TOKEN_TYPE,
        "scopes": scopes or [],
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    secret = _service_secret(settings)
    return jwt.encode(payload, secret, algorithm="HS256")


def service_auth_headers(token: str) -> dict[str, str]:
    """Return Authorization header dict for outgoing service calls."""
    return {"Authorization": f"Bearer {token}", "X-Service-Auth": "true"}


# ── Token verification ─────────────────────────────────────────────────────────


class ServiceTokenError(Exception):
    """Raised when a service token is invalid, expired, or unrecognised."""


class ServiceToken:
    """Parsed and validated service token payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.service_name: str = payload["sub"]
        self.scopes: list[str] = payload.get("scopes", [])
        self.issued_at: datetime = datetime.fromtimestamp(payload["iat"], tz=UTC)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def require_scope(self, scope: str) -> None:
        if not self.has_scope(scope):
            raise ServiceTokenError(f"Service {self.service_name!r} lacks required scope {scope!r}")

    def __repr__(self) -> str:
        return f"ServiceToken(service={self.service_name!r}, scopes={self.scopes})"


def verify_service_token(token: str) -> ServiceToken:
    """Decode and validate a service JWT.

    Raises:
        ServiceTokenError: on any validation failure.
    """
    from jose import JWTError, jwt

    from haiip.api.config import get_settings

    settings = get_settings()
    secret = _service_secret(settings)

    try:
        payload: dict[str, Any] = jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError as exc:
        raise ServiceTokenError(f"Service token decode failed: {exc}") from exc

    if payload.get("type") != SERVICE_TOKEN_TYPE:
        raise ServiceTokenError("Token is not a service token")

    service_name = payload.get("sub", "")
    if service_name not in KNOWN_SERVICES:
        raise ServiceTokenError(f"Unregistered service: {service_name!r}")

    return ServiceToken(payload)


# ── FastAPI dependency ─────────────────────────────────────────────────────────

_service_bearer = HTTPBearer(auto_error=False)


def require_service_token(required_scope: str | None = None):
    """Dependency factory — verifies service token, optionally checks scope.

    Usage:
        @router.post("/internal/retrain")
        async def retrain(svc: ServiceToken = Depends(require_service_token("retrain"))):
            ...
    """

    async def _verify(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Security(_service_bearer),
        ],
    ) -> ServiceToken:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Service token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            svc_token = verify_service_token(credentials.credentials)
        except ServiceTokenError as exc:
            logger.warning("service_auth.rejected: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service token",
            ) from exc

        if required_scope is not None:
            try:
                svc_token.require_scope(required_scope)
            except ServiceTokenError as exc:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=str(exc),
                ) from exc

        logger.debug(
            "service_auth.ok: service=%s scopes=%s",
            svc_token.service_name,
            svc_token.scopes,
        )
        return svc_token

    return _verify


# ── Helpers ────────────────────────────────────────────────────────────────────


def _service_secret(settings: Any) -> str:
    """Return the service signing secret.

    Uses SERVICE_SECRET_KEY if set, otherwise derives from SECRET_KEY + salt.
    In production, always set SERVICE_SECRET_KEY separately.
    """
    import os

    dedicated = os.getenv("SERVICE_SECRET_KEY", "")
    if dedicated:
        return dedicated
    # Fallback: derive a deterministic secret from the main key + salt
    # Not ideal for production — use dedicated secret
    return settings.secret_key + ":service-token-salt-v1"
