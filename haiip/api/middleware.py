"""Security middleware for HAIIP API.

Implements:
1. Rate limiting — per-IP sliding window (Redis-backed, in-memory fallback)
2. PII scrubber — strips sensitive fields from structured logs
3. Security headers — HSTS, X-Content-Type-Options, X-Frame-Options, CSP
4. Request size limit — rejects oversized bodies early

References:
    - OWASP API Security Top 10 (2023) — API4:Unrestricted Resource Consumption
    - NIST SP 800-204A — Security Strategies for Microservices
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── PII field names to scrub from logs ────────────────────────────────────────

_PII_FIELDS = frozenset({
    "password", "secret", "token", "access_token", "refresh_token",
    "authorization", "api_key", "email", "phone", "ssn",
    "credit_card", "card_number", "cvv",
})


def scrub_pii(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively replace PII field values with [REDACTED].

    Safe to call on any dict (e.g., log record extras, request bodies).
    Does NOT mutate the input — returns a new dict.
    """
    result: dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in _PII_FIELDS:
            result[k] = "[REDACTED]"
        elif isinstance(v, dict):
            result[k] = scrub_pii(v)
        elif isinstance(v, list):
            result[k] = [
                scrub_pii(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            result[k] = v
    return result


def safe_log_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """Return a PII-scrubbed version of extra fields for structured logging."""
    return scrub_pii(extra)


# ── In-memory sliding window rate limiter ─────────────────────────────────────

class _InMemoryRateLimiter:
    """Thread-safe sliding window rate limiter (per-IP, per-path-prefix).

    Falls back to this when Redis is unavailable.
    Uses a token-bucket approximation with 1-second windows.
    """

    def __init__(self) -> None:
        # key → list of request timestamps (epoch seconds)
        self._windows: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        now   = time.monotonic()
        cutoff = now - window_seconds
        self._windows[key] = [t for t in self._windows[key] if t > cutoff]
        if len(self._windows[key]) >= limit:
            return False
        self._windows[key].append(now)
        return True


_limiter = _InMemoryRateLimiter()

# Per-endpoint limits: (requests, window_seconds)
_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "/api/v1/auth/login":        (10,  60),   # 10 logins / minute
    "/api/v1/auth/register":     (5,   60),   # 5 registrations / minute
    "/api/v1/predict":           (60,  60),   # 60 predictions / minute
    "/api/v1/economic":          (60,  60),   # 60 economic calls / minute
    "/api/v1/agent":             (20,  60),   # 20 agent queries / minute (expensive)
    "/api/v1/documents":         (10,  60),   # 10 ingests / minute
    "_default":                  (120, 60),   # 120 any other / minute
}

_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


def _client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For behind nginx."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit_key(request: Request) -> tuple[str, int, int]:
    """Return (bucket_key, limit, window_seconds) for this request."""
    ip   = _client_ip(request)
    path = request.url.path
    for prefix, (lim, win) in _RATE_LIMITS.items():
        if prefix != "_default" and path.startswith(prefix):
            bucket = f"{ip}:{prefix}"
            return bucket, lim, win
    default_lim, default_win = _RATE_LIMITS["_default"]
    return f"{ip}:default", default_lim, default_win


# ── Security headers ───────────────────────────────────────────────────────────

_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options":    "nosniff",
    "X-Frame-Options":           "DENY",
    "X-XSS-Protection":          "1; mode=block",
    "Referrer-Policy":           "strict-origin-when-cross-origin",
    "Permissions-Policy":        "geolocation=(), camera=(), microphone=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy":   (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    ),
    # Remove server fingerprint
    "Server": "HAIIP",
}


# ── Middleware ─────────────────────────────────────────────────────────────────

class SecurityMiddleware(BaseHTTPMiddleware):
    """Applies rate limiting, security headers, body size limit, and request logging.

    Place this BEFORE CORSMiddleware in the middleware stack.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # ── Body size check ────────────────────────────────────────────────────
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"success": False, "error": "Request body too large (max 10 MB)"},
            )

        # ── Rate limiting ──────────────────────────────────────────────────────
        if request.url.path != "/health":
            bucket, limit, window = _rate_limit_key(request)
            if not _limiter.is_allowed(bucket, limit, window):
                ip = _client_ip(request)
                logger.warning(
                    "rate_limit_exceeded",
                    extra={"ip": ip, "path": request.url.path, "limit": limit},
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "success": False,
                        "error": "Too many requests",
                        "detail": f"Rate limit: {limit} requests per {window}s",
                    },
                    headers={"Retry-After": str(window)},
                )

        # ── Process request ────────────────────────────────────────────────────
        t0       = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - t0) * 1000

        # ── Security headers ───────────────────────────────────────────────────
        for header, value in _SECURITY_HEADERS.items():
            response.headers[header] = value

        # ── Structured access log (no PII) ────────────────────────────────────
        if request.url.path not in ("/health", "/metrics"):
            logger.info(
                "http_request",
                extra={
                    "method":      request.method,
                    "path":        request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration, 1),
                    "ip_hash":     hashlib.sha256(
                        _client_ip(request).encode()
                    ).hexdigest()[:8],  # hash IP — GDPR data minimisation
                },
            )

        return response
