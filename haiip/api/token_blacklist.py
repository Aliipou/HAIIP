"""JWT token revocation — in-memory blacklist with optional Redis backing.

When a user logs out or an admin revokes a session, the token JTI (JWT ID)
is added to this blacklist. Every auth check verifies the token is not revoked.

Design:
    - Primary: Redis SET with TTL = token remaining lifetime
    - Fallback: In-memory set (single-process only — loses state on restart)
    - Revocation takes effect immediately (no grace period)
    - Thread-safe for concurrent reads/writes

EU AI Act Art. 9 — Risk management:
    Token revocation ensures compromised accounts cannot continue to access
    the AI system after incident detection.

Usage::
    from haiip.api.token_blacklist import blacklist

    # On logout / revocation:
    await blacklist.revoke(jti, expires_in_seconds=1800)

    # In auth middleware:
    if await blacklist.is_revoked(jti):
        raise HTTPException(401, "Token revoked")
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class TokenBlacklist:
    """JWT revocation list — Redis-backed with in-memory fallback.

    Thread-safe for asyncio. Not safe for multi-process without Redis.
    """

    def __init__(self) -> None:
        # In-memory: {jti: expires_at_epoch}
        self._store: dict[str, float] = {}
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        """Lazy Redis connection (returns None if Redis unavailable)."""
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            from haiip.api.config import get_settings
            settings = get_settings()
            url = getattr(settings, "redis_url", None)
            if url:
                self._redis = await aioredis.from_url(url, decode_responses=True)
        except Exception:
            pass
        return self._redis

    async def revoke(self, jti: str, expires_in_seconds: int) -> None:
        """Mark a JWT JTI as revoked until its expiry.

        Args:
            jti:                JWT ID (jti claim)
            expires_in_seconds: Seconds until the token would naturally expire
        """
        redis = await self._get_redis()
        if redis is not None:
            try:
                await redis.setex(f"revoked:{jti}", expires_in_seconds, "1")
                logger.info("token_revoked", extra={"jti_prefix": jti[:8]})
                return
            except Exception as exc:
                logger.warning("redis_revoke_failed", extra={"error": str(exc)})

        # Fallback: in-memory
        self._store[jti] = time.monotonic() + expires_in_seconds
        self._prune()
        logger.info("token_revoked_memory", extra={"jti_prefix": jti[:8]})

    async def is_revoked(self, jti: str) -> bool:
        """Return True if this JTI has been revoked."""
        redis = await self._get_redis()
        if redis is not None:
            try:
                return bool(await redis.exists(f"revoked:{jti}"))
            except Exception:
                pass  # fall through to in-memory

        # Fallback: in-memory
        expires_at = self._store.get(jti)
        if expires_at is None:
            return False
        if time.monotonic() > expires_at:
            del self._store[jti]
            return False
        return True

    def _prune(self) -> None:
        """Remove expired entries from in-memory store."""
        now     = time.monotonic()
        expired = [k for k, v in self._store.items() if v < now]
        for k in expired:
            del self._store[k]


# Process-wide singleton
blacklist = TokenBlacklist()
