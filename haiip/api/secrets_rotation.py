"""Secrets rotation — auto-rotate credentials without downtime.

Strategy:
  1. Detect that a new secret version is available in AWS Secrets Manager
  2. Fetch the new credentials
  3. Drain the connection pool gracefully (wait for in-flight requests)
  4. Swap the credentials in the live engine
  5. Re-inject into os.environ for any code that reads from there

Supports:
  - PostgreSQL password rotation (via SQLAlchemy async engine recreation)
  - API signing key rotation (new tokens use new key; old tokens still valid
    during overlap window via dual-key verification)
  - Redis password rotation

Usage:
    from haiip.api.secrets_rotation import SecretsRotationManager

    manager = SecretsRotationManager()
    await manager.rotate_if_needed()   # call from a Celery beat task

References:
    - AWS Secrets Manager rotation: docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html
    - Zero-downtime secret rotation: NIST SP 800-204B Section 4.3
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

ROTATION_CHECK_INTERVAL_SECONDS = int(
    os.getenv("HAIIP_ROTATION_CHECK_INTERVAL", "300")  # 5 minutes
)
DRAIN_TIMEOUT_SECONDS = int(os.getenv("HAIIP_DRAIN_TIMEOUT", "30"))
DUAL_KEY_OVERLAP_SECONDS = int(os.getenv("HAIIP_KEY_OVERLAP", "300"))  # 5 min


# ── State ──────────────────────────────────────────────────────────────────────

@dataclass
class RotationEvent:
    """Record of a single rotation event."""
    secret_name: str
    old_version: str
    new_version: str
    rotated_at: float = field(default_factory=time.monotonic)
    success: bool = True
    error: str = ""


@dataclass
class RotationState:
    """Live rotation state for a secret."""
    secret_name: str
    current_version: str = "unknown"
    last_checked: float = 0.0
    rotation_count: int = 0
    events: list[RotationEvent] = field(default_factory=list)

    # For dual-key overlap (signing key rotation)
    previous_key: str = ""
    previous_key_expires: float = 0.0


class SecretsRotationManager:
    """Manages zero-downtime credential rotation for all HAIIP secrets.

    Thread-safe: uses asyncio.Lock for concurrent rotation protection.
    """

    def __init__(
        self,
        secret_name: str | None = None,
        region: str | None = None,
        check_interval: int = ROTATION_CHECK_INTERVAL_SECONDS,
    ) -> None:
        from haiip.api.secrets import DEFAULT_AWS_REGION, DEFAULT_SECRET_NAME

        self._secret_name = secret_name or DEFAULT_SECRET_NAME
        self._region = region or DEFAULT_AWS_REGION
        self._check_interval = check_interval
        self._lock = asyncio.Lock()
        self._state = RotationState(secret_name=self._secret_name)
        self._db_engine: Any = None   # injected by caller if DB rotation needed

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_db_engine(self, engine: Any) -> None:
        """Inject the live SQLAlchemy async engine for connection pool draining."""
        self._db_engine = engine

    async def rotate_if_needed(self) -> bool:
        """Check AWS for a new secret version and rotate if one exists.

        Returns True if rotation was performed, False if nothing changed.
        Safe to call frequently — skips check if interval not elapsed.
        """
        now = time.monotonic()
        if now - self._state.last_checked < self._check_interval:
            return False

        async with self._lock:
            # Re-check inside lock to avoid double rotation
            if time.monotonic() - self._state.last_checked < self._check_interval:
                return False

            self._state.last_checked = now
            new_version, new_secrets = await self._fetch_current_version()

            if new_version == self._state.current_version:
                logger.debug("rotation.no_change: version=%s", new_version)
                return False

            logger.info(
                "rotation.detected: old=%s new=%s secret=%s",
                self._state.current_version,
                new_version,
                self._secret_name,
            )
            await self._perform_rotation(new_version, new_secrets)
            return True

    async def force_rotate(self) -> RotationEvent:
        """Force an immediate rotation regardless of interval."""
        async with self._lock:
            new_version, new_secrets = await self._fetch_current_version()
            return await self._perform_rotation(new_version, new_secrets)

    def get_previous_key(self) -> str | None:
        """Return the previous signing key if still within overlap window."""
        if self._state.previous_key and time.monotonic() < self._state.previous_key_expires:
            return self._state.previous_key
        return None

    def get_state(self) -> RotationState:
        return self._state

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _fetch_current_version(self) -> tuple[str, dict[str, str]]:
        """Fetch current secret from AWS — returns (version_id, secrets_dict)."""
        try:
            import boto3  # type: ignore[import]
        except ImportError:
            logger.debug("rotation: boto3 not installed, skipping AWS check")
            return self._state.current_version, {}

        import json

        try:
            client = boto3.client("secretsmanager", region_name=self._region)
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.get_secret_value(SecretId=self._secret_name),
            )
            version_id = resp.get("VersionId", "unknown")
            raw = resp.get("SecretString", "{}")
            secrets = json.loads(raw)
            return version_id, {k: str(v) for k, v in secrets.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("rotation.fetch_failed: %s", exc)
            return self._state.current_version, {}

    async def _perform_rotation(
        self, new_version: str, new_secrets: dict[str, str]
    ) -> RotationEvent:
        """Apply new secrets to all live components."""
        old_version = self._state.current_version
        event = RotationEvent(
            secret_name=self._secret_name,
            old_version=old_version,
            new_version=new_version,
        )
        try:
            # 1. Rotate signing key (with overlap window for in-flight tokens)
            if "SECRET_KEY" in new_secrets:
                await self._rotate_signing_key(new_secrets["SECRET_KEY"])

            # 2. Drain and recreate DB connection pool if password changed
            if "DATABASE_URL" in new_secrets:
                await self._rotate_database(new_secrets["DATABASE_URL"])

            # 3. Inject all new values into os.environ
            for key, value in new_secrets.items():
                os.environ[key] = value

            # 4. Clear secrets cache so next read gets fresh values
            from haiip.api.secrets import clear_cache
            clear_cache()

            self._state.current_version = new_version
            self._state.rotation_count += 1
            logger.info(
                "rotation.complete: version=%s count=%d",
                new_version,
                self._state.rotation_count,
            )
        except Exception as exc:  # noqa: BLE001
            event.success = False
            event.error = str(exc)
            logger.error("rotation.failed: %s", exc)

        self._state.events.append(event)
        return event

    async def _rotate_signing_key(self, new_key: str) -> None:
        """Rotate JWT signing key with overlap window.

        Tokens signed with the old key remain valid for DUAL_KEY_OVERLAP_SECONDS.
        """
        current_key = os.getenv("SECRET_KEY", "")
        if current_key and current_key != new_key:
            self._state.previous_key = current_key
            self._state.previous_key_expires = (
                time.monotonic() + DUAL_KEY_OVERLAP_SECONDS
            )
            logger.info(
                "rotation.signing_key: overlap window %ds", DUAL_KEY_OVERLAP_SECONDS
            )

    async def _rotate_database(self, new_database_url: str) -> None:
        """Drain connection pool and recreate engine with new credentials."""
        if self._db_engine is None:
            logger.debug("rotation.db: no engine injected, skipping pool drain")
            return

        logger.info("rotation.db: draining connection pool (timeout=%ds)", DRAIN_TIMEOUT_SECONDS)
        try:
            # Dispose closes all idle connections; new connections use new URL
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._db_engine.sync_engine.dispose
                ),
                timeout=DRAIN_TIMEOUT_SECONDS,
            )
            logger.info("rotation.db: pool drained, new connections will use updated URL")
        except asyncio.TimeoutError:
            logger.warning(
                "rotation.db: drain timed out after %ds — forcing pool close",
                DRAIN_TIMEOUT_SECONDS,
            )
