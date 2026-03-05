"""AWS Secrets Manager integration with env-var fallback.

Strategy:
  - Production (APP_ENV=production): fetch a JSON blob from AWS Secrets Manager
    and inject each key as an environment variable BEFORE get_settings() is called.
  - Development / CI: no-op — .env.local / environment variables are used as-is.

Usage (in production entrypoint, before FastAPI app is imported):

    from haiip.api.secrets import inject_aws_secrets
    inject_aws_secrets()   # populates os.environ from AWS
    # then normal app startup proceeds, get_settings() reads from os.environ

AWS Secrets Manager secret format (JSON string):
    {
        "SECRET_KEY": "...",
        "DATABASE_URL": "postgresql+asyncpg://...",
        "REDIS_URL": "redis://...:6379/0",
        "OPENAI_API_KEY": "...",
        "GROQ_API_KEY": "..."
    }

References:
    - AWS Secrets Manager best practices: docs.aws.amazon.com/secretsmanager
    - 12-Factor App: config in environment (12factor.net/config)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# Override via HAIIP_AWS_SECRETS_NAME env var if needed
DEFAULT_SECRET_NAME = os.getenv("HAIIP_AWS_SECRETS_NAME", "haiip/production/config")
DEFAULT_AWS_REGION  = os.getenv("AWS_REGION", "eu-north-1")

# In-memory cache: (secret_dict, fetched_at_epoch)
_cache: dict[str, tuple[dict[str, str], float]] = {}
_CACHE_TTL_SECONDS = 300  # refresh every 5 minutes


# ── Public API ─────────────────────────────────────────────────────────────────

def inject_aws_secrets(
    secret_name: str = DEFAULT_SECRET_NAME,
    region: str = DEFAULT_AWS_REGION,
    ttl: int = _CACHE_TTL_SECONDS,
    overwrite_existing: bool = False,
) -> dict[str, str]:
    """Fetch secrets from AWS Secrets Manager and inject into os.environ.

    Args:
        secret_name:        Full ARN or friendly name of the secret.
        region:             AWS region where the secret lives.
        ttl:                Cache TTL in seconds — avoids repeated API calls.
        overwrite_existing: If False (default), skip keys already in os.environ.
                            Set True to force AWS values to win.

    Returns:
        Dict of key→value that were injected (for logging/debugging — values
        are never logged by this function itself).

    Raises:
        RuntimeError: if APP_ENV=production and secret cannot be fetched.
    """
    app_env = os.getenv("APP_ENV", "development")
    is_production = app_env == "production"

    secrets = _fetch_with_cache(secret_name, region, ttl)
    if not secrets:
        if is_production:
            raise RuntimeError(
                f"Production environment requires secrets from AWS Secrets Manager "
                f"(secret: {secret_name!r}, region: {region!r}) "
                f"but none were returned. Check IAM permissions and secret name."
            )
        logger.debug("secrets.inject: no secrets fetched (dev mode — using env vars)")
        return {}

    injected: dict[str, str] = {}
    for key, value in secrets.items():
        if not isinstance(value, str):
            value = str(value)
        if overwrite_existing or key not in os.environ:
            os.environ[key] = value
            injected[key] = value

    logger.info(
        "secrets.inject: injected %d keys from %r (overwrite=%s)",
        len(injected),
        secret_name,
        overwrite_existing,
    )
    return injected


def get_secret_value(key: str, default: str = "") -> str:
    """Return a single secret value, with an env-var / default fallback.

    Checks in order:
      1. os.environ (already injected or set externally)
      2. AWS Secrets Manager cache (if already populated)
      3. default
    """
    if key in os.environ:
        return os.environ[key]

    cached = _get_cache(DEFAULT_SECRET_NAME)
    if cached and key in cached:
        return cached[key]

    return default


def clear_cache() -> None:
    """Evict all cached secrets (useful in tests)."""
    _cache.clear()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fetch_with_cache(secret_name: str, region: str, ttl: int) -> dict[str, str]:
    """Return cached secrets dict, refreshing from AWS if TTL expired."""
    cached = _get_cache(secret_name)
    if cached is not None:
        return cached

    return _fetch_from_aws(secret_name, region)


def _get_cache(secret_name: str) -> dict[str, str] | None:
    if secret_name not in _cache:
        return None
    secrets, fetched_at = _cache[secret_name]
    if time.monotonic() - fetched_at > _CACHE_TTL_SECONDS:
        del _cache[secret_name]
        return None
    return secrets


def _fetch_from_aws(secret_name: str, region: str) -> dict[str, str]:
    """Call boto3 to retrieve and parse the secret.  Returns {} on any failure in dev."""
    try:
        import boto3  # type: ignore[import]
        from botocore.exceptions import ClientError  # type: ignore[import]
    except ImportError:
        logger.debug("boto3 not installed — AWS Secrets Manager unavailable")
        return {}

    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
    except Exception as exc:  # noqa: BLE001
        app_env = os.getenv("APP_ENV", "development")
        if app_env == "production":
            raise RuntimeError(f"Failed to fetch secret {secret_name!r}: {exc}") from exc
        logger.debug("secrets.aws: fetch failed (non-prod, continuing): %s", exc)
        return {}

    raw = response.get("SecretString") or ""
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("secrets.aws: secret %r is not valid JSON — treating as empty", secret_name)
        return {}

    result = {k: str(v) for k, v in parsed.items() if isinstance(k, str)}
    _cache[secret_name] = (result, time.monotonic())
    return result
