"""Tests for haiip/api/secrets.py — 100% branch coverage."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from haiip.api import secrets as secrets_module
from haiip.api.secrets import (
    _CACHE_TTL_SECONDS,
    _cache,
    _fetch_from_aws,
    _get_cache,
    clear_cache,
    get_secret_value,
    inject_aws_secrets,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_boto3_mock(secret_dict: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_secret_value.return_value = {"SecretString": json.dumps(secret_dict)}
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client
    return mock_boto3


# ── Fixtures ───────────────────────────────────────────────────────────────────

_INJECTED_KEYS = [
    "SECRET_KEY",
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "PORT",
    "ENABLED",
    "MY_KEY",
    "NONEXISTENT_KEY",
    "_HAIIP_SENTINEL",
]


@pytest.fixture(autouse=True)
def reset_state():
    """Clean os.environ and cache between every test."""
    # Pre-set sentinel to guarantee the restore (else) branch runs every teardown
    os.environ["_HAIIP_SENTINEL"] = "sentinel_value"

    clear_cache()
    saved = {k: os.environ.get(k) for k in _INJECTED_KEYS + ["APP_ENV"]}
    for k in _INJECTED_KEYS + ["APP_ENV"]:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v  # exercises else branch: _HAIIP_SENTINEL always has a value
    clear_cache()


# ── inject_aws_secrets ─────────────────────────────────────────────────────────


def test_inject_populates_os_environ():
    payload = {"SECRET_KEY": "prod-xyz", "DATABASE_URL": "postgresql://..."}
    with patch.dict(
        "sys.modules",
        {"boto3": _make_boto3_mock(payload), "botocore.exceptions": MagicMock()},
    ):
        injected = inject_aws_secrets()
    assert os.environ["SECRET_KEY"] == "prod-xyz"
    assert os.environ["DATABASE_URL"] == "postgresql://..."
    assert set(injected.keys()) == {"SECRET_KEY", "DATABASE_URL"}


def test_inject_does_not_overwrite_existing_by_default():
    os.environ["SECRET_KEY"] = "already-set"
    payload = {"SECRET_KEY": "from-aws"}
    with patch.dict(
        "sys.modules",
        {"boto3": _make_boto3_mock(payload), "botocore.exceptions": MagicMock()},
    ):
        injected = inject_aws_secrets()
    assert os.environ["SECRET_KEY"] == "already-set"
    assert "SECRET_KEY" not in injected


def test_inject_overwrites_when_flag_set():
    os.environ["SECRET_KEY"] = "already-set"
    payload = {"SECRET_KEY": "from-aws"}
    with patch.dict(
        "sys.modules",
        {"boto3": _make_boto3_mock(payload), "botocore.exceptions": MagicMock()},
    ):
        injected = inject_aws_secrets(overwrite_existing=True)
    assert os.environ["SECRET_KEY"] == "from-aws"
    assert "SECRET_KEY" in injected


def test_inject_returns_empty_when_no_boto3_in_dev():
    with patch.dict("sys.modules", {"boto3": None, "botocore": None, "botocore.exceptions": None}):
        result = inject_aws_secrets()
    assert result == {}


def test_inject_raises_in_production_when_boto3_missing():
    os.environ["APP_ENV"] = "production"
    with patch.dict("sys.modules", {"boto3": None, "botocore": None, "botocore.exceptions": None}):
        with pytest.raises(RuntimeError, match="Production environment requires"):
            inject_aws_secrets()


def test_inject_raises_in_production_when_aws_fails():
    os.environ["APP_ENV"] = "production"
    mock_client = MagicMock()
    mock_client.get_secret_value.side_effect = Exception("AccessDeniedException")
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client
    with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
        with pytest.raises(RuntimeError, match="Failed to fetch secret"):
            inject_aws_secrets()


def test_inject_aws_fails_in_dev_returns_empty():
    """In dev mode, AWS failure is swallowed and returns {}."""
    mock_client = MagicMock()
    mock_client.get_secret_value.side_effect = Exception("timeout")
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client
    with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
        result = inject_aws_secrets()
    assert result == {}


def test_inject_non_string_values_are_coerced():
    """_fetch_from_aws stringifies, but inject_aws_secrets has a defensive check too."""
    payload = {"PORT": 8000, "ENABLED": True}
    with patch.dict(
        "sys.modules",
        {"boto3": _make_boto3_mock(payload), "botocore.exceptions": MagicMock()},
    ):
        inject_aws_secrets()
    assert os.environ.get("PORT") == "8000"
    assert os.environ.get("ENABLED") == "True"


def test_inject_non_string_value_directly_in_cache():
    """Force secrets dict to contain non-str value to cover line 90 (value = str(value))."""
    secret_name = secrets_module.DEFAULT_SECRET_NAME
    # Inject a dict with a non-string value directly into cache, bypassing _fetch_from_aws
    _cache[secret_name] = ({"DIRECT_INT_KEY": 9999}, time.monotonic())  # type: ignore[dict-item]
    inject_aws_secrets()
    assert os.environ.get("DIRECT_INT_KEY") == "9999"


def test_inject_invalid_json_returns_empty():
    mock_client = MagicMock()
    mock_client.get_secret_value.return_value = {"SecretString": "not-json!!"}
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client
    with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
        result = inject_aws_secrets()
    assert result == {}


def test_inject_empty_secret_string_returns_empty():
    mock_client = MagicMock()
    mock_client.get_secret_value.return_value = {"SecretString": ""}
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client
    with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
        result = inject_aws_secrets()
    assert result == {}


# ── get_secret_value ───────────────────────────────────────────────────────────


def test_get_secret_value_from_env():
    os.environ["MY_KEY"] = "env-value"
    assert get_secret_value("MY_KEY") == "env-value"


def test_get_secret_value_from_cache():
    """Key not in os.environ but populated in cache → returns cached value."""
    secret_name = secrets_module.DEFAULT_SECRET_NAME
    _cache[secret_name] = ({"GROQ_API_KEY": "cached-groq"}, time.monotonic())
    os.environ.pop("GROQ_API_KEY", None)
    result = get_secret_value("GROQ_API_KEY")
    assert result == "cached-groq"


def test_get_secret_value_key_not_in_cache_falls_to_default():
    """Cache populated but does not contain the requested key → default."""
    secret_name = secrets_module.DEFAULT_SECRET_NAME
    _cache[secret_name] = ({"OTHER_KEY": "something"}, time.monotonic())
    os.environ.pop("NONEXISTENT_KEY", None)
    assert get_secret_value("NONEXISTENT_KEY", default="fallback") == "fallback"


def test_get_secret_value_default_when_missing():
    os.environ.pop("NONEXISTENT_KEY", None)
    assert get_secret_value("NONEXISTENT_KEY", default="my-default") == "my-default"


def test_get_secret_value_empty_default():
    os.environ.pop("NONEXISTENT_KEY", None)
    assert get_secret_value("NONEXISTENT_KEY") == ""


# ── _get_cache branches ────────────────────────────────────────────────────────


def test_get_cache_returns_none_when_not_present():
    assert _get_cache("nonexistent-secret") is None


def test_get_cache_returns_dict_when_fresh():
    name = "test-secret"
    data = {"KEY": "val"}
    _cache[name] = (data, time.monotonic())
    result = _get_cache(name)
    assert result == data


def test_get_cache_returns_none_when_ttl_expired():
    name = "expired-secret"
    data = {"KEY": "stale"}
    # Store with a timestamp far in the past
    _cache[name] = (data, time.monotonic() - (_CACHE_TTL_SECONDS + 1))
    result = _get_cache(name)
    assert result is None
    # Entry must be evicted
    assert name not in _cache


# ── Caching behaviour ──────────────────────────────────────────────────────────


def test_cache_prevents_double_aws_call():
    payload = {"SECRET_KEY": "cached-secret"}
    mock_boto3 = _make_boto3_mock(payload)
    with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
        inject_aws_secrets()
        inject_aws_secrets()  # second call — uses cache
    assert mock_boto3.client.call_count == 1


def test_clear_cache_forces_refetch():
    payload = {"SECRET_KEY": "cached-secret"}
    mock_boto3 = _make_boto3_mock(payload)
    with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
        inject_aws_secrets()
        clear_cache()
        inject_aws_secrets()
    assert mock_boto3.client.call_count == 2


# ── _fetch_from_aws directly ───────────────────────────────────────────────────


def test_fetch_from_aws_returns_dict():
    payload = {"SECRET_KEY": "direct-fetch"}
    with patch.dict(
        "sys.modules",
        {"boto3": _make_boto3_mock(payload), "botocore.exceptions": MagicMock()},
    ):
        result = _fetch_from_aws("haiip/production/config", "eu-north-1")
    assert result["SECRET_KEY"] == "direct-fetch"


def test_fetch_from_aws_returns_empty_without_boto3():
    with patch.dict("sys.modules", {"boto3": None, "botocore": None, "botocore.exceptions": None}):
        result = _fetch_from_aws("haiip/production/config", "eu-north-1")
    assert result == {}


def test_fetch_from_aws_caches_result():
    payload = {"KEY": "value"}
    secret_name = "test/cache-check"
    with patch.dict(
        "sys.modules",
        {"boto3": _make_boto3_mock(payload), "botocore.exceptions": MagicMock()},
    ):
        _fetch_from_aws(secret_name, "eu-north-1")
    assert secret_name in _cache
    assert _cache[secret_name][0] == {"KEY": "value"}
