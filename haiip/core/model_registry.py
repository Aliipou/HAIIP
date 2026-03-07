"""Model version registry — writes to DB ModelRegistry on every train/promote event.

This is the missing link between AutoRetrainPipeline and the ModelRegistry table.
Every time a model is trained, saved, or promoted, call `register_model_version()`.

The `get_active_version()` function is used by the predict route to stamp
every prediction response with the currently active model version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# In-memory fallback when DB is not available (e.g. during tests / edge nodes)
_in_memory_registry: dict[str, dict[str, Any]] = {}


def _generate_version(model_name: str, artifact_path: str) -> str:
    """Generate a deterministic version string from model name + artifact hash.

    Format: v{YYYYMMDD}-{8-char-hash}
    Example: v20260307-3f2a1b8c
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path_hash = hashlib.sha256(f"{model_name}:{artifact_path}".encode()).hexdigest()[:8]
    return f"v{today}-{path_hash}"


def register_model_version(
    tenant_id: str,
    model_name: str,
    artifact_path: str | Path,
    metrics: dict[str, float] | None = None,
    dataset_hash: str | None = None,
    is_active: bool = True,
) -> str:
    """Register a model version in the in-memory registry (sync, no DB dependency).

    Call this from AutoRetrainPipeline after promotion, and from the train tasks.

    Args:
        tenant_id:    Tenant this model belongs to
        model_name:   "anomaly_detector" | "maintenance_predictor" | "anomaly_autoencoder" | ...
        artifact_path: Local path where model artifacts are saved
        metrics:      Evaluation metrics dict (F1, AUC, etc.)
        dataset_hash: SHA-256 of training dataset (for reproducibility)
        is_active:    Whether this version should be set as the active version

    Returns:
        version string, e.g. "v20260307-3f2a1b8c"
    """
    version = _generate_version(model_name, str(artifact_path))
    key = f"{tenant_id}:{model_name}"

    entry: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "model_name": model_name,
        "model_version": version,
        "artifact_path": str(artifact_path),
        "metrics": metrics or {},
        "is_active": is_active,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
    }

    if is_active:
        # Deactivate previous version
        prev = _in_memory_registry.get(key)
        if prev:
            prev["is_active"] = False
            _in_memory_registry[f"{key}:prev"] = prev

    _in_memory_registry[key] = entry
    logger.info(
        "Model version registered: tenant=%s model=%s version=%s active=%s",
        tenant_id, model_name, version, is_active,
    )

    # Emit Prometheus metric
    try:
        from haiip.api.ml_metrics import record_model_version
        record_model_version(tenant_id, model_name, version)
    except Exception:  # noqa: BLE001
        pass

    return version


def get_active_version(tenant_id: str, model_name: str) -> str:
    """Return the currently active version string for a given tenant + model."""
    key = f"{tenant_id}:{model_name}"
    entry = _in_memory_registry.get(key)
    if entry and entry.get("is_active"):
        return entry["model_version"]
    return "unregistered"


def get_version_history(tenant_id: str, model_name: str) -> list[dict[str, Any]]:
    """Return all registered versions for a tenant + model (current + previous)."""
    key = f"{tenant_id}:{model_name}"
    history = []
    for k, v in _in_memory_registry.items():
        if k.startswith(f"{tenant_id}:{model_name}"):
            history.append(v)
    return sorted(history, key=lambda x: x.get("trained_at", ""), reverse=True)


def list_active_versions(tenant_id: str) -> list[dict[str, Any]]:
    """Return all active model versions for a tenant."""
    return [
        v for k, v in _in_memory_registry.items()
        if v.get("tenant_id") == tenant_id and v.get("is_active")
    ]


async def sync_to_db(tenant_id: str, model_name: str, db: Any) -> None:
    """Persist the in-memory registry entry to the ModelRegistry DB table.

    Called asynchronously after registration to ensure DB consistency.
    """
    from haiip.api.models import ModelRegistry

    key = f"{tenant_id}:{model_name}"
    entry = _in_memory_registry.get(key)
    if not entry:
        return

    try:
        record = ModelRegistry(
            tenant_id=entry["tenant_id"],
            model_name=entry["model_name"],
            model_version=entry["model_version"],
            artifact_path=entry["artifact_path"],
            metrics=json.dumps(entry["metrics"]),
            is_active=entry["is_active"],
            dataset_hash=entry.get("dataset_hash"),
        )
        db.add(record)
        await db.flush()
        logger.info("Model version synced to DB: %s", entry["model_version"])
    except Exception as exc:  # noqa: BLE001
        logger.error("DB sync failed for model version: %s", exc)
