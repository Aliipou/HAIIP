"""ML Operations API routes — auto-retraining, ONNX export, latency benchmarking.

Endpoints:
    POST /ml-ops/retrain          — trigger auto-retraining for a tenant
    POST /ml-ops/export-onnx      — export champion model to ONNX
    GET  /ml-ops/benchmark        — run latency benchmark on ONNX model
    GET  /ml-ops/pipeline-status  — current AutoRetrainPipeline status + metrics

All endpoints require Engineer or Admin role (EngineerUser dep).
All actions are audit-logged.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from haiip.api.deps import EngineerUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-ops")


# ── Request / Response schemas ────────────────────────────────────────────────


class RetrainRequest(BaseModel):
    tenant_id: str = Field(default="default", description="Target tenant")
    feedback_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Current model accuracy (triggers if < threshold)",
    )
    force_reason: str | None = Field(
        default=None,
        description="Force retrain with this reason: manual | drift_critical | accuracy_drop | scheduled",
    )


class OnnxExportRequest(BaseModel):
    tenant_id: str = Field(default="default")
    model_type: str = Field(
        default="anomaly",
        description="Model to export: 'anomaly' or 'maintenance'",
    )
    opset: int = Field(default=17, ge=11, le=21, description="ONNX opset version")


class BenchmarkRequest(BaseModel):
    tenant_id: str = Field(default="default")
    model_type: str = Field(default="anomaly")
    n_runs: int = Field(default=100, ge=10, le=1000)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/retrain",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger automatic retraining",
    description=(
        "Enqueues an auto_retrain_pipeline Celery task for the specified tenant. "
        "The pipeline evaluates drift/accuracy triggers and runs champion-challenger "
        "evaluation, promoting the challenger if it outperforms the current champion."
    ),
)
async def trigger_retrain(
    request: RetrainRequest,
    current_user: EngineerUser,
) -> dict[str, Any]:
    try:
        from haiip.workers.tasks import auto_retrain_pipeline

        task = auto_retrain_pipeline.delay(
            tenant_id=request.tenant_id,
            feedback_accuracy=request.feedback_accuracy,
            force_reason=request.force_reason,
        )
        logger.info(
            "Retrain task enqueued: task_id=%s, tenant=%s, user=%s",
            task.id,
            request.tenant_id,
            current_user.email,
        )
        return {
            "task_id": task.id,
            "status": "queued",
            "message": f"Retraining enqueued for tenant '{request.tenant_id}'",
        }
    except Exception as exc:
        logger.error("Failed to enqueue retrain task: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Celery broker unavailable — retraining not enqueued",
        ) from exc


@router.post(
    "/export-onnx",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Export champion model to ONNX",
    description=(
        "Exports the current champion model to ONNX format for edge deployment. "
        "Compatible with ONNX Runtime 1.18+ on Jetson / Hailo / Industrial PC."
    ),
)
async def export_onnx(
    request: OnnxExportRequest,
    current_user: EngineerUser,
) -> dict[str, Any]:
    if request.model_type not in ("anomaly", "maintenance"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type must be 'anomaly' or 'maintenance'",
        )
    try:
        from haiip.workers.tasks import export_onnx_model

        task = export_onnx_model.delay(
            tenant_id=request.tenant_id,
            model_type=request.model_type,
            opset=request.opset,
        )
        logger.info(
            "ONNX export enqueued: task_id=%s, tenant=%s, model=%s",
            task.id,
            request.tenant_id,
            request.model_type,
        )
        return {
            "task_id": task.id,
            "status": "queued",
            "message": f"ONNX export enqueued for '{request.model_type}' model",
        }
    except Exception as exc:
        logger.error("Failed to enqueue ONNX export: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Celery broker unavailable — export not enqueued",
        ) from exc


@router.post(
    "/benchmark",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Benchmark ONNX model latency",
    description=(
        "Runs latency benchmark (p50/p95/p99) on the exported ONNX model. "
        "Verifies SLA compliance (≤50ms p99). Results include SLA pass/fail."
    ),
)
async def run_benchmark(
    request: BenchmarkRequest,
    current_user: EngineerUser,
) -> dict[str, Any]:
    try:
        from haiip.workers.tasks import benchmark_onnx_model

        task = benchmark_onnx_model.delay(
            tenant_id=request.tenant_id,
            model_type=request.model_type,
            n_runs=request.n_runs,
        )
        return {
            "task_id": task.id,
            "status": "queued",
            "message": f"Benchmark started: {request.n_runs} runs for '{request.model_type}'",
        }
    except Exception as exc:
        logger.error("Failed to enqueue benchmark: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Celery broker unavailable",
        ) from exc


@router.get(
    "/model-versions",
    summary="List active model versions",
    description=(
        "Returns the currently active model version for each registered model type. "
        "Every prediction response is stamped with this version for full auditability."
    ),
)
async def list_model_versions(
    tenant_id: str = "default",
    current_user: EngineerUser = Depends(),  # noqa: B008
) -> dict[str, Any]:
    from haiip.core.model_registry import get_active_version, list_active_versions

    active = list_active_versions(tenant_id)
    return {
        "tenant_id": tenant_id,
        "active_versions": active,
        "summary": {
            "anomaly_detector": get_active_version(tenant_id, "anomaly_detector"),
            "anomaly_autoencoder": get_active_version(tenant_id, "anomaly_autoencoder"),
            "maintenance_lstm": get_active_version(tenant_id, "maintenance_lstm"),
        },
    }


@router.get(
    "/pipeline-status",
    summary="Get ML pipeline health",
    description="Returns current model accuracy, drift status, and retraining history.",
)
async def pipeline_status(
    tenant_id: str = "default",
    current_user: EngineerUser = Depends(),  # noqa: B008
) -> dict[str, Any]:
    """Return latest pipeline state — no Celery required (reads artifact files)."""
    from pathlib import Path

    import numpy as np

    from haiip.api.config import get_settings

    settings = get_settings()
    artifact_dir = Path(settings.model_artifacts_path) / tenant_id

    has_champion = (artifact_dir / "anomaly" / "scaler.joblib").exists()
    has_autoencoder = (artifact_dir / "autoencoder" / "meta.npz").exists()
    has_onnx_anomaly = (artifact_dir / "onnx" / "anomaly_autoencoder.onnx").exists()
    has_onnx_maint = (artifact_dir / "onnx" / "maintenance_lstm.onnx").exists()

    drift_detected: bool | None = None
    drift_severity: str | None = None
    ref_path = artifact_dir / "drift_reference.npy"
    cur_path = artifact_dir / "drift_current.npy"

    if ref_path.exists() and cur_path.exists():
        try:
            from haiip.core.drift import DriftDetector

            X_ref = np.load(str(ref_path))
            X_cur = np.load(str(cur_path))
            dd = DriftDetector(drift_threshold=settings.drift_threshold)
            dd.fit_reference(X_ref)
            results = dd.check(X_cur)
            summary = dd.summary(results)
            drift_detected = summary["drift_detected"]
            drift_severity = summary["severity"]
        except Exception as exc:
            logger.warning("Drift check failed in pipeline_status: %s", exc)

    return {
        "tenant_id": tenant_id,
        "models": {
            "sklearn_champion": has_champion,
            "pytorch_autoencoder": has_autoencoder,
            "onnx_anomaly": has_onnx_anomaly,
            "onnx_maintenance": has_onnx_maint,
        },
        "drift": {
            "checked": drift_detected is not None,
            "detected": drift_detected,
            "severity": drift_severity,
        },
        "sla_target_ms": 50,
    }
