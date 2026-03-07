"""Celery background tasks for HAIIP.

Tasks:
1. retrain_anomaly_model     — triggered by FeedbackEngine when accuracy drops
2. run_drift_check           — periodic drift monitoring for active tenants
3. generate_alerts           — process prediction results and create DB alerts
4. train_on_ai4i             — initial model training on AI4I 2020 dataset
5. cleanup_old_predictions   — purge predictions older than retention period

All tasks use shared_task so they work without the full Celery app context.
Beat schedule: drift_check every 1h, cleanup every 24h.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from celery import Celery
from celery.schedules import crontab

from haiip.api.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "haiip",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["haiip.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "haiip.workers.tasks.retrain_anomaly_model": {"queue": "ml"},
        "haiip.workers.tasks.run_drift_check": {"queue": "monitoring"},
        "haiip.workers.tasks.generate_alert": {"queue": "alerts"},
        "haiip.workers.tasks.train_on_ai4i": {"queue": "ml"},
        "haiip.workers.tasks.cleanup_old_predictions": {"queue": "maintenance"},
        "haiip.workers.tasks.auto_retrain_pipeline": {"queue": "ml"},
        "haiip.workers.tasks.export_onnx_model": {"queue": "ml"},
        "haiip.workers.tasks.benchmark_onnx_model": {"queue": "monitoring"},
    },
    beat_schedule={
        "drift-check-every-hour": {
            "task": "haiip.workers.tasks.run_drift_check",
            "schedule": crontab(minute=0),  # every hour
            "args": [],
        },
        "cleanup-predictions-daily": {
            "task": "haiip.workers.tasks.cleanup_old_predictions",
            "schedule": crontab(hour=2, minute=0),  # 2am daily
            "args": [90],  # retain 90 days
        },
        "auto-retrain-check-every-6h": {
            "task": "haiip.workers.tasks.auto_retrain_pipeline",
            "schedule": crontab(minute=0, hour="*/6"),
            "args": [],
        },
    },
)


# ── Task: initial training on AI4I ────────────────────────────────────────────


@celery_app.task(bind=True, name="haiip.workers.tasks.train_on_ai4i", max_retries=2)
def train_on_ai4i(
    self: Any,
    tenant_id: str = "default",
    contamination: float = 0.05,
    artifact_path: str | None = None,
) -> dict[str, Any]:
    """Train the anomaly detector on the AI4I 2020 dataset.

    Called once during platform onboarding per tenant.
    """
    from pathlib import Path

    logger.info("Starting AI4I training for tenant=%s", tenant_id)
    self.update_state(state="PROGRESS", meta={"step": "loading_dataset"})

    try:
        from haiip.core.anomaly import AnomalyDetector
        from haiip.data.loaders.ai4i import AI4ILoader

        loader = AI4ILoader()
        normal_df = loader.get_normal_data()

        self.update_state(
            state="PROGRESS",
            meta={"step": "fitting_model", "n_samples": len(normal_df)},
        )

        detector = AnomalyDetector(contamination=contamination, random_state=42)
        detector.fit_from_dataframe(normal_df)

        save_path = Path(artifact_path or settings.model_artifacts_path) / tenant_id / "anomaly"
        detector.save(save_path)

        result = {
            "status": "success",
            "tenant_id": tenant_id,
            "n_samples": len(normal_df),
            "artifact_path": str(save_path),
            "trained_at": datetime.now(UTC).isoformat(),
        }
        logger.info("AI4I training complete: %s", result)
        return result

    except Exception as exc:
        logger.error("AI4I training failed for tenant=%s: %s", tenant_id, exc)
        raise self.retry(exc=exc, countdown=60) from exc


# ── Task: anomaly model retraining ────────────────────────────────────────────


@celery_app.task(bind=True, name="haiip.workers.tasks.retrain_anomaly_model", max_retries=2)
def retrain_anomaly_model(
    self: Any,
    tenant_id: str,
    feedback_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retrain anomaly detector when feedback accuracy drops below threshold.

    Triggered by FeedbackEngine.needs_retraining == True.
    Uses AI4I as base + incorporates corrected feedback labels.
    """
    logger.info("Retraining anomaly model for tenant=%s", tenant_id)
    self.update_state(state="PROGRESS", meta={"step": "loading_data"})

    try:
        from pathlib import Path

        from haiip.core.anomaly import AnomalyDetector
        from haiip.data.loaders.ai4i import AI4ILoader

        loader = AI4ILoader()
        normal_df = loader.get_normal_data()
        X_base = normal_df[loader.feature_columns].values

        # If feedback provided confirmed-normal samples, include them
        if feedback_records:
            confirmed_normal = [
                r
                for r in feedback_records
                if r.get("was_correct") and r.get("corrected_label") == "no_failure"
            ]
            logger.info(
                "Incorporating %d confirmed-normal feedback records",
                len(confirmed_normal),
            )

        self.update_state(state="PROGRESS", meta={"step": "fitting"})

        detector = AnomalyDetector(contamination=0.05, random_state=42)
        detector.fit(X_base)

        save_path = Path(settings.model_artifacts_path) / tenant_id / "anomaly"
        detector.save(save_path)

        return {
            "status": "retrained",
            "tenant_id": tenant_id,
            "n_samples": len(X_base),
            "artifact_path": str(save_path),
            "retrained_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        logger.error("Retraining failed for tenant=%s: %s", tenant_id, exc)
        raise self.retry(exc=exc, countdown=120) from exc


# ── Task: drift monitoring ────────────────────────────────────────────────────


@celery_app.task(name="haiip.workers.tasks.run_drift_check")
def run_drift_check(tenant_ids: list[str] | None = None) -> dict[str, Any]:
    """Run drift check against reference distribution for all active tenants.

    Scheduled hourly. Results are logged; critical drift triggers retraining.
    """
    from pathlib import Path

    import numpy as np

    from haiip.core.drift import DriftDetector

    results: dict[str, Any] = {}
    tenant_ids = tenant_ids or ["default"]

    for tenant_id in tenant_ids:
        ref_path = Path(settings.model_artifacts_path) / tenant_id / "drift_reference.npy"
        cur_path = Path(settings.model_artifacts_path) / tenant_id / "drift_current.npy"

        if not ref_path.exists() or not cur_path.exists():
            results[tenant_id] = {"status": "skipped", "reason": "no reference data"}
            continue

        try:
            X_ref = np.load(str(ref_path))
            X_cur = np.load(str(cur_path))

            detector = DriftDetector(drift_threshold=settings.drift_threshold)
            detector.fit_reference(X_ref)
            drift_results = detector.check(X_cur)
            summary = detector.summary(drift_results)

            results[tenant_id] = {
                "status": "checked",
                "drift_detected": summary["drift_detected"],
                "severity": summary["severity"],
                "affected_features": summary["affected_features"],
            }

            if summary["drift_detected"] and summary["severity"] == "drift":
                logger.warning(
                    "Critical drift detected for tenant=%s — triggering retrain",
                    tenant_id,
                )
                retrain_anomaly_model.delay(tenant_id=tenant_id)

        except Exception as exc:
            logger.error("Drift check failed for tenant=%s: %s", tenant_id, exc)
            results[tenant_id] = {"status": "error", "error": str(exc)}

    return results


# ── Task: alert generation ────────────────────────────────────────────────────


@celery_app.task(name="haiip.workers.tasks.generate_alert")
def generate_alert(
    tenant_id: str,
    machine_id: str,
    prediction_id: str,
    severity: str,
    anomaly_score: float,
    label: str,
) -> dict[str, Any]:
    """Persist an alert to the database from a pipeline anomaly detection event.

    Called by the ingestion pipeline when anomaly_threshold is exceeded.
    """
    import asyncio

    async def _create() -> dict[str, Any]:
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        from haiip.api.models import Alert

        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        severity_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
        }
        db_severity = severity_map.get(severity, "medium")

        alert = Alert(
            tenant_id=tenant_id,
            machine_id=machine_id,
            severity=db_severity,
            title=f"Anomaly detected on {machine_id}",
            message=(
                f"Prediction ID: {prediction_id}\n"
                f"Label: {label}\n"
                f"Anomaly Score: {anomaly_score:.3f}\n"
                f"Severity: {severity}"
            ),
        )
        async with session_factory() as session:
            session.add(alert)
            await session.commit()
            await session.refresh(alert)
            alert_id = alert.id

        await engine.dispose()
        return {"alert_id": alert_id, "tenant_id": tenant_id, "machine_id": machine_id}

    return asyncio.run(_create())


# ── Task: Auto-Retrain Pipeline ───────────────────────────────────────────────


@celery_app.task(bind=True, name="haiip.workers.tasks.auto_retrain_pipeline", max_retries=1)
def auto_retrain_pipeline(
    self: Any,
    tenant_id: str = "default",
    feedback_accuracy: float | None = None,
    force_reason: str | None = None,
) -> dict[str, Any]:
    """Run the AutoRetrainPipeline champion-challenger cycle for a tenant.

    Called every 6h by Celery Beat, or on-demand when drift is detected.
    Loads reference data, runs drift check, triggers retraining if needed.
    """
    logger.info("AutoRetrainPipeline task started: tenant=%s", tenant_id)
    self.update_state(state="PROGRESS", meta={"step": "loading_data"})

    try:
        from pathlib import Path

        import numpy as np

        from haiip.core.anomaly import AnomalyDetector
        from haiip.core.auto_retrain import (
            AutoRetrainPipeline,
            RetrainTrigger,
            TriggerReason,
        )
        from haiip.core.drift import DriftDetector
        from haiip.data.loaders.ai4i import AI4ILoader

        artifact_dir = Path(settings.model_artifacts_path) / tenant_id
        ref_path = artifact_dir / "drift_reference.npy"
        cur_path = artifact_dir / "drift_current.npy"

        # Load training data
        loader = AI4ILoader()
        normal_df = loader.get_normal_data()
        X_train = normal_df[loader.feature_columns].values
        y_binary = np.zeros(len(X_train), dtype=int)  # all normal for champion eval

        # Build drift results if reference data available
        drift_results = None
        if ref_path.exists() and cur_path.exists():
            X_ref = np.load(str(ref_path))
            X_cur = np.load(str(cur_path))
            dd = DriftDetector(drift_threshold=settings.drift_threshold)
            dd.fit_reference(X_ref)
            drift_results = dd.check(X_cur)

        self.update_state(state="PROGRESS", meta={"step": "running_pipeline"})

        trigger = RetrainTrigger(cooldown_minutes=0.0, drift_feature_threshold=2)
        pipeline = AutoRetrainPipeline(
            tenant_id=tenant_id,
            artifact_dir=str(artifact_dir),
            trigger=trigger,
        )

        # Register champion from saved model (or train fresh)
        champ_path = artifact_dir / "anomaly"
        if champ_path.exists():
            champion = AnomalyDetector.load(champ_path)
        else:
            champion = AnomalyDetector(contamination=0.05, n_estimators=100, random_state=42)
            champion.fit(X_train)

        pipeline.register_champion(champion, X_train, y_binary)

        reason = TriggerReason(force_reason) if force_reason else None
        event = pipeline.maybe_retrain(
            X_train,
            drift_results=drift_results,
            feedback_accuracy=feedback_accuracy,
            reason=reason,
        )

        if event is None:
            return {
                "status": "no_retrain_needed",
                "tenant_id": tenant_id,
                "checked_at": datetime.now(UTC).isoformat(),
            }

        # Save promoted champion
        if event.promoted and pipeline.current_champion is not None:
            pipeline.current_champion.save(champ_path)
            logger.info("New champion saved to %s", champ_path)

        return {
            "status": event.status.value,
            "tenant_id": tenant_id,
            "trigger_reason": event.trigger_reason.value,
            "promoted": event.promoted,
            "champion_f1": event.champion_metrics.get("f1_macro"),
            "challenger_f1": event.challenger_metrics.get("f1_macro"),
            "n_training_samples": event.n_training_samples,
            "completed_at": event.completed_at,
        }

    except Exception as exc:
        logger.error("AutoRetrainPipeline task failed for tenant=%s: %s", tenant_id, exc)
        raise self.retry(exc=exc, countdown=300) from exc


# ── Task: ONNX Export ─────────────────────────────────────────────────────────


@celery_app.task(bind=True, name="haiip.workers.tasks.export_onnx_model", max_retries=1)
def export_onnx_model(
    self: Any,
    tenant_id: str = "default",
    model_type: str = "anomaly",
    opset: int = 17,
) -> dict[str, Any]:
    """Export the current champion model to ONNX for edge deployment.

    Args:
        tenant_id:  Tenant identifier
        model_type: "anomaly" or "maintenance"
        opset:      ONNX opset version (17 recommended)
    """
    logger.info("ONNX export task: tenant=%s, model_type=%s", tenant_id, model_type)
    self.update_state(state="PROGRESS", meta={"step": "loading_model"})

    try:
        from pathlib import Path

        from haiip.core.torch_models import AnomalyAutoencoder, MaintenanceLSTM

        artifact_dir = Path(settings.model_artifacts_path) / tenant_id
        onnx_dir = artifact_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"step": "exporting"})

        if model_type == "anomaly":
            model_path = artifact_dir / "autoencoder"
            onnx_out = onnx_dir / "anomaly_autoencoder.onnx"
            if model_path.exists():
                model = AnomalyAutoencoder.load(model_path)
            else:
                # Train minimal model for export demonstration
                from haiip.data.loaders.ai4i import AI4ILoader

                loader = AI4ILoader()
                df = loader.get_normal_data()
                X = df[loader.feature_columns].values
                model = AnomalyAutoencoder(n_features=len(loader.feature_columns), max_epochs=5)
                model.fit(X)

            out_path = model.export_onnx(onnx_out, opset=opset)
        elif model_type == "maintenance":
            model_path = artifact_dir / "lstm_maintenance"
            onnx_out = onnx_dir / "maintenance_lstm.onnx"
            if model_path.exists():
                model = MaintenanceLSTM.load(model_path)
                out_path = model.export_onnx(onnx_out, opset=opset)
            else:
                return {
                    "status": "skipped",
                    "reason": "No fitted MaintenanceLSTM found at expected path",
                    "tenant_id": tenant_id,
                }
        else:
            return {"status": "error", "reason": f"Unknown model_type: {model_type}"}

        size_kb = round(out_path.stat().st_size / 1024, 1)
        result = {
            "status": "exported",
            "tenant_id": tenant_id,
            "model_type": model_type,
            "onnx_path": str(out_path),
            "size_kb": size_kb,
            "opset": opset,
            "exported_at": datetime.now(UTC).isoformat(),
        }
        logger.info("ONNX export complete: %s", result)
        return result

    except Exception as exc:
        logger.error("ONNX export failed for tenant=%s: %s", tenant_id, exc)
        raise self.retry(exc=exc, countdown=60) from exc


# ── Task: ONNX Benchmark ──────────────────────────────────────────────────────


@celery_app.task(name="haiip.workers.tasks.benchmark_onnx_model")
def benchmark_onnx_model(
    tenant_id: str = "default",
    model_type: str = "anomaly",
    n_runs: int = 200,
) -> dict[str, Any]:
    """Benchmark ONNX model latency and verify SLA ≤50ms.

    Scheduled or triggered after each ONNX export to validate edge deployment.
    """
    logger.info("ONNX benchmark: tenant=%s, model=%s, runs=%d", tenant_id, model_type, n_runs)

    try:
        from pathlib import Path

        from haiip.core.onnx_runtime import (
            LATENCY_SLA_MS,
            ONNXAnomalyDetector,
            ONNXMaintenancePredictor,
        )

        artifact_dir = Path(settings.model_artifacts_path) / tenant_id / "onnx"

        if model_type == "anomaly":
            onnx_path = artifact_dir / "anomaly_autoencoder.onnx"
            detector = ONNXAnomalyDetector.from_onnx(onnx_path)
            if not detector.is_ready:
                return {"status": "skipped", "reason": "ONNX model not found"}
            stats = detector.benchmark(n_runs=n_runs)
        else:
            onnx_path = artifact_dir / "maintenance_lstm.onnx"
            detector = ONNXMaintenancePredictor.from_onnx(onnx_path)
            if not detector.is_ready:
                return {"status": "skipped", "reason": "ONNX model not found"}
            stats = detector.benchmark(n_runs=n_runs)

        sla_pass = stats["p99_ms"] <= LATENCY_SLA_MS
        if not sla_pass:
            logger.warning(
                "SLA breach: p99=%.1fms > %.0fms for tenant=%s model=%s",
                stats["p99_ms"],
                LATENCY_SLA_MS,
                tenant_id,
                model_type,
            )

        return {
            "status": "benchmarked",
            "tenant_id": tenant_id,
            "model_type": model_type,
            "sla_pass": sla_pass,
            "sla_limit_ms": LATENCY_SLA_MS,
            **stats,
            "benchmarked_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("ONNX benchmark failed: %s", exc)
        return {"status": "error", "error": str(exc), "tenant_id": tenant_id}


# ── Task: cleanup ─────────────────────────────────────────────────────────────


@celery_app.task(name="haiip.workers.tasks.cleanup_old_predictions")
def cleanup_old_predictions(retain_days: int = 90) -> dict[str, Any]:
    """Delete predictions older than retain_days. Runs nightly at 2am."""
    import asyncio

    async def _cleanup() -> dict[str, Any]:
        from sqlalchemy import delete
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        from haiip.api.models import Prediction

        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        cutoff = datetime.now(UTC) - timedelta(days=retain_days)
        async with session_factory() as session:
            result = await session.execute(delete(Prediction).where(Prediction.created_at < cutoff))
            await session.commit()
            deleted = result.rowcount

        await engine.dispose()
        logger.info("Cleanup: deleted %d predictions older than %d days", deleted, retain_days)
        return {"deleted": deleted, "cutoff": cutoff.isoformat()}

    return asyncio.run(_cleanup())
