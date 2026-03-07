"""Custom Prometheus metrics for HAIIP ML operations.

Exposes machine-learning-specific signals that generic HTTP metrics miss:
  - Prediction latency (histogram, per model_type)
  - Anomaly score distribution (histogram, per tenant)
  - Drift detected (gauge, per feature)
  - Retraining events (counter, per trigger_reason)
  - Active model version (info gauge, per tenant + model_name)
  - ONNX SLA compliance (counter: pass / breach)

All metrics are registered once at import time (idempotent).

Usage:
    from haiip.api.ml_metrics import (
        record_prediction, record_drift, record_retrain, record_onnx_latency
    )
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

# ── Lazy Prometheus registration ──────────────────────────────────────────────
# prometheus_client raises if you register the same metric name twice.
# We guard with a module-level dict so this is safe even with hot-reload.

_metrics: dict[str, object] = {}


def _get_or_create(factory_fn: object, name: str, *args: object, **kwargs: object) -> object:
    if name not in _metrics:
        try:
            _metrics[name] = factory_fn(name, *args, **kwargs)  # type: ignore[call-arg, operator]
        except Exception as exc:
            logger.debug("Metric %s already registered or prometheus unavailable: %s", name, exc)
    return _metrics.get(name)


def _init_metrics() -> None:
    """Initialise all ML metrics — called once on first use."""
    try:
        from prometheus_client import Counter, Gauge, Histogram

        _get_or_create(
            Histogram,
            "haiip_prediction_latency_seconds",
            "End-to-end prediction latency (model inference + DB write)",
            ["model_type", "tenant_id"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )
        _get_or_create(
            Histogram,
            "haiip_anomaly_score",
            "Distribution of anomaly scores [0, 1]",
            ["tenant_id", "label"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        _get_or_create(
            Gauge,
            "haiip_drift_feature_psi",
            "Population Stability Index per feature (last drift check)",
            ["tenant_id", "feature_name"],
        )
        _get_or_create(
            Gauge,
            "haiip_drift_detected",
            "1 if drift detected in last check, 0 otherwise",
            ["tenant_id"],
        )
        _get_or_create(
            Counter,
            "haiip_retraining_total",
            "Total retraining cycles",
            ["tenant_id", "trigger_reason", "promoted"],
        )
        _get_or_create(
            Gauge,
            "haiip_active_model_version",
            "Numeric hash of active model version (for change detection)",
            ["tenant_id", "model_name"],
        )
        _get_or_create(
            Counter,
            "haiip_onnx_sla_total",
            "ONNX inference SLA outcomes",
            ["tenant_id", "model_type", "outcome"],  # outcome: pass | breach
        )
        _get_or_create(
            Histogram,
            "haiip_onnx_latency_ms",
            "ONNX Runtime inference latency in milliseconds",
            ["tenant_id", "model_type"],
            buckets=[1, 5, 10, 20, 30, 40, 50, 75, 100, 200],
        )
        _get_or_create(
            Counter,
            "haiip_feedback_total",
            "Operator feedback events recorded",
            ["tenant_id", "was_correct"],
        )
    except ImportError:
        logger.debug("prometheus_client not installed — ML metrics disabled")


_init_metrics()


# ── Public helpers ─────────────────────────────────────────────────────────────


def record_prediction(
    model_type: str,
    tenant_id: str,
    anomaly_score: float | None,
    label: str,
    latency_s: float,
) -> None:
    """Record a single prediction event."""
    try:
        lat = _metrics.get("haiip_prediction_latency_seconds")
        if lat:
            lat.labels(model_type=model_type, tenant_id=tenant_id).observe(latency_s)  # type: ignore[union-attr]

        score_hist = _metrics.get("haiip_anomaly_score")
        if score_hist and anomaly_score is not None:
            score_hist.labels(tenant_id=tenant_id, label=label).observe(  # type: ignore[union-attr]
                max(0.0, min(1.0, anomaly_score))
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_prediction metric failed: %s", exc)


def record_drift(
    tenant_id: str,
    drift_detected: bool,
    feature_psi: dict[str, float],
) -> None:
    """Record drift check results."""
    try:
        gauge = _metrics.get("haiip_drift_detected")
        if gauge:
            gauge.labels(tenant_id=tenant_id).set(1.0 if drift_detected else 0.0)  # type: ignore[union-attr]

        psi_gauge = _metrics.get("haiip_drift_feature_psi")
        if psi_gauge:
            for feature, psi in feature_psi.items():
                psi_gauge.labels(tenant_id=tenant_id, feature_name=feature).set(psi)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_drift metric failed: %s", exc)


def record_retrain(
    tenant_id: str,
    trigger_reason: str,
    promoted: bool,
) -> None:
    """Increment retraining counter."""
    try:
        ctr = _metrics.get("haiip_retraining_total")
        if ctr:
            ctr.labels(  # type: ignore[union-attr]
                tenant_id=tenant_id,
                trigger_reason=trigger_reason,
                promoted=str(promoted).lower(),
            ).inc()
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_retrain metric failed: %s", exc)


def record_model_version(tenant_id: str, model_name: str, version: str) -> None:
    """Update active model version gauge (numeric hash for change detection)."""
    try:
        gauge = _metrics.get("haiip_active_model_version")
        if gauge:
            # Use last 8 hex chars of sha256 as float — detects version changes in Grafana
            import hashlib
            h = int(hashlib.sha256(version.encode()).hexdigest()[-8:], 16)
            gauge.labels(tenant_id=tenant_id, model_name=model_name).set(float(h % 1_000_000))  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_model_version metric failed: %s", exc)


def record_onnx_latency(
    tenant_id: str,
    model_type: str,
    latency_ms: float,
    sla_ms: float = 50.0,
) -> None:
    """Record ONNX inference latency and SLA outcome."""
    try:
        hist = _metrics.get("haiip_onnx_latency_ms")
        if hist:
            hist.labels(tenant_id=tenant_id, model_type=model_type).observe(latency_ms)  # type: ignore[union-attr]

        ctr = _metrics.get("haiip_onnx_sla_total")
        if ctr:
            outcome = "pass" if latency_ms <= sla_ms else "breach"
            ctr.labels(tenant_id=tenant_id, model_type=model_type, outcome=outcome).inc()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_onnx_latency metric failed: %s", exc)


def record_feedback(tenant_id: str, was_correct: bool) -> None:
    """Record an operator feedback event."""
    try:
        ctr = _metrics.get("haiip_feedback_total")
        if ctr:
            ctr.labels(tenant_id=tenant_id, was_correct=str(was_correct).lower()).inc()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_feedback metric failed: %s", exc)


@contextmanager
def prediction_timer(model_type: str, tenant_id: str) -> Generator[None, None, None]:
    """Context manager that automatically records prediction latency.

    Usage:
        with prediction_timer("anomaly_detection", current_user.tenant_id):
            result = detector.predict(features)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        try:
            lat = _metrics.get("haiip_prediction_latency_seconds")
            if lat:
                lat.labels(model_type=model_type, tenant_id=tenant_id).observe(elapsed)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.debug("prediction_timer metric failed: %s", exc)
