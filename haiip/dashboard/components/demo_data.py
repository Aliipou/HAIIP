"""Synthetic demo data for offline/demo mode.

Generates realistic-looking data when the API is not reachable.
All values match the AI4I 2020 statistical distributions.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

_rng = np.random.default_rng(42)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def demo_machines() -> list[dict[str, Any]]:
    return [
        {"machine_id": "CNC-001", "location": "Jakobstad", "status": "normal",   "uptime_pct": 98.2},
        {"machine_id": "CNC-002", "location": "Jakobstad", "status": "warning",  "uptime_pct": 94.7},
        {"machine_id": "LATHE-001","location": "Sundsvall", "status": "normal",  "uptime_pct": 99.1},
        {"machine_id": "LATHE-002","location": "Sundsvall", "status": "anomaly", "uptime_pct": 87.3},
        {"machine_id": "PRESS-001","location": "Narvik",    "status": "normal",  "uptime_pct": 96.8},
        {"machine_id": "PRESS-002","location": "Narvik",    "status": "normal",  "uptime_pct": 97.4},
    ]


def demo_kpis() -> dict[str, Any]:
    return {
        "total_machines": 6,
        "anomaly_rate": 0.034,
        "active_alerts": 3,
        "avg_rul_cycles": 187,
        "model_accuracy": 0.943,
        "uptime_pct": 95.9,
        "predictions_today": 1842,
        "feedback_count": 47,
    }


def demo_sensor_stream(machine_id: str = "CNC-001", n: int = 60) -> list[dict[str, Any]]:
    """Generate n seconds of sensor readings."""
    base_time = _now() - timedelta(seconds=n)
    rng = np.random.default_rng(hash(machine_id) % 2**32)
    wear = 0.0
    readings = []

    for i in range(n):
        wear = min(wear + 0.5, 253.0)
        is_anomaly = i > n * 0.85 and machine_id == "LATHE-002"

        readings.append({
            "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
            "machine_id": machine_id,
            "air_temperature": round(float(rng.normal(303.0 if is_anomaly else 300.0, 2.5)), 2),
            "process_temperature": round(float(rng.normal(315.0 if is_anomaly else 310.0, 2.0)), 2),
            "rotational_speed": round(float(rng.normal(1300.0 if is_anomaly else 1538.0, 200.0)), 1),
            "torque": round(float(rng.normal(65.0 if is_anomaly else 40.0, 10.0)), 2),
            "tool_wear": round(wear, 1),
            "label": "anomaly" if is_anomaly else "normal",
            "confidence": round(float(rng.uniform(0.82, 0.97)), 3),
            "anomaly_score": round(float(rng.uniform(0.72, 0.95)), 3) if is_anomaly else round(float(rng.uniform(0.02, 0.18)), 3),
        })

    return readings


def demo_alerts() -> list[dict[str, Any]]:
    now = _now()
    return [
        {
            "id": "alert-001",
            "machine_id": "LATHE-002",
            "severity": "critical",
            "title": "Anomaly Detected — Torque Spike",
            "message": "Torque exceeded 3σ threshold. Anomaly score: 0.94",
            "is_acknowledged": False,
            "created_at": (now - timedelta(minutes=3)).isoformat(),
        },
        {
            "id": "alert-002",
            "machine_id": "CNC-002",
            "severity": "high",
            "title": "Tool Wear Approaching Limit",
            "message": "Tool wear at 231/253 min. Estimated RUL: 22 cycles.",
            "is_acknowledged": False,
            "created_at": (now - timedelta(minutes=18)).isoformat(),
        },
        {
            "id": "alert-003",
            "machine_id": "CNC-001",
            "severity": "medium",
            "title": "Heat Dissipation Warning",
            "message": "Process temperature delta exceeded 12K threshold.",
            "is_acknowledged": True,
            "created_at": (now - timedelta(hours=2)).isoformat(),
        },
        {
            "id": "alert-004",
            "machine_id": "PRESS-001",
            "severity": "low",
            "title": "Model Drift Detected",
            "message": "PSI drift detected on rotational_speed feature (PSI=0.23).",
            "is_acknowledged": True,
            "created_at": (now - timedelta(hours=5)).isoformat(),
        },
    ]


def demo_predictions(machine_id: str | None = None, n: int = 20) -> list[dict[str, Any]]:
    rng = np.random.default_rng(99)
    machines = ["CNC-001", "CNC-002", "LATHE-001", "LATHE-002", "PRESS-001"]
    labels = ["no_failure"] * 16 + ["TWF", "HDF", "PWF", "OSF"]
    preds = []
    base = _now() - timedelta(hours=1)

    for i in range(n):
        mach = machine_id or machines[i % len(machines)]
        lbl = labels[i % len(labels)]
        preds.append({
            "id": f"pred-{i:04d}",
            "machine_id": mach,
            "model_type": "predictive_maintenance",
            "prediction_label": lbl,
            "confidence": round(float(rng.uniform(0.78, 0.99)), 3),
            "anomaly_score": round(float(rng.uniform(0.05, 0.35 if lbl == "no_failure" else 0.85)), 3),
            "rul_cycles": int(rng.integers(20, 350)) if lbl == "no_failure" else int(rng.integers(5, 40)),
            "human_verified": i % 4 == 0,
            "created_at": (base + timedelta(minutes=i * 3)).isoformat(),
        })

    return preds


def demo_rul_per_machine() -> dict[str, int]:
    return {
        "CNC-001":  312,
        "CNC-002":  22,
        "LATHE-001": 287,
        "LATHE-002": 8,
        "PRESS-001": 198,
        "PRESS-002": 241,
    }


def demo_drift_results() -> list[dict[str, Any]]:
    return [
        {"feature": "air_temperature",      "psi": 0.04, "severity": "stable"},
        {"feature": "process_temperature",  "psi": 0.08, "severity": "stable"},
        {"feature": "rotational_speed",     "psi": 0.23, "severity": "drift"},
        {"feature": "torque",               "psi": 0.13, "severity": "monitoring"},
        {"feature": "tool_wear",            "psi": 0.06, "severity": "stable"},
    ]


def demo_audit_log() -> list[dict[str, Any]]:
    now = _now()
    return [
        {"id": "alog-001", "action": "prediction.created", "resource_type": "Prediction",
         "user_id": "usr-001", "tenant_id": "demo-sme", "details": '{"label":"no_failure","confidence":0.94}',
         "created_at": (now - timedelta(minutes=1)).isoformat()},
        {"id": "alog-002", "action": "feedback.submitted", "resource_type": "FeedbackLog",
         "user_id": "usr-002", "tenant_id": "demo-sme", "details": '{"was_correct":false,"corrected_label":"HDF"}',
         "created_at": (now - timedelta(minutes=5)).isoformat()},
        {"id": "alog-003", "action": "alert.acknowledged", "resource_type": "Alert",
         "user_id": "usr-001", "tenant_id": "demo-sme", "details": '{"alert_id":"alert-003"}',
         "created_at": (now - timedelta(hours=2)).isoformat()},
        {"id": "alog-004", "action": "model.retrained",    "resource_type": "ModelRegistry",
         "user_id": "system",  "tenant_id": "demo-sme", "details": '{"trigger":"feedback_accuracy_drop","accuracy":0.77}',
         "created_at": (now - timedelta(hours=4)).isoformat()},
        {"id": "alog-005", "action": "document.ingested",  "resource_type": "Document",
         "user_id": "usr-001", "tenant_id": "demo-sme", "details": '{"title":"CNC Maintenance Manual","chunks":14}',
         "created_at": (now - timedelta(days=1)).isoformat()},
    ]
