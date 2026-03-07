"""Tests for dashboard demo data generators."""

from haiip.dashboard.components.demo_data import (
    demo_alerts,
    demo_audit_log,
    demo_drift_results,
    demo_kpis,
    demo_machines,
    demo_predictions,
    demo_rul_per_machine,
    demo_sensor_stream,
)


def test_demo_machines_returns_list():
    machines = demo_machines()
    assert isinstance(machines, list)
    assert len(machines) > 0
    for m in machines:
        assert "machine_id" in m
        assert "status" in m
        assert m["status"] in ("normal", "warning", "anomaly")


def test_demo_kpis_keys():
    kpis = demo_kpis()
    required = ["total_machines", "anomaly_rate", "active_alerts", "predictions_today"]
    for key in required:
        assert key in kpis


def test_demo_kpis_anomaly_rate_range():
    kpis = demo_kpis()
    assert 0.0 <= kpis["anomaly_rate"] <= 1.0


def test_demo_sensor_stream_length():
    stream = demo_sensor_stream("CNC-001", n=60)
    assert len(stream) == 60


def test_demo_sensor_stream_required_fields():
    stream = demo_sensor_stream("CNC-001", n=5)
    required = [
        "timestamp",
        "machine_id",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "label",
        "confidence",
    ]
    for reading in stream:
        for key in required:
            assert key in reading, f"Missing key: {key}"


def test_demo_sensor_stream_label_valid():
    stream = demo_sensor_stream("CNC-001", n=10)
    for r in stream:
        assert r["label"] in ("normal", "anomaly")


def test_demo_sensor_stream_different_machines():
    s1 = demo_sensor_stream("CNC-001", n=5)
    s2 = demo_sensor_stream("LATHE-002", n=5)
    # Different machines should produce different machine_ids
    assert all(r["machine_id"] == "CNC-001" for r in s1)
    assert all(r["machine_id"] == "LATHE-002" for r in s2)


def test_demo_alerts_structure():
    alerts = demo_alerts()
    assert len(alerts) > 0
    for alert in alerts:
        assert "id" in alert
        assert "severity" in alert
        assert alert["severity"] in ("critical", "high", "medium", "low")
        assert "is_acknowledged" in alert
        assert isinstance(alert["is_acknowledged"], bool)


def test_demo_predictions_structure():
    preds = demo_predictions(n=10)
    assert len(preds) == 10
    for pred in preds:
        assert "id" in pred
        assert "machine_id" in pred
        assert "confidence" in pred
        assert 0.0 <= pred["confidence"] <= 1.0


def test_demo_rul_per_machine():
    rul = demo_rul_per_machine()
    assert isinstance(rul, dict)
    for machine_id, cycles in rul.items():
        assert isinstance(machine_id, str)
        assert cycles >= 0


def test_demo_drift_results():
    drift = demo_drift_results()
    assert len(drift) > 0
    for d in drift:
        assert "feature" in d
        assert "psi" in d
        assert "severity" in d
        assert d["severity"] in ("stable", "monitoring", "drift")
        assert 0.0 <= d["psi"] <= 1.0


def test_demo_audit_log():
    logs = demo_audit_log()
    assert len(logs) > 0
    for log in logs:
        assert "id" in log
        assert "action" in log
        assert "created_at" in log
        assert "tenant_id" in log


def test_demo_predictions_filter_by_machine():
    preds = demo_predictions(machine_id="CNC-001", n=5)
    assert all(p["machine_id"] == "CNC-001" for p in preds)
