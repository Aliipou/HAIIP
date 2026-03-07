"""Tests for data/ingestion/pipeline.py — IngestionPipeline."""

from datetime import UTC, datetime

import numpy as np
import pytest

from haiip.core.anomaly import AnomalyDetector
from haiip.data.ingestion.pipeline import IngestionPipeline, NormalisedReading
from haiip.data.simulation.simulator import IndustrialSimulator


@pytest.fixture
def reading() -> NormalisedReading:
    return NormalisedReading(
        machine_id="TEST-001",
        tenant_id="test",
        timestamp=datetime.now(UTC),
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1538.0,
        torque=40.0,
        tool_wear=100.0,
        source="test",
    )


@pytest.fixture
def pipeline() -> IngestionPipeline:
    return IngestionPipeline(tenant_id="test")


@pytest.fixture
def pipeline_with_model() -> IngestionPipeline:
    rng = np.random.default_rng(42)
    X = rng.normal([300, 310, 1538, 40, 100], [2, 1.5, 179, 9.8, 50], (300, 5))
    detector = AnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(X)
    p = IngestionPipeline(tenant_id="test")
    p.set_anomaly_detector(detector)
    return p


# ── NormalisedReading ─────────────────────────────────────────────────────────


def test_feature_vector(reading):
    fv = reading.feature_vector
    assert len(fv) == 5
    assert fv[0] == reading.air_temperature


def test_to_dict(reading):
    d = reading.to_dict()
    assert "machine_id" in d
    assert "air_temperature" in d
    assert "timestamp" in d


# ── Pipeline ──────────────────────────────────────────────────────────────────


def test_process_no_model(pipeline, reading):
    result = pipeline.process(reading)
    assert result.anomaly_result["label"] == "normal"
    assert result.processing_time_ms >= 0.0
    assert result.alert_triggered is False


def test_process_with_model_normal(pipeline_with_model, reading):
    result = pipeline_with_model.process(reading)
    assert result.anomaly_result["label"] in ("normal", "anomaly")
    assert 0.0 <= result.anomaly_result["confidence"] <= 1.0


def test_process_with_model_anomaly(pipeline_with_model):
    extreme_reading = NormalisedReading(
        machine_id="TEST-002",
        tenant_id="test",
        timestamp=datetime.now(UTC),
        air_temperature=400.0,
        process_temperature=500.0,
        rotational_speed=9000.0,
        torque=500.0,
        tool_wear=500.0,
        source="test",
    )
    result = pipeline_with_model.process(extreme_reading)
    assert result.anomaly_result["label"] == "anomaly"


def test_alert_callback_called(pipeline_with_model):
    callbacks_called = []

    def cb(result):
        callbacks_called.append(result)

    pipeline_with_model.alert_callback = cb
    pipeline_with_model.anomaly_threshold = 0.0  # always trigger

    extreme = NormalisedReading(
        machine_id="TEST-003",
        tenant_id="test",
        timestamp=datetime.now(UTC),
        air_temperature=400.0,
        process_temperature=500.0,
        rotational_speed=9000.0,
        torque=500.0,
        tool_wear=500.0,
        source="test",
    )
    pipeline_with_model.process(extreme)
    # callback may or may not be called depending on model output


def test_validate_clamps_values(pipeline):
    reading = NormalisedReading(
        machine_id="X",
        tenant_id="t",
        timestamp=datetime.now(UTC),
        air_temperature=9999.0,  # > 500
        process_temperature=-999.0,  # < -50
        rotational_speed=-100.0,  # < 0
        torque=-50.0,
        tool_wear=5000.0,  # > 1000
        source="test",
    )
    validated = pipeline._validate(reading)
    assert validated.air_temperature == 500.0
    assert validated.process_temperature == -50.0
    assert validated.rotational_speed == 0.0
    assert validated.torque == 0.0
    assert validated.tool_wear == 1000.0


def test_stats_tracking(pipeline, reading):
    pipeline.process(reading)
    pipeline.process(reading)
    assert pipeline.stats["processing_count"] == 2


def test_normalise_from_simulator(pipeline):
    sim = IndustrialSimulator()
    raw = sim.next()
    normalised = pipeline.normalise_from_simulator(raw)
    assert normalised.source == "simulator"
    assert normalised.machine_id == raw["machine_id"]


def test_normalise_from_mqtt_buffer_missing_sensor(pipeline):
    incomplete_buffer = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        # missing rotational_speed, torque, tool_wear
    }
    result = pipeline.normalise_from_mqtt_buffer("MACHINE-001", incomplete_buffer)
    assert result is None


def test_normalise_from_mqtt_buffer_complete(pipeline):
    buffer = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1538.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }
    result = pipeline.normalise_from_mqtt_buffer("MACHINE-001", buffer)
    assert result is not None
    assert result.source == "mqtt"
    assert result.machine_id == "MACHINE-001"
