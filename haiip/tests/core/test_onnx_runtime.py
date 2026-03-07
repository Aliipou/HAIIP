"""Tests for haiip.core.onnx_runtime — ONNX Runtime inference engine.

Test categories:
    - Unit: init, fallback mode (no ONNX file), latency tracking
    - Integration: end-to-end with mock session
    - Crash/Edge: empty batch, zero features, missing file
    - Benchmark: latency stats, SLA tracking
    - Interface parity: same output keys as sklearn models
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from haiip.core.onnx_runtime import (
    LATENCY_SLA_MS,
    ONNXAnomalyDetector,
    ONNXMaintenancePredictor,
    _ORT_AVAILABLE,
)

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES = 5
SEQ_LEN = 10
FEATURE_NAMES = ["air_temp", "proc_temp", "rpm", "torque", "tool_wear"]
CLASS_NAMES = ["no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_ort_session(output_shape: tuple[int, ...] | None = None) -> MagicMock:
    """Build a mock ORT InferenceSession that returns zeros."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]

    def run_side_effect(output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = next(iter(inputs.values()))
        batch_size = batch.shape[0]
        # Autoencoder: return reconstructed input (same shape)
        return [batch.copy()]

    session.run.side_effect = run_side_effect
    return session


def _make_mock_maintenance_session() -> MagicMock:
    """Mock ORT session for maintenance predictor — returns logits + RUL."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]

    def run_side_effect(output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch_size = next(iter(inputs.values())).shape[0]
        logits = np.zeros((batch_size, len(CLASS_NAMES)), dtype=np.float32)
        logits[:, 0] = 5.0  # strongly predict no_failure
        rul = np.ones(batch_size, dtype=np.float32) * 100.0
        return [logits, rul]

    session.run.side_effect = run_side_effect
    return session


# ══════════════════════════════════════════════════════════════════════════════
# ONNXAnomalyDetector — Init & Fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXAnomalyDetectorInit:
    def test_defaults(self) -> None:
        d = ONNXAnomalyDetector("nonexistent.onnx")
        assert d.seq_len == 10
        assert d.n_features == 5
        assert d.latency_sla_ms == LATENCY_SLA_MS
        assert not d.is_ready

    def test_custom_params(self) -> None:
        d = ONNXAnomalyDetector(
            "nonexistent.onnx",
            seq_len=20,
            n_features=8,
            threshold=0.05,
            feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        )
        assert d.seq_len == 20
        assert d.n_features == 8
        assert d.threshold == 0.05

    def test_fallback_when_file_missing(self) -> None:
        d = ONNXAnomalyDetector("does_not_exist.onnx", n_features=N_FEATURES)
        assert not d.is_ready

    def test_scaler_defaults_to_identity(self) -> None:
        d = ONNXAnomalyDetector("nonexistent.onnx", n_features=N_FEATURES)
        np.testing.assert_array_equal(d.scaler_mean, np.zeros(N_FEATURES))
        np.testing.assert_array_equal(d.scaler_std, np.ones(N_FEATURES))


class TestONNXAnomalyDetectorFallback:
    """When session is None, model returns safe fallback results."""

    def test_predict_fallback(self) -> None:
        d = ONNXAnomalyDetector("missing.onnx", n_features=N_FEATURES)
        result = d.predict([1.0] * N_FEATURES)
        assert result["label"] == "normal"
        assert result["confidence"] == 0.5
        assert result["sla_ok"] is True

    def test_predict_batch_fallback(self) -> None:
        d = ONNXAnomalyDetector("missing.onnx", n_features=N_FEATURES)
        results = d.predict_batch(np.zeros((3, N_FEATURES)))
        assert len(results) == 3
        assert all(r["label"] == "normal" for r in results)

    def test_predict_batch_empty_fallback(self) -> None:
        d = ONNXAnomalyDetector("missing.onnx", n_features=N_FEATURES)
        results = d.predict_batch(np.zeros((0, N_FEATURES)))
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# ONNXAnomalyDetector — With Mock Session
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXAnomalyDetectorWithMock:
    @pytest.fixture()
    def detector(self) -> ONNXAnomalyDetector:
        d = ONNXAnomalyDetector(
            "fake.onnx",
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            threshold=0.01,
            feature_names=FEATURE_NAMES,
        )
        d._session = _make_mock_ort_session()
        return d

    def test_is_ready_with_session(self, detector: ONNXAnomalyDetector) -> None:
        assert detector.is_ready

    def test_predict_required_keys(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.predict([1.0] * N_FEATURES)
        required = {"label", "confidence", "anomaly_score", "reconstruction_error",
                    "threshold", "latency_ms", "sla_ok", "explanation"}
        assert required.issubset(result.keys())

    def test_predict_label_valid(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.predict([1.0] * N_FEATURES)
        assert result["label"] in ("normal", "anomaly")

    def test_predict_confidence_in_range(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.predict([1.0] * N_FEATURES)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_anomaly_score_in_range(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.predict([1.0] * N_FEATURES)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_predict_latency_ms_positive(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.predict([1.0] * N_FEATURES)
        assert result["latency_ms"] >= 0.0

    def test_predict_normal_for_normal_input(self, detector: ONNXAnomalyDetector) -> None:
        """Mock returns reconstructed = input → zero error → normal."""
        result = detector.predict([0.0] * N_FEATURES)
        assert result["label"] == "normal"
        assert result["reconstruction_error"] == pytest.approx(0.0, abs=1e-6)

    def test_predict_records_latency(self, detector: ONNXAnomalyDetector) -> None:
        detector.predict([1.0] * N_FEATURES)
        assert len(detector._latency_history) == 1

    def test_predict_batch_length(self, detector: ONNXAnomalyDetector) -> None:
        results = detector.predict_batch(np.zeros((5, N_FEATURES)))
        assert len(results) == 5

    def test_predict_batch_all_keys(self, detector: ONNXAnomalyDetector) -> None:
        results = detector.predict_batch(np.ones((3, N_FEATURES)))
        for r in results:
            assert "label" in r
            assert "latency_ms" in r

    def test_predict_batch_empty(self, detector: ONNXAnomalyDetector) -> None:
        results = detector.predict_batch(np.zeros((0, N_FEATURES)))
        assert results == []

    def test_sla_warning_logged(self, detector: ONNXAnomalyDetector) -> None:
        """SLA breach should be logged (we just verify sla_ok flag)."""
        # Inject a slow session
        def slow_run(output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            time.sleep(0.001)  # tiny delay — won't actually breach SLA in most envs
            batch = next(iter(inputs.values()))
            return [batch.copy()]

        detector._session.run.side_effect = slow_run
        detector.latency_sla_ms = 0.0  # force breach
        result = detector.predict([0.0] * N_FEATURES)
        assert result["sla_ok"] is False

    def test_sklearn_interface_parity(self, detector: ONNXAnomalyDetector) -> None:
        """Must have same base keys as sklearn AnomalyDetector."""
        sklearn_keys = {"label", "confidence", "anomaly_score", "explanation"}
        result = detector.predict([1.0] * N_FEATURES)
        assert sklearn_keys.issubset(result.keys())


# ══════════════════════════════════════════════════════════════════════════════
# ONNXAnomalyDetector — Benchmark
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXAnomalyDetectorBenchmark:
    @pytest.fixture()
    def detector(self) -> ONNXAnomalyDetector:
        d = ONNXAnomalyDetector("fake.onnx", n_features=N_FEATURES)
        d._session = _make_mock_ort_session()
        return d

    def test_benchmark_returns_required_keys(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.benchmark(n_runs=5)
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "sla_pass_rate", "n_runs"):
            assert key in result

    def test_benchmark_n_runs(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.benchmark(n_runs=10)
        assert result["n_runs"] == 10

    def test_benchmark_sla_pass_rate_in_range(self, detector: ONNXAnomalyDetector) -> None:
        result = detector.benchmark(n_runs=5)
        assert 0.0 <= result["sla_pass_rate"] <= 1.0

    def test_latency_stats_empty_initially(self) -> None:
        d = ONNXAnomalyDetector("missing.onnx", n_features=N_FEATURES)
        assert d.latency_stats == {}

    def test_latency_stats_after_predict(self, detector: ONNXAnomalyDetector) -> None:
        detector.predict([1.0] * N_FEATURES)
        stats = detector.latency_stats
        assert "n_calls" in stats
        assert "mean_ms" in stats
        assert stats["n_calls"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# ONNXAnomalyDetector — from_onnx constructor
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXAnomalyDetectorFromOnnx:
    def test_from_onnx_missing_file(self) -> None:
        d = ONNXAnomalyDetector.from_onnx("nonexistent.onnx")
        assert not d.is_ready

    def test_from_onnx_with_meta(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta.npz"
        np.savez(
            str(meta),
            seq_len=15,
            n_features=3,
            threshold=0.05,
            scaler_mean=np.zeros(3),
            scaler_std=np.ones(3),
        )
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")  # not a real model

        d = ONNXAnomalyDetector.from_onnx(onnx_file, meta_path=meta)
        assert d.seq_len == 15
        assert d.n_features == 3
        assert d.threshold == pytest.approx(0.05)


# ══════════════════════════════════════════════════════════════════════════════
# ONNXMaintenancePredictor — Init & Fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXMaintenancePredictorInit:
    def test_defaults(self) -> None:
        m = ONNXMaintenancePredictor("missing.onnx")
        assert m.seq_len == 10
        assert m.n_features == 5
        assert not m.is_ready

    def test_custom_class_names(self) -> None:
        m = ONNXMaintenancePredictor("missing.onnx", class_names=["ok", "fail"])
        assert m.class_names == ["ok", "fail"]

    def test_fallback_predict(self) -> None:
        m = ONNXMaintenancePredictor("missing.onnx", n_features=N_FEATURES)
        result = m.predict([1.0] * N_FEATURES)
        assert result["label"] == "no_failure"
        assert result["failure_probability"] == 0.0
        assert result["sla_ok"] is True

    def test_fallback_predict_batch(self) -> None:
        m = ONNXMaintenancePredictor("missing.onnx", n_features=N_FEATURES)
        results = m.predict_batch(np.zeros((3, N_FEATURES)))
        assert len(results) == 3

    def test_fallback_predict_batch_empty(self) -> None:
        m = ONNXMaintenancePredictor("missing.onnx", n_features=N_FEATURES)
        assert m.predict_batch(np.zeros((0, N_FEATURES))) == []


# ══════════════════════════════════════════════════════════════════════════════
# ONNXMaintenancePredictor — With Mock Session
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXMaintenancePredictorWithMock:
    @pytest.fixture()
    def predictor(self) -> ONNXMaintenancePredictor:
        m = ONNXMaintenancePredictor(
            "fake.onnx",
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            class_names=CLASS_NAMES,
            rul_mean=200.0,
            rul_std=50.0,
        )
        m._session = _make_mock_maintenance_session()
        return m

    def test_is_ready(self, predictor: ONNXMaintenancePredictor) -> None:
        assert predictor.is_ready

    def test_predict_required_keys(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.predict([1.0] * N_FEATURES)
        required = {"label", "confidence", "failure_probability", "rul_cycles",
                    "class_probabilities", "latency_ms", "sla_ok", "explanation"}
        assert required.issubset(result.keys())

    def test_predict_label_in_classes(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.predict([1.0] * N_FEATURES)
        assert result["label"] in CLASS_NAMES

    def test_predict_no_failure_dominant(self, predictor: ONNXMaintenancePredictor) -> None:
        """Mock returns logits strongly biased to no_failure."""
        result = predictor.predict([1.0] * N_FEATURES)
        assert result["label"] == "no_failure"

    def test_predict_failure_proba_in_range(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.predict([1.0] * N_FEATURES)
        assert 0.0 <= result["failure_probability"] <= 1.0

    def test_predict_rul_non_negative(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.predict([1.0] * N_FEATURES)
        if result["rul_cycles"] is not None:
            assert result["rul_cycles"] >= 0

    def test_predict_class_proba_sum_to_one(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.predict([1.0] * N_FEATURES)
        total = sum(result["class_probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_predict_batch_length(self, predictor: ONNXMaintenancePredictor) -> None:
        results = predictor.predict_batch(np.zeros((5, N_FEATURES)))
        assert len(results) == 5

    def test_predict_batch_all_keys(self, predictor: ONNXMaintenancePredictor) -> None:
        results = predictor.predict_batch(np.ones((3, N_FEATURES)))
        for r in results:
            assert "label" in r
            assert "rul_cycles" in r

    def test_predict_batch_empty(self, predictor: ONNXMaintenancePredictor) -> None:
        results = predictor.predict_batch(np.zeros((0, N_FEATURES)))
        assert results == []

    def test_predict_latency_tracked(self, predictor: ONNXMaintenancePredictor) -> None:
        predictor.predict([1.0] * N_FEATURES)
        assert len(predictor._latency_history) == 1

    def test_sklearn_interface_parity(self, predictor: ONNXMaintenancePredictor) -> None:
        """Must have same base keys as sklearn MaintenancePredictor."""
        sklearn_keys = {"label", "confidence", "failure_probability", "rul_cycles", "explanation"}
        result = predictor.predict([1.0] * N_FEATURES)
        assert sklearn_keys.issubset(result.keys())


# ══════════════════════════════════════════════════════════════════════════════
# ONNXMaintenancePredictor — Benchmark
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXMaintenancePredictorBenchmark:
    @pytest.fixture()
    def predictor(self) -> ONNXMaintenancePredictor:
        m = ONNXMaintenancePredictor(
            "fake.onnx", n_features=N_FEATURES, class_names=CLASS_NAMES
        )
        m._session = _make_mock_maintenance_session()
        return m

    def test_benchmark_keys(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.benchmark(n_runs=5)
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "sla_pass_rate", "n_runs"):
            assert key in result

    def test_benchmark_sla_rate_range(self, predictor: ONNXMaintenancePredictor) -> None:
        result = predictor.benchmark(n_runs=5)
        assert 0.0 <= result["sla_pass_rate"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ONNXMaintenancePredictor — from_onnx
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXMaintenancePredictorFromOnnx:
    def test_from_onnx_missing(self) -> None:
        m = ONNXMaintenancePredictor.from_onnx("nonexistent.onnx")
        assert not m.is_ready

    def test_from_onnx_with_meta(self, tmp_path: Path) -> None:
        import json

        meta = tmp_path / "meta.npz"
        np.savez(
            str(meta),
            seq_len=12,
            n_features=4,
            scaler_mean=np.zeros(4),
            scaler_std=np.ones(4),
            rul_mean=300.0,
            rul_std=75.0,
        )
        classes_file = tmp_path / "classes.json"
        with open(classes_file, "w") as f:
            json.dump({"class_names": ["ok", "fail"]}, f)
        onnx_file = tmp_path / "maint.onnx"
        onnx_file.write_bytes(b"fake")

        m = ONNXMaintenancePredictor.from_onnx(onnx_file, meta_path=meta, classes_path=classes_file)
        assert m.seq_len == 12
        assert m.n_features == 4
        assert m.class_names == ["ok", "fail"]
        assert m.rul_mean == pytest.approx(300.0)


# ══════════════════════════════════════════════════════════════════════════════
# Module constants
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleConstants:
    def test_latency_sla_ms(self) -> None:
        assert LATENCY_SLA_MS == 50.0

    def test_ort_available_is_bool(self) -> None:
        assert isinstance(_ORT_AVAILABLE, bool)
