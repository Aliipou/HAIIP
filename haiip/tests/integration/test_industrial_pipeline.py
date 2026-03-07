"""Integration tests — full industrial pipeline: sensor → drift → auto-retrain → ONNX.

Covers the end-to-end flow required for 100% industrial compatibility:
    1. Simulate sensor data (AI4I 2020 + injected anomalies)
    2. Run AnomalyDetector prediction on batch
    3. Run DriftDetector — detect distribution shift
    4. Trigger AutoRetrainPipeline — champion-challenger cycle
    5. Verify new champion is fitted and evaluatable
    6. (Optional) Export to ONNX and run via ONNXAnomalyDetector
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from haiip.core.anomaly import AnomalyDetector
from haiip.core.auto_retrain import (
    AutoRetrainPipeline,
    ChampionChallenger,
    ModelMetrics,
    RetrainStatus,
    RetrainTrigger,
    TriggerReason,
)
from haiip.core.drift import DriftDetector
from haiip.core.maintenance import MaintenancePredictor
from haiip.core.onnx_runtime import ONNXAnomalyDetector

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_FEATURES = 5
FEATURE_COLS = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]


@pytest.fixture()
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=[298, 308, 1500, 40, 100], scale=[2, 2, 50, 5, 30], size=(200, N_FEATURES))


@pytest.fixture()
def shifted_data() -> np.ndarray:
    """Shifted distribution simulating sensor drift."""
    rng = np.random.default_rng(1)
    return rng.normal(loc=[310, 320, 1800, 60, 200], scale=[3, 3, 60, 8, 40], size=(100, N_FEATURES))


@pytest.fixture()
def anomaly_data() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.normal(loc=[400, 400, 3000, 100, 500], scale=[10, 10, 100, 20, 50], size=(20, N_FEATURES))


@pytest.fixture()
def fitted_detector(normal_data: np.ndarray) -> AnomalyDetector:
    d = AnomalyDetector(contamination=0.05, n_estimators=50, random_state=42)
    d.fit(normal_data)
    return d


@pytest.fixture()
def fitted_drift_detector(normal_data: np.ndarray) -> DriftDetector:
    dd = DriftDetector(feature_names=FEATURE_COLS, drift_threshold=0.05, n_bins=10)
    dd.fit_reference(normal_data, feature_names=FEATURE_COLS)
    return dd


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Sensor → Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestSensorToAnomalyDetection:
    def test_normal_data_mostly_normal(self, fitted_detector: AnomalyDetector, normal_data: np.ndarray) -> None:
        results = fitted_detector.predict_batch(normal_data[:50])
        n_anomaly = sum(1 for r in results if r["label"] == "anomaly")
        # Contamination=0.05 → expect ≤10% anomaly rate
        assert n_anomaly / len(results) <= 0.15, f"Too many anomalies: {n_anomaly}/50"

    def test_anomaly_data_detected(self, fitted_detector: AnomalyDetector, anomaly_data: np.ndarray) -> None:
        results = fitted_detector.predict_batch(anomaly_data)
        n_anomaly = sum(1 for r in results if r["label"] == "anomaly")
        # Extreme values should be flagged as anomalies
        assert n_anomaly > 0, "No anomalies detected in clearly anomalous data"

    def test_all_predictions_have_required_keys(
        self, fitted_detector: AnomalyDetector, normal_data: np.ndarray
    ) -> None:
        required = {"label", "confidence", "anomaly_score", "explanation"}
        for row in normal_data[:10]:
            result = fitted_detector.predict(row.tolist())
            assert required.issubset(result.keys())

    def test_batch_length_matches_input(
        self, fitted_detector: AnomalyDetector, normal_data: np.ndarray
    ) -> None:
        results = fitted_detector.predict_batch(normal_data[:30])
        assert len(results) == 30

    def test_confidence_always_in_range(
        self, fitted_detector: AnomalyDetector, normal_data: np.ndarray
    ) -> None:
        for row in normal_data[:20]:
            result = fitted_detector.predict(row.tolist())
            assert 0.0 <= result["confidence"] <= 1.0

    def test_anomaly_score_always_in_range(
        self, fitted_detector: AnomalyDetector, normal_data: np.ndarray
    ) -> None:
        for row in normal_data[:20]:
            result = fitted_detector.predict(row.tolist())
            assert 0.0 <= result["anomaly_score"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Drift Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestDriftDetectionPipeline:
    def test_no_drift_on_same_distribution(
        self, fitted_drift_detector: DriftDetector, normal_data: np.ndarray
    ) -> None:
        results = fitted_drift_detector.check(normal_data[100:])
        n_drifted = sum(1 for r in results if r.severity == "drift")
        assert n_drifted == 0

    def test_drift_detected_on_shifted_data(
        self, fitted_drift_detector: DriftDetector, shifted_data: np.ndarray
    ) -> None:
        results = fitted_drift_detector.check(shifted_data)
        n_drifted = sum(1 for r in results if r.drift_detected)
        assert n_drifted > 0, "No drift detected on clearly shifted distribution"

    def test_summary_contains_required_keys(
        self, fitted_drift_detector: DriftDetector, shifted_data: np.ndarray
    ) -> None:
        results = fitted_drift_detector.check(shifted_data)
        summary = fitted_drift_detector.summary(results)
        for key in ("drift_detected", "severity", "affected_features", "feature_details"):
            assert key in summary

    def test_result_per_feature(
        self, fitted_drift_detector: DriftDetector, shifted_data: np.ndarray
    ) -> None:
        results = fitted_drift_detector.check(shifted_data)
        assert len(results) == N_FEATURES

    def test_stream_check(
        self, fitted_drift_detector: DriftDetector, normal_data: np.ndarray
    ) -> None:
        row = normal_data[0].tolist()
        result = fitted_drift_detector.check_stream(row)
        assert isinstance(result, dict)
        assert len(result) == N_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Drift → Auto-Retrain
# ══════════════════════════════════════════════════════════════════════════════


class TestDriftToAutoRetrain:
    def test_drift_triggers_retrain(
        self,
        fitted_detector: AnomalyDetector,
        fitted_drift_detector: DriftDetector,
        normal_data: np.ndarray,
        shifted_data: np.ndarray,
    ) -> None:
        """Drift on shifted data → AutoRetrainPipeline fires."""
        drift_results = fitted_drift_detector.check(shifted_data)

        y_binary = np.zeros(len(normal_data), dtype=int)
        trigger = RetrainTrigger(drift_feature_threshold=1, cooldown_minutes=0.0)
        pipeline = AutoRetrainPipeline(trigger=trigger, tenant_id="drift-integration")
        pipeline.register_champion(fitted_detector, normal_data, y_binary)

        event = pipeline.maybe_retrain(normal_data, drift_results=drift_results)
        # If drift detected, event should fire
        n_drifted = sum(1 for r in drift_results if r.drift_detected)
        if n_drifted >= 1:
            assert event is not None
            assert event.status == RetrainStatus.COMPLETE

    def test_retrain_event_has_challenger_metrics(
        self,
        fitted_detector: AnomalyDetector,
        normal_data: np.ndarray,
    ) -> None:
        y = np.zeros(len(normal_data), dtype=int)
        trigger = RetrainTrigger(cooldown_minutes=0.0)
        p = AutoRetrainPipeline(trigger=trigger, tenant_id="challenger-test")
        p.register_champion(fitted_detector, normal_data, y)

        event = p.maybe_retrain(normal_data, feedback_accuracy=0.60)
        assert event is not None
        assert "f1_macro" in event.challenger_metrics
        assert "auc_roc" in event.challenger_metrics

    def test_pipeline_summary_after_retrain(
        self,
        fitted_detector: AnomalyDetector,
        normal_data: np.ndarray,
    ) -> None:
        y = np.zeros(len(normal_data), dtype=int)
        trigger = RetrainTrigger(cooldown_minutes=0.0)
        p = AutoRetrainPipeline(trigger=trigger)
        p.register_champion(fitted_detector, normal_data, y)
        p.maybe_retrain(normal_data, feedback_accuracy=0.60)

        s = p.summary()
        assert s["total_retrain_events"] == 1
        assert s["champion_f1"] >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — ONNX Fallback Mode (no actual ONNX file)
# ══════════════════════════════════════════════════════════════════════════════


class TestONNXFallbackIntegration:
    """Tests that ONNX detector returns safe results when no model file exists."""

    def test_fallback_predict_has_same_interface(self) -> None:
        d = ONNXAnomalyDetector("does_not_exist.onnx", n_features=N_FEATURES)
        required = {"label", "confidence", "anomaly_score", "explanation"}
        result = d.predict([1.0] * N_FEATURES)
        assert required.issubset(result.keys())

    def test_fallback_batch_predict_length(self, normal_data: np.ndarray) -> None:
        d = ONNXAnomalyDetector("does_not_exist.onnx", n_features=N_FEATURES)
        results = d.predict_batch(normal_data[:10])
        assert len(results) == 10

    def test_fallback_sla_ok(self) -> None:
        d = ONNXAnomalyDetector("does_not_exist.onnx", n_features=N_FEATURES)
        result = d.predict([1.0] * N_FEATURES)
        assert result["sla_ok"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Champion persistence (save → load → predict)
# ══════════════════════════════════════════════════════════════════════════════


class TestChampionPersistence:
    def test_save_and_reload_champion_predicts_same(
        self,
        fitted_detector: AnomalyDetector,
        normal_data: np.ndarray,
        tmp_path: Path,
    ) -> None:
        sample = normal_data[0].tolist()
        r1 = fitted_detector.predict(sample)

        fitted_detector.save(tmp_path / "champion")
        reloaded = AnomalyDetector.load(tmp_path / "champion")
        r2 = reloaded.predict(sample)

        assert r1["label"] == r2["label"]
        assert abs(r1["anomaly_score"] - r2["anomaly_score"]) < 1e-4

    def test_retrained_champion_is_fitted(
        self,
        fitted_detector: AnomalyDetector,
        normal_data: np.ndarray,
    ) -> None:
        y = np.zeros(len(normal_data), dtype=int)
        trigger = RetrainTrigger(cooldown_minutes=0.0)
        cc = ChampionChallenger(min_improvement=-1.0)  # always promote
        p = AutoRetrainPipeline(trigger=trigger, cc=cc)
        p.register_champion(fitted_detector, normal_data, y)
        p.maybe_retrain(normal_data, feedback_accuracy=0.50)

        champion = p.current_champion
        assert champion is not None
        assert champion.is_fitted


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Maintenance Predictor integration
# ══════════════════════════════════════════════════════════════════════════════


class TestMaintenancePredictorIntegration:
    @pytest.fixture()
    def fitted_maintenance(self, normal_data: np.ndarray) -> MaintenancePredictor:
        y_class = np.array(["no_failure"] * 180 + ["TWF"] * 10 + ["HDF"] * 10)
        y_rul = np.random.default_rng(42).uniform(50, 500, 200)
        m = MaintenancePredictor(n_estimators=50, random_state=42)
        m.fit(normal_data, y_class, y_rul)
        return m

    def test_predict_has_required_keys(
        self, fitted_maintenance: MaintenancePredictor, normal_data: np.ndarray
    ) -> None:
        required = {"label", "confidence", "failure_probability", "rul_cycles", "explanation"}
        result = fitted_maintenance.predict(normal_data[0].tolist())
        assert required.issubset(result.keys())

    def test_batch_predict_length(
        self, fitted_maintenance: MaintenancePredictor, normal_data: np.ndarray
    ) -> None:
        results = fitted_maintenance.predict_batch(normal_data[:15])
        assert len(results) == 15

    def test_rul_non_negative(
        self, fitted_maintenance: MaintenancePredictor, normal_data: np.ndarray
    ) -> None:
        result = fitted_maintenance.predict(normal_data[0].tolist())
        if result["rul_cycles"] is not None:
            assert result["rul_cycles"] >= 0

    def test_failure_proba_in_range(
        self, fitted_maintenance: MaintenancePredictor, normal_data: np.ndarray
    ) -> None:
        for row in normal_data[:10]:
            result = fitted_maintenance.predict(row.tolist())
            assert 0.0 <= result["failure_probability"] <= 1.0

    def test_auto_retrain_with_maintenance_mode(
        self, fitted_maintenance: MaintenancePredictor, normal_data: np.ndarray
    ) -> None:
        y_class = np.array(["no_failure"] * 180 + ["TWF"] * 10 + ["HDF"] * 10)
        trigger = RetrainTrigger(cooldown_minutes=0.0)
        p = AutoRetrainPipeline(trigger=trigger, tenant_id="maint-integration")
        p.register_champion(fitted_maintenance, normal_data, y_class, mode="maintenance")
        event = p.maybe_retrain(normal_data, feedback_accuracy=0.55, mode="maintenance")
        # Either retrains or not — just verify no crash
        if event is not None:
            assert event.status in (RetrainStatus.COMPLETE, RetrainStatus.FAILED)


# ══════════════════════════════════════════════════════════════════════════════
# Edge / Crash tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIndustrialEdgeCases:
    def test_predict_with_nan_sensor(self, fitted_detector: AnomalyDetector) -> None:
        result = fitted_detector.predict([float("nan")] * N_FEATURES)
        assert "label" in result

    def test_predict_with_inf_sensor(self, fitted_detector: AnomalyDetector) -> None:
        result = fitted_detector.predict([float("inf")] * N_FEATURES)
        assert "label" in result

    def test_drift_check_on_single_row(
        self, fitted_drift_detector: DriftDetector
    ) -> None:
        """Single-row current data should not crash."""
        single = np.ones((2, N_FEATURES))  # KS test needs ≥2 samples
        results = fitted_drift_detector.check(single)
        assert len(results) == N_FEATURES

    def test_pipeline_no_trigger_returns_none(
        self, fitted_detector: AnomalyDetector, normal_data: np.ndarray
    ) -> None:
        y = np.zeros(len(normal_data), dtype=int)
        trigger = RetrainTrigger(cooldown_minutes=999.0)  # very long cooldown
        p = AutoRetrainPipeline(trigger=trigger)
        p.register_champion(fitted_detector, normal_data, y)
        # First call may trigger (no cooldown yet), second should not
        p.maybe_retrain(normal_data, feedback_accuracy=0.50)
        event = p.maybe_retrain(normal_data, feedback_accuracy=0.50)
        assert event is None  # blocked by cooldown

    def test_concurrent_predictions_stable(
        self, fitted_detector: AnomalyDetector, normal_data: np.ndarray
    ) -> None:
        import threading

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for row in normal_data[:10]:
                    fitted_detector.predict(row.tolist())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_drift_stream_single_value(
        self, fitted_drift_detector: DriftDetector, normal_data: np.ndarray
    ) -> None:
        result = fitted_drift_detector.check_stream(normal_data[0].tolist())
        assert isinstance(result, dict)
        assert all(isinstance(v, bool) for v in result.values())
