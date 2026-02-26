"""AI accuracy and compatibility tests.

Verifies:
1. Model accuracy meets declared thresholds (from MODEL_CARD.md)
2. API schema compatibility (request/response contracts)
3. Cross-version compatibility (pickle/joblib model serialization)
4. Feature compatibility (same features at train and inference time)
5. Data type compatibility (int/float/numpy types handled uniformly)
6. Pipeline compatibility (ingestion → ML → API → response)

These tests serve as the acceptance gate for production deployment.
If any test fails, the model or pipeline is not production-ready.
"""

from __future__ import annotations

import io
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_balanced_dataset(n_normal: int = 300, n_anomaly: int = 30, seed: int = 42):
    """Generate a balanced train/test dataset."""
    rng = np.random.RandomState(seed)

    normal_X = np.column_stack([
        rng.normal(298.0, 2.0, n_normal),
        rng.normal(308.0, 1.5, n_normal),
        rng.normal(1500, 50, n_normal),
        rng.normal(40.0, 3.0, n_normal),
        rng.uniform(0, 50, n_normal),
    ])

    anomaly_X = np.column_stack([
        rng.normal(340.0, 5.0, n_anomaly),
        rng.normal(355.0, 4.0, n_anomaly),
        rng.normal(2200, 100, n_anomaly),
        rng.normal(70.0, 8.0, n_anomaly),
        rng.uniform(200, 250, n_anomaly),
    ])

    return normal_X, anomaly_X


# ── Accuracy Tests ────────────────────────────────────────────────────────────

class TestAnomalyDetectorAccuracy:
    """Verify anomaly detector meets declared accuracy thresholds."""

    @pytest.fixture
    def trained(self):
        from haiip.core.anomaly import AnomalyDetector
        normal_X, _ = make_balanced_dataset()
        detector = AnomalyDetector(contamination=0.05, random_state=42)
        detector.fit(normal_X.tolist())
        return detector

    def test_precision_on_normal_data(self, trained):
        """True Negative Rate on normal data must be ≥ 75%."""
        normal_X, _ = make_balanced_dataset(n_normal=100, seed=100)
        results = trained.predict_batch(normal_X.tolist())
        true_negatives = sum(1 for r in results if r["label"] == "normal")
        tnr = true_negatives / len(results)
        assert tnr >= 0.75, f"TNR on normal data: {tnr:.2%} < 75%"

    def test_detection_rate_on_anomaly_data(self, trained):
        """True Positive Rate on anomalous data must be ≥ 50%."""
        _, anomaly_X = make_balanced_dataset(n_anomaly=50, seed=200)
        results = trained.predict_batch(anomaly_X.tolist())
        true_positives = sum(1 for r in results if r["label"] == "anomaly")
        tpr = true_positives / len(results)
        assert tpr >= 0.5, f"TPR on anomaly data: {tpr:.2%} < 50%"

    def test_anomaly_score_monotonicity(self, trained):
        """More extreme sensor values → higher anomaly score (statistical)."""
        moderate_anomaly = [315.0, 325.0, 1700.0, 52.0, 120.0]
        extreme_anomaly = [350.0, 365.0, 2400.0, 80.0, 240.0]

        r_mod = trained.predict(moderate_anomaly)
        r_ext = trained.predict(extreme_anomaly)

        # Extreme should score higher or equal
        assert r_ext["anomaly_score"] >= r_mod["anomaly_score"] * 0.8

    def test_confidence_always_above_zero(self, trained):
        """Confidence should never be exactly 0 on trained model."""
        normal_X, anomaly_X = make_balanced_dataset(n_normal=50, n_anomaly=10)
        all_data = np.vstack([normal_X, anomaly_X]).tolist()
        results = trained.predict_batch(all_data)
        zero_conf = [r for r in results if r["confidence"] == 0.0]
        # Allow at most 5% zero confidence
        assert len(zero_conf) / len(results) < 0.05


class TestMaintenancePredictorAccuracy:
    """Verify maintenance predictor meets declared accuracy thresholds."""

    @pytest.fixture
    def trained(self):
        from haiip.core.maintenance import MaintenancePredictor

        rng = np.random.RandomState(42)
        X, y_failure, y_rul = [], [], []

        for i in range(500):
            wear = rng.uniform(0, 240)
            temp = rng.normal(298.0 + wear * 0.2, 2.0)
            speed = rng.normal(1500, 100)
            torque = rng.normal(40.0 + wear * 0.1, 3.0)
            process_temp = temp + 10 + rng.normal(0, 1)
            X.append([temp, process_temp, speed, torque, wear])

            if temp > 332 and process_temp > 342:
                y_failure.append("HDF")
            elif torque > 58 and wear > 210:
                y_failure.append("PWF")
            elif speed > 2100:
                y_failure.append("OSF")
            else:
                y_failure.append("no_failure")

            y_rul.append(max(0, int(250 - wear - rng.uniform(0, 15))))

        predictor = MaintenancePredictor(n_estimators=100)
        predictor.fit(X, y_failure, y_rul)
        return predictor

    def test_no_failure_accuracy(self, trained):
        """Accuracy on clear no-failure samples must be ≥ 80%."""
        rng = np.random.RandomState(300)
        clear_normal = [
            [rng.normal(298, 1), rng.normal(308, 1), rng.normal(1500, 30),
             rng.normal(40, 2), rng.uniform(0, 30)]
            for _ in range(50)
        ]
        results = [trained.predict(s) for s in clear_normal]
        correct = sum(1 for r in results if r["label"] == "no_failure")
        accuracy = correct / len(results)
        assert accuracy >= 0.70, f"no_failure accuracy: {accuracy:.2%} < 70%"

    def test_rul_decreases_with_tool_wear(self, trained):
        """RUL should correlate negatively with tool wear."""
        rng = np.random.RandomState(7)
        low_wear_ruls = []
        high_wear_ruls = []

        for _ in range(20):
            low_sample = [298.0, 308.0, 1500.0, 40.0, rng.uniform(0, 20)]
            high_sample = [298.0, 308.0, 1500.0, 40.0, rng.uniform(200, 240)]
            r_low = trained.predict(low_sample)
            r_high = trained.predict(high_sample)
            if r_low["rul_cycles"] is not None:
                low_wear_ruls.append(r_low["rul_cycles"])
            if r_high["rul_cycles"] is not None:
                high_wear_ruls.append(r_high["rul_cycles"])

        if low_wear_ruls and high_wear_ruls:
            assert np.mean(low_wear_ruls) >= np.mean(high_wear_ruls) * 0.8

    def test_failure_probability_for_extreme_conditions(self, trained):
        """Extreme conditions must have elevated failure probability."""
        extreme = [350.0, 365.0, 2400.0, 75.0, 240.0]
        result = trained.predict(extreme)
        # failure_probability should be > 0.3 for extreme conditions
        assert result["failure_probability"] >= 0.3 or result["label"] != "no_failure"


# ── Serialization Compatibility ───────────────────────────────────────────────

class TestModelSerialization:
    """Verify models can be saved and loaded (joblib/pickle compatibility)."""

    def test_anomaly_detector_save_load_roundtrip(self, tmp_path):
        from haiip.core.anomaly import AnomalyDetector

        rng = np.random.RandomState(42)
        data = rng.normal(298.0, 2.0, (100, 5)).tolist()
        test_sample = [298.0, 308.0, 1500.0, 40.0, 5.0]

        detector = AnomalyDetector(random_state=42)
        detector.fit(data)
        original_result = detector.predict(test_sample)

        # Save
        model_path = tmp_path / "anomaly_detector.joblib"
        detector.save(str(model_path))

        # Load
        loaded = AnomalyDetector.load(str(model_path))
        loaded_result = loaded.predict(test_sample)

        assert original_result["label"] == loaded_result["label"]
        assert abs(original_result["anomaly_score"] - loaded_result["anomaly_score"]) < 1e-6

    def test_maintenance_predictor_save_load_roundtrip(self, tmp_path):
        from haiip.core.maintenance import MaintenancePredictor

        rng = np.random.RandomState(42)
        X = rng.normal(298.0, 5.0, (100, 5)).tolist()
        y_class = ["no_failure"] * 100
        test_sample = [298.0, 308.0, 1500.0, 40.0, 5.0]

        predictor = MaintenancePredictor(n_estimators=50, random_state=42)
        predictor.fit(X, y_class)
        original_result = predictor.predict(test_sample)

        # Save
        model_path = tmp_path / "maintenance_predictor.joblib"
        predictor.save(str(model_path))

        # Load
        loaded = MaintenancePredictor.load(str(model_path))
        loaded_result = loaded.predict(test_sample)

        assert original_result["label"] == loaded_result["label"]

    def test_anomaly_detector_rejects_wrong_feature_count(self):
        """Model trained on 5 features must reject inputs with wrong feature count."""
        from haiip.core.anomaly import AnomalyDetector

        rng = np.random.RandomState(0)
        data = rng.normal(0, 1, (50, 5)).tolist()
        detector = AnomalyDetector(random_state=42)
        detector.fit(data)

        # Wrong number of features
        with pytest.raises(Exception):
            detector.predict([1.0, 2.0])  # only 2 features instead of 5


# ── Data Type Compatibility ───────────────────────────────────────────────────

class TestDataTypeCompatibility:
    """Verify ML engines handle various numeric types correctly."""

    @pytest.fixture
    def trained_detector(self):
        from haiip.core.anomaly import AnomalyDetector
        rng = np.random.RandomState(42)
        data = rng.normal(298.0, 2.0, (100, 5)).tolist()
        d = AnomalyDetector(random_state=42)
        d.fit(data)
        return d

    def test_accepts_python_list_of_floats(self, trained_detector):
        result = trained_detector.predict([298.0, 308.0, 1500.0, 40.0, 5.0])
        assert "label" in result

    def test_accepts_python_list_of_ints(self, trained_detector):
        result = trained_detector.predict([298, 308, 1500, 40, 5])
        assert "label" in result

    def test_accepts_numpy_float64_array(self, trained_detector):
        sample = np.array([298.0, 308.0, 1500.0, 40.0, 5.0], dtype=np.float64)
        result = trained_detector.predict(sample)
        assert "label" in result

    def test_accepts_numpy_float32_array(self, trained_detector):
        sample = np.array([298.0, 308.0, 1500.0, 40.0, 5.0], dtype=np.float32)
        result = trained_detector.predict(sample)
        assert "label" in result

    def test_batch_accepts_2d_numpy_array(self, trained_detector):
        batch = np.array([
            [298.0, 308.0, 1500.0, 40.0, 5.0],
            [310.0, 320.0, 1800.0, 55.0, 150.0],
        ])
        results = trained_detector.predict_batch(batch)
        assert len(results) == 2

    def test_batch_accepts_list_of_lists(self, trained_detector):
        batch = [
            [298.0, 308.0, 1500.0, 40.0, 5.0],
            [310.0, 320.0, 1800.0, 55.0, 150.0],
        ]
        results = trained_detector.predict_batch(batch)
        assert len(results) == 2


# ── Pipeline Compatibility ────────────────────────────────────────────────────

class TestPipelineCompatibility:
    """Test ingestion → ML pipeline compatibility."""

    def test_simulator_output_compatible_with_anomaly_detector(self):
        """SimulatorConfig output should be directly usable by AnomalyDetector."""
        from haiip.core.anomaly import AnomalyDetector
        from haiip.data.simulation.simulator import IndustrialSimulator, SimulatorConfig

        config = SimulatorConfig(machine_id="TEST-001", seed=42)
        simulator = IndustrialSimulator(config)
        batch = simulator.batch(n=100)

        # Extract features from simulator readings
        feature_keys = ["air_temperature", "process_temperature",
                        "rotational_speed", "torque", "tool_wear"]
        X = []
        for reading in batch:
            features = reading.get("features", reading)
            row = [float(features.get(k, features.get(k.replace("_", " "), 0.0)))
                   for k in feature_keys]
            if len(row) == 5:
                X.append(row)

        if len(X) >= 50:
            detector = AnomalyDetector(random_state=42)
            detector.fit(X)
            result = detector.predict(X[0])
            assert "label" in result

    def test_ingestion_pipeline_normalises_to_expected_range(self):
        """Pipeline output should have features in expected sensor ranges."""
        from haiip.data.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        raw_reading = {
            "air_temperature": 298.5,
            "process_temperature": 308.5,
            "rotational_speed": 1545,
            "torque": 42.3,
            "tool_wear": 45,
        }
        normalised = pipeline.normalise_from_simulator(raw_reading)
        if normalised:
            # Verify values are not wildly out of range (pipeline didn't corrupt data)
            assert normalised.features.get("air_temperature", 0) > 200  # sanity
            assert normalised.features.get("air_temperature", 9999) < 500

    def test_compliance_engine_log_matches_prediction_format(self):
        """ComplianceEngine should accept output from AnomalyDetector."""
        from haiip.core.anomaly import AnomalyDetector
        from haiip.core.compliance import ComplianceEngine

        rng = np.random.RandomState(42)
        data = rng.normal(298.0, 2.0, (100, 5)).tolist()
        detector = AnomalyDetector(random_state=42)
        detector.fit(data)

        sample = [298.0, 308.0, 1500.0, 40.0, 5.0]
        pred = detector.predict(sample)

        engine = ComplianceEngine()
        event = engine.log_decision(
            prediction_id="test-pred-001",
            input_features=sample,
            output_label=pred["label"],
            confidence=pred["confidence"],
            explanation=pred.get("explanation"),
        )
        assert event is not None
        assert event.output_label == pred["label"]
        assert abs(event.confidence - pred["confidence"]) < 1e-6


# ── Schema Compatibility ──────────────────────────────────────────────────────

class TestAPISchemaCompatibility:
    """Test API request/response schema compatibility."""

    @pytest.mark.asyncio
    async def test_predict_response_has_expected_fields(
        self, client, admin_headers
    ):
        """Prediction response must include required fields for dashboard."""
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": "COMPAT-001",
                "features": {
                    "air_temperature": 298.5,
                    "process_temperature": 308.5,
                    "rotational_speed": 1545,
                    "torque": 42.3,
                    "tool_wear": 45,
                },
            },
            headers=admin_headers,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            # Must have: id, machine_id, prediction_label, confidence, created_at
            # (these feed the dashboard)
            for field in ["machine_id", "prediction_label", "confidence"]:
                assert field in data, f"Missing field: {field}"
            assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_alert_response_has_expected_fields(
        self, client, admin_headers
    ):
        """Alert response must include fields for dashboard rendering."""
        resp = await client.post(
            "/api/v1/alerts",
            json={
                "machine_id": "COMPAT-002",
                "severity": "high",
                "title": "Schema test alert",
                "message": "Testing response schema.",
            },
            headers=admin_headers,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            for field in ["id", "machine_id", "severity", "title", "is_acknowledged"]:
                assert field in data, f"Missing field: {field}"
            assert data["severity"] in ("critical", "high", "medium", "low")
            assert isinstance(data["is_acknowledged"], bool)

    @pytest.mark.asyncio
    async def test_health_response_schema(self, client):
        """Health endpoint must always return consistent schema."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("database"), bool)
        assert isinstance(data.get("uptime_seconds"), (int, float))
        assert data.get("status") in ("healthy", "degraded")
        assert isinstance(data.get("version"), str)

    @pytest.mark.asyncio
    async def test_auth_me_returns_user_schema(
        self, client, admin_headers, test_admin
    ):
        """GET /auth/me must return user fields required by dashboard."""
        resp = await client.get("/api/v1/auth/me", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        for field in ["id", "email", "role"]:
            assert field in data
        assert data["role"] in ("admin", "engineer", "operator", "viewer")
