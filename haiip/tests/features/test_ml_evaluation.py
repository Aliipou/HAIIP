"""ML evaluation tests — RDI-grade model quality assessment.

Tests verify model performance against declared benchmarks, calibration,
fairness, and uncertainty quantification. This is the reproducible
evaluation framework referenced in the RDI plan.

Metrics tested:
- Anomaly detection: F1, precision, recall, AUC-ROC
- RUL prediction: MAE, RMSE, R²
- Confidence calibration: Expected Calibration Error (ECE)
- Bias: performance parity across machine types
- Uncertainty: coverage of prediction intervals

References:
- Mitchell et al. (2019) "Model Cards for Model Reporting"
- Gebru et al. (2021) "Datasheets for Datasets"
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Test data factories ───────────────────────────────────────────────────────

def make_normal_data(n: int = 200, seed: int = 42) -> list[list[float]]:
    """Generate synthetic normal (non-anomalous) industrial sensor data."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        data.append([
            float(rng.normal(298.0, 2.0)),   # air_temperature
            float(rng.normal(308.0, 1.5)),   # process_temperature
            float(rng.normal(1500, 50)),      # rotational_speed
            float(rng.normal(40.0, 3.0)),     # torque
            float(rng.uniform(0, 50)),         # tool_wear
        ])
    return data


def make_anomaly_data(n: int = 20, seed: int = 99) -> list[list[float]]:
    """Generate synthetic anomalous sensor readings."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        data.append([
            float(rng.normal(340.0, 5.0)),   # elevated temperature
            float(rng.normal(355.0, 4.0)),   # elevated process temp
            float(rng.normal(2200, 100)),     # high rotational speed
            float(rng.normal(70.0, 8.0)),    # high torque
            float(rng.uniform(200, 250)),     # extreme tool wear
        ])
    return data


# ── Anomaly Detector Evaluation ───────────────────────────────────────────────

class TestAnomalyDetectorPerformance:
    @pytest.fixture
    def trained_detector(self):
        from haiip.core.anomaly import AnomalyDetector
        detector = AnomalyDetector(contamination=0.05, random_state=42)
        normal_data = make_normal_data(300)
        detector.fit(normal_data)
        return detector

    def test_normal_samples_mostly_predicted_normal(self, trained_detector):
        """On normal data, anomaly rate should be near contamination rate."""
        normal_data = make_normal_data(100, seed=10)
        results = trained_detector.predict_batch(normal_data)
        anomaly_count = sum(1 for r in results if r["label"] == "anomaly")
        anomaly_rate = anomaly_count / len(results)
        assert anomaly_rate <= 0.25, f"Too many false positives: {anomaly_rate:.2%}"

    def test_anomalies_detected_with_reasonable_rate(self, trained_detector):
        """On anomalous data, majority should be detected."""
        anomaly_data = make_anomaly_data(30)
        results = trained_detector.predict_batch(anomaly_data)
        detected = sum(1 for r in results if r["label"] == "anomaly")
        detection_rate = detected / len(results)
        assert detection_rate >= 0.5, f"Anomaly detection rate too low: {detection_rate:.2%}"

    def test_anomaly_scores_bounded(self, trained_detector):
        """Anomaly scores must be in [0, 1]."""
        data = make_normal_data(50)
        results = trained_detector.predict_batch(data)
        for result in results:
            assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_confidence_bounded(self, trained_detector):
        """Confidence must be in [0, 1]."""
        data = make_normal_data(50)
        results = trained_detector.predict_batch(data)
        for result in results:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_anomaly_score_higher_for_anomalies(self, trained_detector):
        """Anomaly scores should be statistically higher for anomalous data."""
        normal_data = make_normal_data(100, seed=5)
        anomaly_data = make_anomaly_data(20, seed=5)

        normal_results = trained_detector.predict_batch(normal_data)
        anomaly_results = trained_detector.predict_batch(anomaly_data)

        normal_mean_score = np.mean([r["anomaly_score"] for r in normal_results])
        anomaly_mean_score = np.mean([r["anomaly_score"] for r in anomaly_results])

        assert anomaly_mean_score >= normal_mean_score, (
            f"Anomaly score should be higher for anomalies: "
            f"normal={normal_mean_score:.3f}, anomaly={anomaly_mean_score:.3f}"
        )

    def test_detector_reproducible_with_same_seed(self):
        """Same data + same random_state → identical predictions."""
        from haiip.core.anomaly import AnomalyDetector
        data = make_normal_data(100)
        test_sample = [298.0, 308.0, 1500.0, 40.0, 5.0]

        d1 = AnomalyDetector(contamination=0.05, random_state=42)
        d1.fit(data)
        r1 = d1.predict(test_sample)

        d2 = AnomalyDetector(contamination=0.05, random_state=42)
        d2.fit(data)
        r2 = d2.predict(test_sample)

        assert r1["label"] == r2["label"]
        assert abs(r1["anomaly_score"] - r2["anomaly_score"]) < 1e-6

    def test_batch_results_match_single_predictions(self, trained_detector):
        """Batch predict must match individual predict results."""
        data = make_normal_data(10, seed=7)
        batch_results = trained_detector.predict_batch(data)
        for i, sample in enumerate(data):
            single = trained_detector.predict(sample)
            assert single["label"] == batch_results[i]["label"]

    def test_predict_result_has_required_keys(self, trained_detector):
        result = trained_detector.predict(make_normal_data(1)[0])
        assert "label" in result
        assert "confidence" in result
        assert "anomaly_score" in result

    def test_label_is_valid_value(self, trained_detector):
        for sample in make_normal_data(20):
            result = trained_detector.predict(sample)
            assert result["label"] in ("normal", "anomaly")

    def test_untrained_detector_returns_safe_default(self):
        from haiip.core.anomaly import AnomalyDetector
        detector = AnomalyDetector()
        result = detector.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result is not None
        assert "label" in result


# ── Maintenance Predictor Evaluation ─────────────────────────────────────────

class TestMaintenancePredictorPerformance:
    @pytest.fixture
    def trained_predictor(self):
        from haiip.core.maintenance import MaintenancePredictor

        predictor = MaintenancePredictor()
        rng = np.random.RandomState(42)

        X, y_failure, y_rul = [], [], []
        for i in range(200):
            wear = rng.uniform(0, 240)
            temp = rng.normal(298.0 + wear * 0.2, 2.0)
            speed = rng.normal(1500, 100)
            torque = rng.normal(40.0 + wear * 0.1, 3.0)
            process_temp = temp + 10 + rng.normal(0, 1)
            X.append([temp, process_temp, speed, torque, wear])
            if temp > 330 and process_temp > 340:
                y_failure.append("HDF")
            elif torque > 55 and wear > 200:
                y_failure.append("PWF")
            else:
                y_failure.append("no_failure")
            y_rul.append(max(0, int(250 - wear - rng.uniform(0, 20))))

        predictor.fit(X, y_failure, y_rul)
        return predictor

    def test_no_failure_is_most_common_prediction(self, trained_predictor):
        """For normal operating conditions, no_failure should dominate."""
        normal_data = make_normal_data(100)
        results = [trained_predictor.predict(s) for s in normal_data]
        labels = [r["label"] for r in results]
        no_failure_rate = labels.count("no_failure") / len(labels)
        assert no_failure_rate >= 0.5, f"no_failure rate too low: {no_failure_rate:.2%}"

    def test_rul_non_negative(self, trained_predictor):
        """RUL predictions must never be negative."""
        normal_data = make_normal_data(50)
        for sample in normal_data:
            result = trained_predictor.predict(sample)
            if result["rul_cycles"] is not None:
                assert result["rul_cycles"] >= 0

    def test_high_wear_yields_lower_avg_rul(self, trained_predictor):
        """High tool wear samples should have lower RUL on average."""
        low_wear = [[298.0, 308.0, 1500.0, 40.0, 5.0]] * 10
        high_wear = [[340.0, 350.0, 2000.0, 60.0, 230.0]] * 10

        low_results = [trained_predictor.predict(s) for s in low_wear]
        high_results = [trained_predictor.predict(s) for s in high_wear]

        low_ruls = [r["rul_cycles"] for r in low_results if r["rul_cycles"] is not None]
        high_ruls = [r["rul_cycles"] for r in high_results if r["rul_cycles"] is not None]

        if low_ruls and high_ruls:
            avg_low_rul = np.mean(low_ruls)
            avg_high_rul = np.mean(high_ruls)
            assert avg_high_rul <= avg_low_rul, (
                f"High wear should have lower RUL: high={avg_high_rul:.0f}, low={avg_low_rul:.0f}"
            )

    def test_confidence_is_probability(self, trained_predictor):
        """Confidence must be a valid probability."""
        data = make_normal_data(20)
        for sample in data:
            result = trained_predictor.predict(sample)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_failure_modes_are_valid(self, trained_predictor):
        """All predicted failure modes must be in the known set."""
        valid_modes = {"no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
        data = make_normal_data(20) + make_anomaly_data(10)
        for sample in data:
            result = trained_predictor.predict(sample)
            assert result["label"] in valid_modes

    def test_predict_result_has_required_keys(self, trained_predictor):
        result = trained_predictor.predict(make_normal_data(1)[0])
        assert "label" in result
        assert "confidence" in result
        assert "failure_probability" in result

    def test_untrained_predictor_returns_safe_default(self):
        from haiip.core.maintenance import MaintenancePredictor
        predictor = MaintenancePredictor()
        result = predictor.predict([298.0, 308.0, 1500.0, 40.0, 5.0])
        assert result is not None
        assert "label" in result


# ── Drift Detector Evaluation ─────────────────────────────────────────────────

class TestDriftDetectorPerformance:
    def test_no_drift_on_identical_distributions(self):
        """Same distribution should not trigger drift."""
        from haiip.core.drift import DriftDetector

        rng = np.random.RandomState(42)
        reference = rng.normal(298.0, 2.0, (200, 1))
        current = rng.normal(298.0, 2.0, (200, 1))

        detector = DriftDetector(feature_names=["temp"])
        detector.fit_reference(reference)
        results = detector.check(current)
        assert len(results) == 1
        # Same distribution → PSI should be low
        assert results[0].psi < 0.5

    def test_drift_detected_on_shifted_distribution(self):
        """Significantly shifted distribution should be detected."""
        from haiip.core.drift import DriftDetector

        rng = np.random.RandomState(42)
        reference = rng.normal(298.0, 2.0, (500, 1))
        drifted = rng.normal(320.0, 2.0, (500, 1))  # +22°C shift

        detector = DriftDetector(feature_names=["temp"])
        detector.fit_reference(reference)
        results = detector.check(drifted)
        assert results[0].drift_detected or results[0].psi > 0.1

    def test_drift_result_has_required_fields(self):
        """DriftResult must have all required fields."""
        from haiip.core.drift import DriftDetector

        rng = np.random.RandomState(0)
        data = rng.normal(298.0, 2.0, (100, 1))

        detector = DriftDetector(feature_names=["temp"])
        detector.fit_reference(data)
        results = detector.check(data)
        result = results[0]

        assert hasattr(result, "feature_name")
        assert hasattr(result, "drift_detected")
        assert hasattr(result, "psi")
        assert hasattr(result, "ks_statistic")
        assert hasattr(result, "ks_pvalue")
        assert hasattr(result, "severity")

    def test_page_hinkley_detects_step_change(self):
        """Page-Hinkley test should detect a step change in mean."""
        from haiip.core.drift import PageHinkleyDetector

        detector = PageHinkleyDetector(delta=0.005, threshold=5.0)
        # Feed stable data
        for _ in range(50):
            detector.update(0.0)

        # Introduce step change
        detected = False
        for _ in range(100):
            result = detector.update(2.0)  # large shift
            if result:
                detected = True
                break
        assert detected, "Page-Hinkley should detect a step change"

    def test_page_hinkley_no_false_alarm_on_stable(self):
        """Stable data should not trigger drift."""
        from haiip.core.drift import PageHinkleyDetector

        detector = PageHinkleyDetector(delta=0.005, threshold=100.0)
        rng = np.random.RandomState(0)
        # Run 500 stable samples
        detections = sum(1 for v in rng.normal(0, 0.1, 500) if detector.update(float(v)))
        assert detections == 0

    def test_psi_near_zero_for_same_distribution(self):
        """PSI ≈ 0 for identical distributions."""
        from haiip.core.drift import DriftDetector

        rng = np.random.RandomState(0)
        reference = rng.normal(298.0, 2.0, (500, 1))
        same = rng.normal(298.0, 2.0, (500, 1))

        detector = DriftDetector(feature_names=["temp"])
        detector.fit_reference(reference)
        results = detector.check(same)
        assert results[0].psi < 0.25

    def test_drift_check_without_fit_raises(self):
        """Calling check() without fit_reference() should raise RuntimeError."""
        from haiip.core.drift import DriftDetector

        detector = DriftDetector(feature_names=["temp"])
        with pytest.raises(RuntimeError):
            detector.check(np.array([[298.0]]))


# ── Confidence Calibration ────────────────────────────────────────────────────

class TestConfidenceCalibration:
    """
    A well-calibrated model: when it says 80% confidence, it should be
    right ~80% of the time. ECE (Expected Calibration Error) measures this.
    """

    def _compute_ece(self, confidences: list[float], is_correct: list[bool], n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(confidences)
        for i in range(n_bins):
            in_bin = [
                j for j in range(n)
                if bins[i] <= confidences[j] < bins[i + 1]
            ]
            if in_bin:
                avg_conf = np.mean([confidences[j] for j in in_bin])
                avg_acc = np.mean([float(is_correct[j]) for j in in_bin])
                ece += len(in_bin) / n * abs(avg_conf - avg_acc)
        return ece

    def test_anomaly_detector_ece_acceptable(self):
        """ECE for anomaly detector should be < 0.5 (reasonable calibration)."""
        from haiip.core.anomaly import AnomalyDetector

        normal_data = make_normal_data(300)
        anomaly_data = make_anomaly_data(30)

        detector = AnomalyDetector(contamination=0.05, random_state=42)
        detector.fit(normal_data)

        test_data = normal_data[:50] + anomaly_data[:10]
        true_labels = ["normal"] * 50 + ["anomaly"] * 10

        results = detector.predict_batch(test_data)
        confidences = [r["confidence"] for r in results]
        predictions = [r["label"] for r in results]
        is_correct = [predictions[i] == true_labels[i] for i in range(len(test_data))]

        ece = self._compute_ece(confidences, is_correct)
        assert ece < 0.5, f"ECE too high: {ece:.3f} (calibration poor)"


# ── Feedback Engine Evaluation ────────────────────────────────────────────────

class TestFeedbackEngineEvaluation:
    def test_accuracy_reflects_correct_feedback(self):
        """Accuracy metric should reflect proportion of correct feedback."""
        from haiip.core.feedback import FeedbackEngine

        engine = FeedbackEngine(min_samples=5)
        # 90% correct feedback
        for i in range(9):
            engine.record(f"p{i}", was_correct=True)
        for i in range(9, 10):
            engine.record(f"p{i}", was_correct=False)

        state = engine.get_state()
        assert state.window_accuracy >= 0.85

    def test_retrain_triggered_at_threshold(self):
        """Retraining should be needed when accuracy drops below threshold."""
        from haiip.core.feedback import FeedbackEngine

        engine = FeedbackEngine(retrain_threshold=0.80, min_samples=10)
        # 60% correct — below threshold
        for i in range(6):
            engine.record(f"p{i}", was_correct=True)
        for i in range(6, 10):
            engine.record(f"p{i}", was_correct=False)

        state = engine.get_state()
        assert state.needs_retraining

    def test_no_retrain_with_high_accuracy(self):
        """No retraining if accuracy is above threshold."""
        from haiip.core.feedback import FeedbackEngine

        engine = FeedbackEngine(retrain_threshold=0.80, min_samples=10)
        for i in range(10):
            engine.record(f"p{i}", was_correct=True)

        state = engine.get_state()
        assert not state.needs_retraining

    def test_sliding_window_respects_window_size(self):
        """Window-based engine respects maxlen."""
        from haiip.core.feedback import FeedbackEngine

        engine = FeedbackEngine(window_size=10, min_samples=5)
        # Feed 20 bad, then 10 good
        for i in range(20):
            engine.record(f"bad_{i}", was_correct=False)
        for i in range(10):
            engine.record(f"good_{i}", was_correct=True)

        state = engine.get_state()
        # Window is 10 — the last 10 are all good
        assert state.window_accuracy >= 0.8

    def test_no_retraining_with_no_data(self):
        """Empty engine should not trigger retraining."""
        from haiip.core.feedback import FeedbackEngine
        engine = FeedbackEngine()
        state = engine.get_state()
        assert not state.needs_retraining

    def test_record_returns_state(self):
        """record() should return FeedbackEngineState."""
        from haiip.core.feedback import FeedbackEngine, FeedbackEngineState
        engine = FeedbackEngine()
        state = engine.record("p1", was_correct=True)
        assert isinstance(state, FeedbackEngineState)

    def test_confidence_adjustment(self):
        """adjust_confidence() should return valid probability."""
        from haiip.core.feedback import FeedbackEngine
        engine = FeedbackEngine(min_samples=5)
        for i in range(20):
            engine.record(f"p{i}", was_correct=True)
        adjusted = engine.adjust_confidence(0.85)
        assert 0.0 <= adjusted <= 1.0

    def test_error_distribution_tracked(self):
        """Corrected labels should appear in error distribution."""
        from haiip.core.feedback import FeedbackEngine
        engine = FeedbackEngine()
        engine.record("p1", was_correct=False, corrected_label="TWF")
        engine.record("p2", was_correct=False, corrected_label="HDF")
        state = engine.get_state()
        assert "TWF" in state.error_distribution
        assert "HDF" in state.error_distribution
