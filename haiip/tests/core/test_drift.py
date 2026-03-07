"""Tests for core/drift.py — DriftDetector and PageHinkleyDetector."""

import numpy as np
import pytest

from haiip.core.drift import DriftDetector, PageHinkleyDetector


@pytest.fixture
def reference_data():
    rng = np.random.default_rng(42)
    return rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(500, 5))


@pytest.fixture
def drifted_data():
    rng = np.random.default_rng(99)
    return rng.normal(loc=[305, 318, 1200, 60, 200], scale=[4, 3, 250, 20, 80], size=(200, 5))


@pytest.fixture
def fitted_detector(reference_data):
    d = DriftDetector(
        feature_names=["air_temp", "process_temp", "rpm", "torque", "wear"],
        drift_threshold=0.05,
    )
    d.fit_reference(reference_data)
    return d


# ── PageHinkley ───────────────────────────────────────────────────────────────


def test_ph_no_drift_stable_stream():
    ph = PageHinkleyDetector(threshold=50.0)
    rng = np.random.default_rng(0)
    drifts = [ph.update(float(v)) for v in rng.normal(0, 1, 1000)]
    assert sum(drifts) == 0


def test_ph_detects_step_change():
    ph = PageHinkleyDetector(threshold=10.0, delta=0.001)
    rng = np.random.default_rng(0)
    # Normal phase
    for v in rng.normal(0, 1, 200):
        ph.update(float(v))
    # Shifted phase — large step change
    drifts = [ph.update(float(v)) for v in rng.normal(10, 1, 100)]
    assert any(drifts), "PageHinkley should detect step change"


def test_ph_reset():
    ph = PageHinkleyDetector()
    ph.update(100.0)
    ph.reset()
    assert ph._sum == 0.0
    assert ph._n == 0


# ── DriftDetector ─────────────────────────────────────────────────────────────


def test_check_before_fit_raises():
    d = DriftDetector()
    with pytest.raises(RuntimeError, match="fit_reference"):
        d.check(np.ones((10, 5)))


def test_fit_reference_sets_state(reference_data):
    d = DriftDetector()
    d.fit_reference(reference_data, feature_names=["a", "b", "c", "d", "e"])
    assert d._reference is not None
    assert d.feature_names == ["a", "b", "c", "d", "e"]


def test_no_drift_on_same_distribution(fitted_detector, reference_data):
    results = fitted_detector.check(reference_data[:100])
    # Most features should be stable on same distribution
    stable_count = sum(1 for r in results if r.severity in ("stable", "monitoring"))
    assert stable_count >= 3


def test_drift_detected_on_shifted_distribution(fitted_detector, drifted_data):
    results = fitted_detector.check(drifted_data)
    assert any(r.drift_detected for r in results), "Drift should be detected on shifted data"


def test_check_result_fields(fitted_detector, reference_data):
    results = fitted_detector.check(reference_data[:50])
    for r in results:
        assert hasattr(r, "feature_name")
        assert hasattr(r, "drift_detected")
        assert hasattr(r, "psi")
        assert hasattr(r, "ks_statistic")
        assert hasattr(r, "ks_pvalue")
        assert r.severity in ("stable", "monitoring", "drift")
        assert 0.0 <= r.ks_pvalue <= 1.0


def test_summary_no_drift(fitted_detector, reference_data):
    results = fitted_detector.check(reference_data[:100])
    summary = fitted_detector.summary(results)
    assert "drift_detected" in summary
    assert "severity" in summary
    assert "affected_features" in summary
    assert isinstance(summary["feature_details"], list)


def test_summary_drift(fitted_detector, drifted_data):
    results = fitted_detector.check(drifted_data)
    summary = fitted_detector.summary(results)
    assert summary["drift_detected"] is True


def test_check_stream(fitted_detector):
    values = [300.0, 310.0, 1538.0, 40.0, 100.0]
    result = fitted_detector.check_stream(values)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(fitted_detector.feature_names)
