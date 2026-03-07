"""Tests for core/anomaly.py — AnomalyDetector."""

import numpy as np
import pytest

from haiip.core.anomaly import AnomalyDetector

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(500, 5))


@pytest.fixture
def fitted_detector(normal_data: np.ndarray) -> AnomalyDetector:
    d = AnomalyDetector(contamination=0.05, random_state=42)
    d.fit(normal_data)
    return d


# ── Init ──────────────────────────────────────────────────────────────────────


def test_init_defaults():
    d = AnomalyDetector()
    assert d.contamination == 0.05
    assert d.n_estimators == 100
    assert not d.is_fitted


def test_init_custom():
    d = AnomalyDetector(contamination=0.1, n_estimators=50, random_state=0)
    assert d.contamination == 0.1
    assert d.n_estimators == 50


# ── Fit ───────────────────────────────────────────────────────────────────────


def test_fit_sets_fitted_flag(normal_data):
    d = AnomalyDetector()
    d.fit(normal_data)
    assert d.is_fitted


def test_fit_returns_self(normal_data):
    d = AnomalyDetector()
    result = d.fit(normal_data)
    assert result is d


def test_fit_raises_on_1d():
    d = AnomalyDetector()
    with pytest.raises(ValueError, match="2D"):
        d.fit(np.array([1, 2, 3]))


def test_fit_from_dataframe(normal_data):
    import pandas as pd

    df = pd.DataFrame(
        normal_data,
        columns=[
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ],
    )
    d = AnomalyDetector()
    d.fit_from_dataframe(df)
    assert d.is_fitted


def test_fit_from_dataframe_missing_col(normal_data):
    import pandas as pd

    df = pd.DataFrame(normal_data, columns=["a", "b", "c", "d", "e"])
    d = AnomalyDetector()
    with pytest.raises(ValueError, match="Missing columns"):
        d.fit_from_dataframe(df)


# ── Predict ───────────────────────────────────────────────────────────────────


def test_predict_unfitted_returns_safe_default():
    d = AnomalyDetector()
    result = d.predict([300, 310, 1538, 40, 100])
    assert result["label"] == "normal"
    assert result["confidence"] == 0.5


def test_predict_normal_reading(fitted_detector):
    result = fitted_detector.predict([300, 310, 1538, 40, 100])
    assert result["label"] == "normal"
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert "explanation" in result


def test_predict_anomalous_reading(fitted_detector):
    # Extreme values far outside training distribution
    result = fitted_detector.predict([350, 400, 5000, 200, 500])
    assert result["label"] == "anomaly"
    assert result["anomaly_score"] > 0.5


def test_predict_returns_required_keys(fitted_detector):
    result = fitted_detector.predict([300, 310, 1538, 40, 100])
    assert {"label", "confidence", "anomaly_score", "explanation"}.issubset(result.keys())


def test_predict_confidence_range(fitted_detector, normal_data):
    for row in normal_data[:50]:
        result = fitted_detector.predict(row.tolist())
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["anomaly_score"] <= 1.0


# ── Batch predict ─────────────────────────────────────────────────────────────


def test_predict_batch_returns_list(fitted_detector, normal_data):
    results = fitted_detector.predict_batch(normal_data[:10])
    assert len(results) == 10
    for r in results:
        assert "label" in r
        assert "confidence" in r


def test_predict_batch_unfitted():
    d = AnomalyDetector()
    results = d.predict_batch(np.ones((5, 5)))
    assert len(results) == 5
    assert all(r["label"] == "normal" for r in results)


# ── Save / Load ───────────────────────────────────────────────────────────────


def test_save_raises_if_not_fitted(tmp_path):
    d = AnomalyDetector()
    with pytest.raises(RuntimeError, match="unfitted"):
        d.save(tmp_path / "model")


def test_save_and_load_roundtrip(fitted_detector, tmp_path, normal_data):
    path = tmp_path / "anomaly_model"
    fitted_detector.save(path)

    loaded = AnomalyDetector.load(path)
    assert loaded.is_fitted

    # Predictions should be identical
    original = fitted_detector.predict([300, 310, 1538, 40, 100])
    restored = loaded.predict([300, 310, 1538, 40, 100])
    assert original["label"] == restored["label"]
    assert abs(original["confidence"] - restored["confidence"]) < 0.001
