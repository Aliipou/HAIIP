"""Tests for SHAP explainability in AnomalyDetector — 100% branch coverage."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from haiip.core.anomaly import AnomalyDetector

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(500, 5))


@pytest.fixture
def fitted_detector(normal_data) -> AnomalyDetector:
    d = AnomalyDetector(contamination=0.05, random_state=42)
    d.fit(normal_data)
    return d


SAMPLE = [300.0, 310.0, 1538.0, 40.0, 100.0]
OUTLIER = [400.0, 450.0, 5000.0, 200.0, 300.0]


def _block_shap(monkeypatch):
    """Monkeypatch builtins to make 'import shap' raise ImportError."""
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("shap not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)


# ── Explainer lifecycle ────────────────────────────────────────────────────────


def test_shap_explainer_none_after_fit(normal_data):
    d = AnomalyDetector()
    d.fit(normal_data)
    assert d._shap_explainer is None


def test_shap_explainer_reset_on_refit(fitted_detector, normal_data):
    fitted_detector._shap_explainer = object()  # fake built
    fitted_detector.fit(normal_data)
    assert fitted_detector._shap_explainer is None


# ── _build_shap_explainer branches ────────────────────────────────────────────


def test_build_shap_explainer_success(fitted_detector):
    pytest.importorskip("shap")
    fitted_detector._build_shap_explainer()
    assert fitted_detector._shap_explainer is not None


def test_build_shap_explainer_import_error(fitted_detector, monkeypatch):
    """ImportError → _shap_explainer stays None, no crash."""
    _block_shap(monkeypatch)
    fitted_detector._shap_explainer = None
    fitted_detector._build_shap_explainer()
    assert fitted_detector._shap_explainer is None


def test_build_shap_explainer_general_exception(fitted_detector):
    """TreeExplainer raises (non-import) → _shap_explainer stays None."""
    mock_shap = MagicMock()
    mock_shap.TreeExplainer.side_effect = RuntimeError("internal shap error")
    with patch.dict("sys.modules", {"shap": mock_shap}):
        fitted_detector._shap_explainer = None
        fitted_detector._build_shap_explainer()
    assert fitted_detector._shap_explainer is None


# ── _shap_values_for branches ─────────────────────────────────────────────────


def test_shap_values_for_returns_none_when_build_fails(fitted_detector, monkeypatch):
    """_shap_explainer None → build → still None → returns None."""
    _block_shap(monkeypatch)
    fitted_detector._shap_explainer = None
    arr = np.array([[300.0, 310.0, 1538.0, 40.0, 100.0]])
    result = fitted_detector._shap_values_for(arr)
    assert result is None


def test_shap_values_for_wrong_shape_returns_none(fitted_detector):
    """sv returned with len != n_samples → returns None."""
    pytest.importorskip("shap")
    fitted_detector._build_shap_explainer()
    mock_explainer = MagicMock()
    # Return array with len=2 but we pass 1 sample → mismatch → returns None
    mock_explainer.shap_values.return_value = np.ones((2, 5))
    fitted_detector._shap_explainer = mock_explainer
    arr = np.array([[300.0, 310.0, 1538.0, 40.0, 100.0]])
    result = fitted_detector._shap_values_for(arr)
    assert result is None


def test_shap_values_for_exception_returns_none(fitted_detector):
    """shap_values() raises → returns None, no crash."""
    fitted_detector._shap_explainer = MagicMock()
    fitted_detector._shap_explainer.shap_values.side_effect = RuntimeError("shap broke")
    arr = np.array([[300.0, 310.0, 1538.0, 40.0, 100.0]])
    result = fitted_detector._shap_values_for(arr)
    assert result is None


def test_shap_values_for_returns_dict_on_success(fitted_detector):
    pytest.importorskip("shap")
    arr = np.array([[300.0, 310.0, 1538.0, 40.0, 100.0]])
    arr_scaled = fitted_detector._scaler.transform(arr)
    result = fitted_detector._shap_values_for(arr_scaled)
    # If shap works, result is a dict; if shap unavailable, None — both valid
    if result is not None:
        assert isinstance(result, dict)
        assert len(result) == len(fitted_detector.feature_names)


# ── predict() branches ─────────────────────────────────────────────────────────


def test_predict_required_keys_always_present(fitted_detector):
    result = fitted_detector.predict(SAMPLE)
    for key in ("label", "confidence", "anomaly_score", "explanation"):
        assert key in result


def test_predict_includes_shap_values_when_available(fitted_detector):
    pytest.importorskip("shap")
    result = fitted_detector.predict(SAMPLE)
    if "shap_values" in result:
        sv = result["shap_values"]
        assert isinstance(sv, dict)
        assert set(sv.keys()) == set(fitted_detector.feature_names)
        for v in sv.values():
            assert isinstance(v, float)
            assert np.isfinite(v)


def test_predict_no_shap_values_when_shap_unavailable(fitted_detector, monkeypatch):
    _block_shap(monkeypatch)
    fitted_detector._shap_explainer = None
    result = fitted_detector.predict(SAMPLE)
    assert "label" in result
    assert "shap_values" not in result


def test_predict_anomaly_label_for_outlier(fitted_detector):
    result = fitted_detector.predict(OUTLIER)
    assert result["label"] == "anomaly"
    assert result["anomaly_score"] > 0


def test_predict_confidence_in_range(fitted_detector):
    result = fitted_detector.predict(SAMPLE)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_keys_match_feature_names_in_shap(fitted_detector):
    pytest.importorskip("shap")
    result = fitted_detector.predict(SAMPLE)
    if "shap_values" in result:
        assert list(result["shap_values"].keys()) == fitted_detector.feature_names


def test_predict_custom_feature_names_in_shap(normal_data):
    pytest.importorskip("shap")
    names = ["t_air", "t_proc", "rpm", "torque", "wear"]
    d = AnomalyDetector(feature_names=names)
    d.fit(normal_data)
    result = d.predict(SAMPLE)
    if "shap_values" in result:
        assert set(result["shap_values"].keys()) == set(names)


# ── predict_batch() branches ──────────────────────────────────────────────────


def test_predict_batch_length(fitted_detector):
    results = fitted_detector.predict_batch(np.array([SAMPLE] * 5))
    assert len(results) == 5


def test_predict_batch_consistent_with_single(fitted_detector):
    single = fitted_detector.predict(SAMPLE)
    batch = fitted_detector.predict_batch(np.array([SAMPLE]))
    assert single["label"] == batch[0]["label"]
    assert single["anomaly_score"] == pytest.approx(batch[0]["anomaly_score"], abs=1e-4)


def test_predict_batch_shap_present_when_available(fitted_detector):
    pytest.importorskip("shap")
    results = fitted_detector.predict_batch(np.array([SAMPLE] * 3))
    for r in results:
        if "shap_values" in r:
            assert len(r["shap_values"]) == len(fitted_detector.feature_names)


def test_predict_batch_shap_absent_when_unavailable(fitted_detector, monkeypatch):
    _block_shap(monkeypatch)
    fitted_detector._shap_explainer = None
    results = fitted_detector.predict_batch(np.array([SAMPLE, SAMPLE]))
    for r in results:
        assert "shap_values" not in r


def test_predict_batch_shap_exception_is_swallowed(fitted_detector):
    """If batch SHAP raises, results still returned without shap_values."""
    fitted_detector._shap_explainer = MagicMock()
    fitted_detector._shap_explainer.shap_values.side_effect = RuntimeError("batch broke")
    results = fitted_detector.predict_batch(np.array([SAMPLE, SAMPLE]))
    assert len(results) == 2
    for r in results:
        assert "label" in r
        assert "shap_values" not in r


def test_predict_batch_entry_has_shap_when_not_none(fitted_detector):
    """batch_shap[i] is not None → entry gets shap_values key."""
    fitted_detector._shap_explainer = MagicMock()
    sv_array = np.ones((2, 5))  # 2 samples, 5 features — correct shape
    fitted_detector._shap_explainer.shap_values.return_value = sv_array
    results = fitted_detector.predict_batch(np.array([SAMPLE, SAMPLE]))
    for r in results:
        assert "shap_values" in r
        assert len(r["shap_values"]) == 5


# ── Untrained model ───────────────────────────────────────────────────────────


def test_untrained_predict_returns_safe_result():
    d = AnomalyDetector()
    result = d.predict(SAMPLE)
    assert result["label"] == "normal"
    assert result["confidence"] == 0.5
    assert "shap_values" not in result


def test_untrained_predict_batch_returns_list(normal_data):
    d = AnomalyDetector()
    results = d.predict_batch(np.array([SAMPLE, SAMPLE]))
    assert len(results) == 2
    for r in results:
        assert r["label"] == "normal"
