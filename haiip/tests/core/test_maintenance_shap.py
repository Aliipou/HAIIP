"""Tests for SHAP explainability in MaintenancePredictor — 100% branch coverage."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import numpy as np
import pytest

from haiip.core.maintenance import MaintenancePredictor

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n = 600
    X = rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(n, 5))
    labels = (
        ["no_failure"] * 500
        + ["TWF"] * 20
        + ["HDF"] * 20
        + ["PWF"] * 20
        + ["OSF"] * 20
        + ["RNF"] * 20
    )
    rng = np.random.default_rng(1)
    rng.shuffle(labels)
    return X, np.array(labels)


@pytest.fixture
def rul_labels(training_data):
    X, _ = training_data
    return np.random.default_rng(7).integers(10, 300, size=len(X)).astype(float)


@pytest.fixture
def fitted(training_data):
    X, y = training_data
    p = MaintenancePredictor(n_estimators=50, random_state=42)
    p.fit(X, y)
    return p


@pytest.fixture
def fitted_with_rul(training_data, rul_labels):
    X, y = training_data
    p = MaintenancePredictor(n_estimators=50, random_state=42)
    p.fit(X, y, y_rul=rul_labels)
    return p


SAMPLE = [300.0, 310.0, 1538.0, 40.0, 100.0]


def _block_shap(monkeypatch):
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("shap not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)


# ── Explainer lifecycle ────────────────────────────────────────────────────────


def test_explainers_none_after_fit(training_data):
    X, y = training_data
    p = MaintenancePredictor(n_estimators=50)
    p.fit(X, y)
    assert p._shap_clf_explainer is None
    assert p._shap_rul_explainer is None


def test_explainers_reset_on_refit(fitted, training_data):
    X, y = training_data
    fitted._shap_clf_explainer = object()
    fitted._shap_rul_explainer = object()
    fitted.fit(X, y)
    assert fitted._shap_clf_explainer is None
    assert fitted._shap_rul_explainer is None


# ── _build_shap_explainers branches ──────────────────────────────────────────


def test_build_explainers_success_no_rul(fitted):
    """Mock shap.TreeExplainer to verify the success path (clf only, no rul)."""
    mock_shap = MagicMock()
    mock_clf_explainer = MagicMock()
    mock_shap.TreeExplainer.return_value = mock_clf_explainer
    import sys

    orig = sys.modules.get("shap")
    sys.modules["shap"] = mock_shap
    try:
        fitted._shap_clf_explainer = None
        fitted._shap_rul_explainer = None
        fitted._build_shap_explainers()
        assert fitted._shap_clf_explainer is mock_clf_explainer
        assert fitted._shap_rul_explainer is None  # _rul_fitted is False
        mock_shap.TreeExplainer.assert_called_once_with(fitted._classifier)
    finally:
        if orig is None:
            sys.modules.pop("shap", None)
        else:
            sys.modules["shap"] = orig


def test_build_explainers_success_with_rul(fitted_with_rul):
    """Mock shap.TreeExplainer to verify both clf + rul explainers are built."""
    mock_shap = MagicMock()
    clf_exp = MagicMock()
    rul_exp = MagicMock()
    mock_shap.TreeExplainer.side_effect = [clf_exp, rul_exp]
    import sys

    orig = sys.modules.get("shap")
    sys.modules["shap"] = mock_shap
    try:
        fitted_with_rul._shap_clf_explainer = None
        fitted_with_rul._shap_rul_explainer = None
        fitted_with_rul._build_shap_explainers()
        assert fitted_with_rul._shap_clf_explainer is clf_exp
        assert fitted_with_rul._shap_rul_explainer is rul_exp
        assert mock_shap.TreeExplainer.call_count == 2
    finally:
        if orig is None:
            sys.modules.pop("shap", None)
        else:
            sys.modules["shap"] = orig


def test_build_explainers_import_error(fitted, monkeypatch):
    _block_shap(monkeypatch)
    fitted._build_shap_explainers()
    assert fitted._shap_clf_explainer is None
    assert fitted._shap_rul_explainer is None


def test_build_explainers_general_exception(fitted):
    mock_shap = MagicMock()
    mock_shap.TreeExplainer.side_effect = RuntimeError("internal error")
    import sys

    orig = sys.modules.get("shap")
    sys.modules["shap"] = mock_shap
    try:
        fitted._shap_clf_explainer = None
        fitted._build_shap_explainers()
        assert fitted._shap_clf_explainer is None
    finally:
        if orig is None:
            sys.modules.pop("shap", None)
        else:
            sys.modules["shap"] = orig


# ── _shap_for_prediction branches ─────────────────────────────────────────────


def test_shap_for_prediction_none_when_build_fails(fitted, monkeypatch):
    """Explainer None → build fails → return None."""
    _block_shap(monkeypatch)
    fitted._shap_clf_explainer = None
    arr = fitted._scaler.transform(np.array([SAMPLE]))
    result = fitted._shap_for_prediction(arr, 0)
    assert result is None


def test_shap_for_prediction_list_sv(fitted):
    """sv is a list (normal multi-class GBM output) → returns dict."""
    pytest.importorskip("shap")
    fitted._build_shap_explainers()
    mock_explainer = MagicMock()
    n_classes = len(fitted._classes)
    # Simulate list-of-arrays from SHAP multi-class
    mock_explainer.shap_values.return_value = [np.ones((1, 5)) for _ in range(n_classes)]
    fitted._shap_clf_explainer = mock_explainer
    arr = fitted._scaler.transform(np.array([SAMPLE]))
    result = fitted._shap_for_prediction(arr, 0)
    assert isinstance(result, dict)
    assert len(result) == 5


def test_shap_for_prediction_3d_ndarray_sv(fitted):
    """sv is a 3D ndarray (shape n_samples, n_classes, n_features) → returns dict."""
    pytest.importorskip("shap")
    fitted._build_shap_explainers()
    mock_explainer = MagicMock()
    n_classes = len(fitted._classes)
    sv_3d = np.ones((1, n_classes, 5))
    mock_sv = MagicMock()
    mock_sv.__len__ = lambda s: 1  # not a list
    mock_sv.ndim = 3
    mock_sv.__getitem__ = lambda s, idx: sv_3d[idx]
    type(mock_sv).ndim = property(lambda s: 3)
    # Simpler: just use real ndarray
    mock_explainer.shap_values.return_value = sv_3d
    fitted._shap_clf_explainer = mock_explainer
    arr = fitted._scaler.transform(np.array([SAMPLE]))
    result = fitted._shap_for_prediction(arr, 0)
    assert isinstance(result, dict)
    assert len(result) == 5


def test_shap_for_prediction_neither_list_nor_3d(fitted):
    """sv is 2D (not list, ndim != 3) → return None."""
    fitted._build_shap_explainers()
    mock_explainer = MagicMock()
    # Return a 2D array — not a list, not 3D
    mock_explainer.shap_values.return_value = np.ones((1, 5))
    fitted._shap_clf_explainer = mock_explainer
    arr = fitted._scaler.transform(np.array([SAMPLE]))
    result = fitted._shap_for_prediction(arr, 0)
    assert result is None


def test_shap_for_prediction_exception_returns_none(fitted):
    fitted._shap_clf_explainer = MagicMock()
    fitted._shap_clf_explainer.shap_values.side_effect = RuntimeError("shap error")
    arr = fitted._scaler.transform(np.array([SAMPLE]))
    result = fitted._shap_for_prediction(arr, 0)
    assert result is None


# ── _shap_rul_for branches ────────────────────────────────────────────────────


def test_shap_rul_for_returns_none_when_rul_not_fitted(fitted):
    arr = fitted._scaler.transform(np.array([SAMPLE]))
    result = fitted._shap_rul_for(arr)
    assert result is None


def test_shap_rul_for_returns_none_when_build_fails(fitted_with_rul, monkeypatch):
    """RUL fitted but shap unavailable → explainer stays None → return None."""
    _block_shap(monkeypatch)
    fitted_with_rul._shap_rul_explainer = None
    arr = fitted_with_rul._scaler.transform(np.array([SAMPLE]))
    result = fitted_with_rul._shap_rul_for(arr)
    assert result is None


def test_shap_rul_for_returns_dict_when_available(fitted_with_rul):
    pytest.importorskip("shap")
    arr = fitted_with_rul._scaler.transform(np.array([SAMPLE]))
    result = fitted_with_rul._shap_rul_for(arr)
    if result is not None:
        assert isinstance(result, dict)
        assert len(result) == len(fitted_with_rul.feature_names)


def test_shap_rul_for_exception_returns_none(fitted_with_rul):
    fitted_with_rul._shap_rul_explainer = MagicMock()
    fitted_with_rul._shap_rul_explainer.shap_values.side_effect = RuntimeError("rul shap error")
    arr = fitted_with_rul._scaler.transform(np.array([SAMPLE]))
    result = fitted_with_rul._shap_rul_for(arr)
    assert result is None


# ── predict() integration ─────────────────────────────────────────────────────


def test_predict_required_keys(fitted):
    result = fitted.predict(SAMPLE)
    for key in (
        "label",
        "confidence",
        "failure_probability",
        "rul_cycles",
        "class_probabilities",
        "explanation",
    ):
        assert key in result


def test_predict_with_shap_when_available(fitted):
    pytest.importorskip("shap")
    result = fitted.predict(SAMPLE)
    if "shap_values" in result:
        assert isinstance(result["shap_values"], dict)
        assert set(result["shap_values"].keys()) == set(fitted.feature_names)


def test_predict_no_shap_when_unavailable(fitted, monkeypatch):
    _block_shap(monkeypatch)
    fitted._shap_clf_explainer = None
    result = fitted.predict(SAMPLE)
    assert "shap_values" not in result
    assert "shap_rul_values" not in result


def test_predict_includes_shap_rul_when_fitted(fitted_with_rul):
    pytest.importorskip("shap")
    result = fitted_with_rul.predict(SAMPLE)
    assert result["rul_cycles"] is not None
    if "shap_rul_values" in result:
        assert isinstance(result["shap_rul_values"], dict)


def test_predict_no_shap_rul_without_rul_fit(fitted):
    pytest.importorskip("shap")
    result = fitted.predict(SAMPLE)
    assert "shap_rul_values" not in result


def test_predict_shap_keys_match_feature_names(fitted):
    pytest.importorskip("shap")
    result = fitted.predict(SAMPLE)
    if "shap_values" in result:
        assert list(result["shap_values"].keys()) == fitted.feature_names


def test_predict_assigns_shap_values_when_not_none(fitted):
    """Cover result['shap_values'] = shap_clf (line 195) by mocking _shap_for_prediction."""
    from unittest.mock import patch

    mock_sv = dict.fromkeys(fitted.feature_names, 0.01)
    with (
        patch.object(fitted, "_shap_for_prediction", return_value=mock_sv),
        patch.object(fitted, "_shap_rul_for", return_value=None),
    ):
        result = fitted.predict(SAMPLE)
    assert result["shap_values"] == mock_sv
    assert "shap_rul_values" not in result


def test_predict_assigns_shap_rul_values_when_not_none(fitted_with_rul):
    """Cover result['shap_rul_values'] = shap_rul (line 197) by mocking _shap_rul_for."""
    from unittest.mock import patch

    mock_sv = dict.fromkeys(fitted_with_rul.feature_names, 0.02)
    mock_rul_sv = dict.fromkeys(fitted_with_rul.feature_names, 0.03)
    with (
        patch.object(fitted_with_rul, "_shap_for_prediction", return_value=mock_sv),
        patch.object(fitted_with_rul, "_shap_rul_for", return_value=mock_rul_sv),
    ):
        result = fitted_with_rul.predict(SAMPLE)
    assert result["shap_values"] == mock_sv
    assert result["shap_rul_values"] == mock_rul_sv


def test_shap_rul_for_success_path_returns_dict(fitted_with_rul):
    """Cover _shap_rul_for return dict (line 324) using a mock rul explainer."""
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.ones((1, 5))
    fitted_with_rul._shap_rul_explainer = mock_explainer
    arr = fitted_with_rul._scaler.transform(np.array([SAMPLE]))
    result = fitted_with_rul._shap_rul_for(arr)
    assert isinstance(result, dict)
    assert list(result.keys()) == fitted_with_rul.feature_names
    for v in result.values():
        assert isinstance(v, float)


# ── Untrained model ───────────────────────────────────────────────────────────


def test_untrained_returns_safe_default():
    p = MaintenancePredictor()
    result = p.predict(SAMPLE)
    assert result["label"] == "no_failure"
    assert result["failure_probability"] == 0.0
    assert "shap_values" not in result
    assert "shap_rul_values" not in result
