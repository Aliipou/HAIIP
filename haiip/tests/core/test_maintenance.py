"""Tests for core/maintenance.py — MaintenancePredictor."""

import numpy as np
import pytest

from haiip.core.maintenance import FAILURE_MODES, MaintenancePredictor


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n = 600
    X = rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(n, 5))
    labels = ["no_failure"] * 500 + ["TWF"] * 20 + ["HDF"] * 20 + ["PWF"] * 20 + ["OSF"] * 20 + ["RNF"] * 20
    rng.shuffle(labels)
    return X, np.array(labels)


@pytest.fixture
def fitted_predictor(training_data):
    X, y = training_data
    p = MaintenancePredictor(n_estimators=50, random_state=42)
    p.fit(X, y)
    return p


def test_init_defaults():
    p = MaintenancePredictor()
    assert not p.is_fitted


def test_fit_sets_fitted(training_data):
    X, y = training_data
    p = MaintenancePredictor(n_estimators=50)
    p.fit(X, y)
    assert p.is_fitted


def test_fit_returns_self(training_data):
    X, y = training_data
    p = MaintenancePredictor(n_estimators=50)
    assert p.fit(X, y) is p


def test_predict_unfitted_returns_default():
    p = MaintenancePredictor()
    result = p.predict([300, 310, 1538, 40, 100])
    assert result["label"] == "no_failure"
    assert result["confidence"] == 0.5


def test_predict_returns_required_keys(fitted_predictor):
    result = fitted_predictor.predict([300, 310, 1538, 40, 100])
    assert {"label", "confidence", "failure_probability", "explanation"}.issubset(result.keys())


def test_predict_label_in_known_modes(fitted_predictor):
    result = fitted_predictor.predict([300, 310, 1538, 40, 100])
    assert result["label"] in FAILURE_MODES


def test_predict_confidence_range(fitted_predictor, training_data):
    X, _ = training_data
    for row in X[:20]:
        result = fitted_predictor.predict(row.tolist())
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["failure_probability"] <= 1.0


def test_predict_batch(fitted_predictor, training_data):
    X, _ = training_data
    results = fitted_predictor.predict_batch(X[:10])
    assert len(results) == 10
    for r in results:
        assert r["label"] in FAILURE_MODES


def test_predict_batch_unfitted():
    p = MaintenancePredictor()
    results = p.predict_batch(np.ones((3, 5)))
    assert len(results) == 3


def test_save_load_roundtrip(fitted_predictor, tmp_path):
    path = tmp_path / "maintenance_model"
    fitted_predictor.save(path)
    loaded = MaintenancePredictor.load(path)
    assert loaded.is_fitted

    orig = fitted_predictor.predict([300, 310, 1538, 40, 100])
    rest = loaded.predict([300, 310, 1538, 40, 100])
    assert orig["label"] == rest["label"]


def test_save_raises_if_unfitted(tmp_path):
    p = MaintenancePredictor()
    with pytest.raises(RuntimeError):
        p.save(tmp_path / "model")


def test_fit_with_rul(training_data):
    X, y = training_data
    rul = np.random.default_rng(42).integers(0, 500, size=len(y)).astype(float)
    p = MaintenancePredictor(n_estimators=50)
    p.fit(X, y, y_rul=rul)
    result = p.predict([300, 310, 1538, 40, 100])
    assert result["rul_cycles"] is not None
    assert result["rul_cycles"] >= 0
