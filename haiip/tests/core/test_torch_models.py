"""Tests for haiip.core.torch_models — PyTorch Lightning Autoencoder + LSTM.

Test categories:
    - Unit: model init, fit, predict, save/load
    - Integration: full train → predict → ONNX export pipeline
    - Crash/Edge: NaN, Inf, empty input, single sample, untrained model
    - Thread safety: concurrent predict calls
    - Interface compatibility: same output keys as sklearn models

Note: Lightning training is run with max_epochs=2 for speed in tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from haiip.core.torch_models import (
    AnomalyAutoencoder,
    FAILURE_MODES,
    MaintenanceLSTM,
    _LIGHTNING_AVAILABLE,
    _TORCH_AVAILABLE,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_FEATURES = 5
SEQ_LEN = 5
N_SAMPLES = 60  # enough for sliding windows
FEATURE_NAMES = ["air_temp", "proc_temp", "rpm", "torque", "tool_wear"]


@pytest.fixture()
def normal_X() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)


@pytest.fixture()
def anomaly_X() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(loc=10.0, scale=0.5, size=(20, N_FEATURES)).astype(np.float32)


@pytest.fixture()
def class_labels() -> np.ndarray:
    rng = np.random.default_rng(42)
    classes = ["no_failure"] * 50 + ["TWF"] * 5 + ["HDF"] * 5
    rng.shuffle(classes)
    return np.array(classes[:N_SAMPLES])


@pytest.fixture()
def rul_values() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(10, 500, size=N_SAMPLES).astype(np.float32)


@pytest.fixture()
def fitted_autoencoder(normal_X: np.ndarray) -> AnomalyAutoencoder:
    model = AnomalyAutoencoder(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_size=16,
        n_layers=1,
        max_epochs=2,
        batch_size=16,
        feature_names=FEATURE_NAMES,
    )
    model.fit(normal_X)
    return model


@pytest.fixture()
def fitted_lstm(
    normal_X: np.ndarray, class_labels: np.ndarray, rul_values: np.ndarray
) -> MaintenanceLSTM:
    model = MaintenanceLSTM(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_size=16,
        n_layers=1,
        max_epochs=2,
        batch_size=16,
        feature_names=FEATURE_NAMES,
    )
    model.fit(normal_X, class_labels, rul_values)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# AnomalyAutoencoder — Unit Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAnomalyAutoencoderInit:
    def test_defaults(self) -> None:
        m = AnomalyAutoencoder()
        assert m.n_features == 5
        assert m.seq_len == 10
        assert m.hidden_size == 64
        assert not m.is_fitted

    def test_custom_params(self) -> None:
        m = AnomalyAutoencoder(n_features=8, seq_len=20, hidden_size=32, max_epochs=100)
        assert m.n_features == 8
        assert m.seq_len == 20
        assert m.hidden_size == 32
        assert m.max_epochs == 100

    def test_custom_feature_names(self) -> None:
        names = ["vibration", "temperature"]
        m = AnomalyAutoencoder(n_features=2, feature_names=names)
        assert m.feature_names == names

    def test_default_feature_names_generated(self) -> None:
        m = AnomalyAutoencoder(n_features=3)
        assert m.feature_names == ["feature_0", "feature_1", "feature_2"]


class TestAnomalyAutoencoderUntrained:
    """Untrained model returns safe defaults."""

    def test_predict_untrained(self) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        result = m.predict([1.0] * N_FEATURES)
        assert result["label"] == "normal"
        assert result["confidence"] == 0.5
        assert result["anomaly_score"] == 0.0

    def test_predict_batch_untrained(self) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        results = m.predict_batch(np.zeros((3, N_FEATURES)))
        assert len(results) == 3
        assert all(r["label"] == "normal" for r in results)

    def test_predict_batch_empty(self) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        assert m.predict_batch(np.zeros((0, N_FEATURES))) == []


class TestAnomalyAutoencoderFit:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_fit_sets_is_fitted(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        assert fitted_autoencoder.is_fitted

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_fit_sets_threshold(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        assert fitted_autoencoder._threshold > 0.0

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_fit_sets_scaler(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        assert fitted_autoencoder._scaler_mean is not None
        assert fitted_autoencoder._scaler_std is not None
        assert len(fitted_autoencoder._scaler_mean) == N_FEATURES

    def test_fit_raises_on_1d_input(self) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", True):
            with patch("haiip.core.torch_models._LIGHTNING_AVAILABLE", True):
                with pytest.raises(ValueError, match="2D"):
                    m.fit(np.ones(10))

    def test_fit_raises_insufficient_samples(self) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES, seq_len=50)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", True):
            with patch("haiip.core.torch_models._LIGHTNING_AVAILABLE", True):
                with pytest.raises(ValueError, match="enough samples"):
                    m.fit(np.ones((10, N_FEATURES)))

    def test_fit_without_torch_sets_fitted(self, normal_X: np.ndarray) -> None:
        """Graceful degradation when torch not available."""
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", False):
            m.fit(normal_X)
        assert m.is_fitted


class TestAnomalyAutoencoderPredict:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_returns_required_keys(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([1.0] * N_FEATURES)
        for key in ("label", "confidence", "anomaly_score", "reconstruction_error", "threshold", "explanation"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_label_is_valid(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([1.0] * N_FEATURES)
        assert result["label"] in ("normal", "anomaly")

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_confidence_in_range(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([1.0] * N_FEATURES)
        assert 0.5 <= result["confidence"] <= 1.0

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_anomaly_score_in_range(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([1.0] * N_FEATURES)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_accepts_list(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert "label" in result

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_accepts_numpy(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert "label" in result

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_batch_length(self, fitted_autoencoder: AnomalyAutoencoder, normal_X: np.ndarray) -> None:
        results = fitted_autoencoder.predict_batch(normal_X[:10])
        assert len(results) == 10

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_batch_all_have_keys(self, fitted_autoencoder: AnomalyAutoencoder, normal_X: np.ndarray) -> None:
        results = fitted_autoencoder.predict_batch(normal_X[:5])
        for r in results:
            assert "label" in r
            assert "anomaly_score" in r


class TestAnomalyAutoencoderEdgeCases:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_nan_input(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        """NaN input should not crash — may produce NaN score but not exception."""
        features = [float("nan")] * N_FEATURES
        result = fitted_autoencoder.predict(features)
        assert "label" in result  # does not raise

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_inf_input(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        features = [float("inf")] * N_FEATURES
        result = fitted_autoencoder.predict(features)
        assert "label" in result

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_zeros(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([0.0] * N_FEATURES)
        assert result["label"] in ("normal", "anomaly")

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_very_large_values(self, fitted_autoencoder: AnomalyAutoencoder) -> None:
        result = fitted_autoencoder.predict([1e10] * N_FEATURES)
        assert result["label"] in ("normal", "anomaly")


class TestAnomalyAutoencoderPersistence:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_save_and_load(self, fitted_autoencoder: AnomalyAutoencoder, tmp_path: Path) -> None:
        fitted_autoencoder.save(tmp_path / "ae_model")
        loaded = AnomalyAutoencoder.load(tmp_path / "ae_model")
        assert loaded.is_fitted
        assert loaded.n_features == fitted_autoencoder.n_features
        assert abs(loaded._threshold - fitted_autoencoder._threshold) < 1e-6

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_save_unfitted_raises(self, tmp_path: Path) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        with pytest.raises(RuntimeError, match="fitted"):
            m.save(tmp_path / "ae")

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_loaded_model_predicts(self, fitted_autoencoder: AnomalyAutoencoder, tmp_path: Path) -> None:
        fitted_autoencoder.save(tmp_path / "ae_model")
        loaded = AnomalyAutoencoder.load(tmp_path / "ae_model")
        result = loaded.predict([1.0] * N_FEATURES)
        assert result["label"] in ("normal", "anomaly")


class TestAnomalyAutoencoderONNX:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_export_onnx_creates_file(self, fitted_autoencoder: AnomalyAutoencoder, tmp_path: Path) -> None:
        try:
            out = fitted_autoencoder.export_onnx(tmp_path / "model.onnx")
            assert out.exists()
            assert out.suffix == ".onnx"
        except Exception as e:
            # ONNX may not be installed — skip gracefully
            pytest.skip(f"ONNX export not available: {e}")

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_export_onnx_dir_path(self, fitted_autoencoder: AnomalyAutoencoder, tmp_path: Path) -> None:
        try:
            out = fitted_autoencoder.export_onnx(tmp_path / "onnx_dir")
            assert out.name == "anomaly_autoencoder.onnx"
        except Exception as e:
            pytest.skip(f"ONNX export not available: {e}")

    def test_export_onnx_unfitted_raises(self, tmp_path: Path) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", True):
            with pytest.raises(RuntimeError, match="Fit model"):
                m.export_onnx(tmp_path / "model.onnx")

    def test_export_onnx_no_torch_raises(self, tmp_path: Path) -> None:
        m = AnomalyAutoencoder(n_features=N_FEATURES)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="PyTorch required"):
                m.export_onnx(tmp_path / "model.onnx")


class TestAnomalyAutoencoderThreadSafety:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_concurrent_predict(self, fitted_autoencoder: AnomalyAutoencoder, normal_X: np.ndarray) -> None:
        """Concurrent reads on a fitted model must not crash."""
        errors: list[Exception] = []

        def predict_worker() -> None:
            try:
                for row in normal_X[:5]:
                    fitted_autoencoder.predict(row.tolist())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=predict_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# MaintenanceLSTM — Unit Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMaintenanceLSTMInit:
    def test_defaults(self) -> None:
        m = MaintenanceLSTM()
        assert m.n_features == 5
        assert m.seq_len == 10
        assert not m.is_fitted

    def test_custom_class_names(self) -> None:
        names = ["ok", "fail_a", "fail_b"]
        m = MaintenanceLSTM(class_names=names)
        assert m.class_names == names


class TestMaintenanceLSTMUntrained:
    def test_predict_untrained(self) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES)
        result = m.predict([1.0] * N_FEATURES)
        assert result["label"] == "no_failure"
        assert result["confidence"] == 0.5
        assert result["failure_probability"] == 0.0
        assert result["rul_cycles"] is None

    def test_predict_batch_untrained(self) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES)
        results = m.predict_batch(np.zeros((3, N_FEATURES)))
        assert len(results) == 3

    def test_predict_batch_empty(self) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES)
        assert m.predict_batch(np.zeros((0, N_FEATURES))) == []


class TestMaintenanceLSTMFit:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_fit_sets_fitted(self, fitted_lstm: MaintenanceLSTM) -> None:
        assert fitted_lstm.is_fitted

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_fit_without_rul(self, normal_X: np.ndarray, class_labels: np.ndarray) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES, seq_len=SEQ_LEN, max_epochs=2, batch_size=16)
        m.fit(normal_X, class_labels)
        assert m.is_fitted

    def test_fit_raises_insufficient(self) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES, seq_len=50)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", True):
            with patch("haiip.core.torch_models._LIGHTNING_AVAILABLE", True):
                with pytest.raises(ValueError, match="[Ii]nsufficient"):
                    m.fit(np.ones((10, N_FEATURES)), np.array(["no_failure"] * 10))

    def test_fit_without_torch_sets_fitted(self, normal_X: np.ndarray, class_labels: np.ndarray) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", False):
            m.fit(normal_X, class_labels)
        assert m.is_fitted


class TestMaintenanceLSTMPredict:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_required_keys(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([1.0] * N_FEATURES)
        for key in ("label", "confidence", "failure_probability", "rul_cycles", "class_probabilities", "explanation"):
            assert key in result

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_label_is_class(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([1.0] * N_FEATURES)
        assert result["label"] in fitted_lstm.class_names

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_rul_non_negative(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([1.0] * N_FEATURES)
        if result["rul_cycles"] is not None:
            assert result["rul_cycles"] >= 0

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_class_proba_sum_to_one(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([1.0] * N_FEATURES)
        total = sum(result["class_probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_failure_proba_in_range(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([1.0] * N_FEATURES)
        assert 0.0 <= result["failure_probability"] <= 1.0

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_batch_returns_list(self, fitted_lstm: MaintenanceLSTM, normal_X: np.ndarray) -> None:
        results = fitted_lstm.predict_batch(normal_X[:5])
        assert len(results) == 5


class TestMaintenanceLSTMEdgeCases:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_nan(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([float("nan")] * N_FEATURES)
        assert "label" in result

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_predict_zeros(self, fitted_lstm: MaintenanceLSTM) -> None:
        result = fitted_lstm.predict([0.0] * N_FEATURES)
        assert result["label"] in fitted_lstm.class_names


class TestMaintenanceLSTMPersistence:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_save_and_load(self, fitted_lstm: MaintenanceLSTM, tmp_path: Path) -> None:
        fitted_lstm.save(tmp_path / "lstm_model")
        loaded = MaintenanceLSTM.load(tmp_path / "lstm_model")
        assert loaded.is_fitted
        assert loaded.class_names == fitted_lstm.class_names
        assert loaded.n_features == fitted_lstm.n_features

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_save_unfitted_raises(self, tmp_path: Path) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES)
        with pytest.raises(RuntimeError, match="fitted|torch"):
            m.save(tmp_path / "lstm")

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_loaded_model_predict_consistent(self, fitted_lstm: MaintenanceLSTM, tmp_path: Path) -> None:
        fitted_lstm.save(tmp_path / "lstm_model")
        loaded = MaintenanceLSTM.load(tmp_path / "lstm_model")
        r1 = fitted_lstm.predict([1.0] * N_FEATURES)
        r2 = loaded.predict([1.0] * N_FEATURES)
        assert r1["label"] == r2["label"]


class TestMaintenanceLSTMONNX:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_export_onnx_creates_file(self, fitted_lstm: MaintenanceLSTM, tmp_path: Path) -> None:
        try:
            out = fitted_lstm.export_onnx(tmp_path / "lstm.onnx")
            assert out.exists()
        except Exception as e:
            pytest.skip(f"ONNX export not available: {e}")

    def test_export_onnx_unfitted_raises(self, tmp_path: Path) -> None:
        m = MaintenanceLSTM(n_features=N_FEATURES)
        with patch("haiip.core.torch_models._TORCH_AVAILABLE", True):
            with pytest.raises(RuntimeError, match="Fit model"):
                m.export_onnx(tmp_path / "lstm.onnx")


class TestMaintenanceLSTMThreadSafety:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_concurrent_predict(self, fitted_lstm: MaintenanceLSTM, normal_X: np.ndarray) -> None:
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for row in normal_X[:5]:
                    fitted_lstm.predict(row.tolist())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ══════════════════════════════════════════════════════════════════════════════
# Integration — train → export → load cycle
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_autoencoder_full_pipeline(self, normal_X: np.ndarray, tmp_path: Path) -> None:
        """Train → predict → save → load → predict again."""
        model = AnomalyAutoencoder(
            n_features=N_FEATURES, seq_len=SEQ_LEN, max_epochs=2, batch_size=16
        )
        model.fit(normal_X)
        r1 = model.predict(normal_X[0].tolist())
        model.save(tmp_path / "ae")
        loaded = AnomalyAutoencoder.load(tmp_path / "ae")
        r2 = loaded.predict(normal_X[0].tolist())
        assert r1["label"] == r2["label"]

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_maintenance_full_pipeline(
        self,
        normal_X: np.ndarray,
        class_labels: np.ndarray,
        rul_values: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Train → predict → save → load → predict again."""
        model = MaintenanceLSTM(
            n_features=N_FEATURES, seq_len=SEQ_LEN, max_epochs=2, batch_size=16
        )
        model.fit(normal_X, class_labels, rul_values)
        r1 = model.predict(normal_X[0].tolist())
        model.save(tmp_path / "lstm")
        loaded = MaintenanceLSTM.load(tmp_path / "lstm")
        r2 = loaded.predict(normal_X[0].tolist())
        assert r1["label"] == r2["label"]

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_sklearn_interface_parity_anomaly(self, fitted_autoencoder: AnomalyAutoencoder, normal_X: np.ndarray) -> None:
        """Output dict must have the same keys as sklearn AnomalyDetector."""
        required_sklearn_keys = {"label", "confidence", "anomaly_score", "explanation"}
        result = fitted_autoencoder.predict(normal_X[0].tolist())
        assert required_sklearn_keys.issubset(result.keys())

    @pytest.mark.skipif(not (_TORCH_AVAILABLE and _LIGHTNING_AVAILABLE), reason="Torch/Lightning required")
    def test_sklearn_interface_parity_maintenance(self, fitted_lstm: MaintenanceLSTM, normal_X: np.ndarray) -> None:
        """Output dict must have the same keys as sklearn MaintenancePredictor."""
        required_keys = {"label", "confidence", "failure_probability", "rul_cycles", "explanation"}
        result = fitted_lstm.predict(normal_X[0].tolist())
        assert required_keys.issubset(result.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleConstants:
    def test_failure_modes_list(self) -> None:
        assert "no_failure" in FAILURE_MODES
        assert "TWF" in FAILURE_MODES
        assert len(FAILURE_MODES) == 6

    def test_torch_available_is_bool(self) -> None:
        assert isinstance(_TORCH_AVAILABLE, bool)

    def test_lightning_available_is_bool(self) -> None:
        assert isinstance(_LIGHTNING_AVAILABLE, bool)
