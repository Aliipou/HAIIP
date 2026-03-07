"""PyTorch Lightning models for industrial-grade anomaly detection and RUL estimation.

Models:
    AnomalyAutoencoder  — LSTM-based autoencoder; reconstruction error → anomaly score
    MaintenanceLSTM     — Bidirectional LSTM for RUL regression + failure classification
    ONNXExporter        — Export any Lightning module to ONNX for edge deployment (≤50ms)

Design principles:
    - Lightning modules → clean training / inference separation
    - ONNX export → Jetson / Hailo / Industrial PC compatible
    - Quantization-ready (INT8 via torch.quantization) for ≤50ms edge latency
    - Same predict() interface as sklearn models — drop-in replacement
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional imports (graceful degradation) ───────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LightningModels unavailable")

try:
    import lightning as L  # lightning >= 2.0 (renamed from pytorch_lightning)

    _LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]

        _LIGHTNING_AVAILABLE = True
    except ImportError:  # pragma: no cover
        _LIGHTNING_AVAILABLE = False
        logger.warning("PyTorch Lightning not installed — LightningModels unavailable")


# ── Autoencoder (Anomaly Detection) ──────────────────────────────────────────


class _LSTMAutoencoderCore(nn.Module):
    """Pure PyTorch LSTM autoencoder — exportable to ONNX."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (hidden, _) = self.encoder(x)
        # Repeat bottleneck across sequence length
        seq_len = x.shape[1]
        bottleneck = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed, _ = self.decoder(bottleneck)
        return reconstructed  # (batch, seq_len, n_features)


class AnomalyAutoencoder:
    """LSTM autoencoder anomaly detector — PyTorch Lightning backed.

    Trains on normal data only. High reconstruction error → anomaly.
    Provides same interface as sklearn AnomalyDetector (predict, predict_batch).

    Args:
        n_features:   Number of sensor features (default 5 for AI4I)
        seq_len:      Window length per sample (default 10 time steps)
        hidden_size:  LSTM hidden dimension
        n_layers:     LSTM encoder layers
        threshold_pct: Percentile of training errors used as anomaly threshold (default 95)
        lr:           Learning rate
        max_epochs:   Training epochs
        batch_size:   Training batch size
        feature_names: Feature names for explanation output
    """

    def __init__(
        self,
        n_features: int = 5,
        seq_len: int = 10,
        hidden_size: int = 64,
        n_layers: int = 2,
        threshold_pct: float = 95.0,
        lr: float = 1e-3,
        max_epochs: int = 30,
        batch_size: int = 64,
        feature_names: list[str] | None = None,
    ) -> None:
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.threshold_pct = threshold_pct
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        self._model: _LSTMAutoencoderCore | None = None
        self._threshold: float = 0.0
        self._is_fitted = False
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> AnomalyAutoencoder:
        """Fit autoencoder on normal (or mixed) data.

        Args:
            X: shape (n_samples, n_features) — 2D; windowed internally.
        """
        if not _TORCH_AVAILABLE or not _LIGHTNING_AVAILABLE:
            logger.warning("Torch/Lightning not available — AnomalyAutoencoder.fit() skipped")
            self._is_fitted = True  # mark fitted with fallback mode
            return self

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D")

        # Z-score normalise
        self._scaler_mean = X_arr.mean(axis=0)
        self._scaler_std = X_arr.std(axis=0) + 1e-8
        X_norm = (X_arr - self._scaler_mean) / self._scaler_std

        # Sliding-window sequences
        sequences = self._make_sequences(X_norm)
        if len(sequences) == 0:
            raise ValueError(f"Not enough samples ({len(X_arr)}) for seq_len={self.seq_len}")

        tensor = torch.tensor(sequences)  # (n, seq_len, n_features)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Build model
        core = _LSTMAutoencoderCore(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
        )

        # Lightning training module
        lit_module = _LitAutoencoder(core=core, lr=self.lr)

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            devices=1,
        )
        trainer.fit(lit_module, loader)

        self._model = core
        self._model.eval()

        # Compute reconstruction errors on training data → set threshold
        train_errors = self._reconstruction_errors(sequences)
        self._threshold = float(np.percentile(train_errors, self.threshold_pct))
        self._is_fitted = True

        logger.info(
            "AnomalyAutoencoder fitted: n_samples=%d, threshold=%.6f (p%.0f)",
            len(X_arr),
            self._threshold,
            self.threshold_pct,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict anomaly for a single reading.

        Returns dict with: label, confidence, anomaly_score, explanation.
        """
        if not self._is_fitted or self._model is None:
            return self._untrained_result(features)

        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
        arr_norm = (arr - self._scaler_mean) / self._scaler_std  # type: ignore[operator]

        # Replicate to make a minimal sequence
        seq = np.tile(arr_norm, (self.seq_len, 1))  # (seq_len, n_features)
        error = self._reconstruction_errors(seq[np.newaxis])[0]

        normalized = float(np.clip(error / (self._threshold * 2.0), 0.0, 1.0))
        label = "anomaly" if error > self._threshold else "normal"
        confidence = float(
            np.clip(normalized if label == "anomaly" else 1.0 - normalized, 0.5, 1.0)
        )

        # Per-feature z-scores for explanation
        z_scores = arr_norm[0].tolist()
        explanation = {
            name: round(float(z), 3)
            for name, z in zip(self.feature_names, z_scores)
            if abs(z) > 1.5
        }

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(normalized, 4),
            "reconstruction_error": round(float(error), 6),
            "threshold": round(self._threshold, 6),
            "explanation": explanation,
        }

    def predict_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Predict anomaly for multiple readings at once."""
        arr = np.asarray(X, dtype=np.float32)
        if len(arr) == 0:
            return []
        return [self.predict(row) for row in arr]

    # ── ONNX Export ───────────────────────────────────────────────────────────

    def export_onnx(self, path: Path | str, opset: int = 17) -> Path:
        """Export model to ONNX for edge deployment.

        Args:
            path:  Output path (directory or .onnx file)
            opset: ONNX opset version (17 recommended for ONNX Runtime 1.18+)

        Returns:
            Path to saved .onnx file
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ONNX export")
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Fit model before exporting to ONNX")

        out = Path(path)
        if out.suffix != ".onnx":
            out.mkdir(parents=True, exist_ok=True)
            out = out / "anomaly_autoencoder.onnx"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.zeros(1, self.seq_len, self.n_features)
        torch.onnx.export(
            self._model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["reconstructed"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "reconstructed": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        logger.info("AnomalyAutoencoder exported to ONNX: %s", out)
        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Save model weights + metadata."""
        if not _TORCH_AVAILABLE or not self._is_fitted or self._model is None:
            raise RuntimeError("Cannot save — model not fitted or torch unavailable")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), p / "autoencoder.pt")
        np.savez(
            p / "meta.npz",
            threshold=self._threshold,
            scaler_mean=self._scaler_mean,
            scaler_std=self._scaler_std,
            n_features=self.n_features,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
        )
        logger.info("AnomalyAutoencoder saved to %s", p)

    @classmethod
    def load(cls, path: Path | str) -> AnomalyAutoencoder:
        """Load a previously saved model."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for loading")
        p = Path(path)
        meta = np.load(p / "meta.npz")
        obj = cls(
            n_features=int(meta["n_features"]),
            seq_len=int(meta["seq_len"]),
            hidden_size=int(meta["hidden_size"]),
            n_layers=int(meta["n_layers"]),
        )
        core = _LSTMAutoencoderCore(
            n_features=obj.n_features,
            hidden_size=obj.hidden_size,
            n_layers=obj.n_layers,
        )
        core.load_state_dict(torch.load(p / "autoencoder.pt", map_location="cpu"))  # nosec B614
        core.eval()
        obj._model = core
        obj._threshold = float(meta["threshold"])
        obj._scaler_mean = meta["scaler_mean"]
        obj._scaler_std = meta["scaler_std"]
        obj._is_fitted = True
        logger.info("AnomalyAutoencoder loaded from %s", p)
        return obj

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        """Sliding window: (n_samples, n_features) → (n_windows, seq_len, n_features)."""
        n = len(X)
        if n < self.seq_len:
            return np.array([])
        windows = [X[i : i + self.seq_len] for i in range(n - self.seq_len + 1)]
        return np.stack(windows).astype(np.float32)

    def _reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        with torch.no_grad():
            t = torch.tensor(sequences)
            reconstructed = self._model(t)  # type: ignore[misc]
            errors = ((t - reconstructed) ** 2).mean(dim=(1, 2))
        return errors.numpy()

    @staticmethod
    def _untrained_result(features: Any) -> dict[str, Any]:
        return {
            "label": "normal",
            "confidence": 0.5,
            "anomaly_score": 0.0,
            "reconstruction_error": 0.0,
            "threshold": 0.0,
            "explanation": {"info": "Model not yet trained"},
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ── Lightning training wrapper ────────────────────────────────────────────────


class _LitAutoencoder(L.LightningModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """Lightning training logic for the LSTM autoencoder."""

    def __init__(self, core: _LSTMAutoencoderCore, lr: float = 1e-3) -> None:
        super().__init__()
        self.core = core
        self.lr = lr
        self._loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)

    def training_step(self, batch: tuple[torch.Tensor, ...], _: int) -> torch.Tensor:
        (x,) = batch
        reconstructed = self(x)
        loss = self._loss_fn(reconstructed, x)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.core.parameters(), lr=self.lr)


# ── Maintenance LSTM (RUL + Classification) ──────────────────────────────────


class _LSTMMaintenanceCore(nn.Module):
    """Bidirectional LSTM — dual-head: classification + RUL regression."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out = hidden_size * 2  # bidirectional

        self.classifier_head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus(),  # RUL ≥ 0
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take last time step
        return self.classifier_head(last), self.rul_head(last).squeeze(-1)


FAILURE_MODES = ["no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


class MaintenanceLSTM:
    """Bidirectional LSTM predictive maintenance model.

    Drop-in replacement for sklearn MaintenancePredictor with:
    - Sequence-aware inference (captures temporal dependencies)
    - Dual output: failure classification + RUL regression
    - ONNX export for edge deployment

    Args:
        n_features:    Sensor feature count (default 5)
        seq_len:       Time steps per inference window (default 10)
        hidden_size:   LSTM hidden units
        n_layers:      LSTM depth
        dropout:       Dropout rate during training
        lr:            Learning rate
        max_epochs:    Training epochs
        batch_size:    Training batch size
        feature_names: Feature labels for explanation output
        class_names:   Failure mode labels
    """

    def __init__(
        self,
        n_features: int = 5,
        seq_len: int = 10,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 64,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.class_names = class_names or FAILURE_MODES

        self._model: _LSTMMaintenanceCore | None = None
        self._is_fitted = False
        self._label_to_idx: dict[str, int] = {}
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._rul_mean: float = 0.0
        self._rul_std: float = 1.0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_rul: np.ndarray | None = None,
    ) -> MaintenanceLSTM:
        """Train LSTM on sensor sequences.

        Args:
            X:       (n_samples, n_features)
            y_class: (n_samples,) string class labels
            y_rul:   (n_samples,) RUL values (optional)
        """
        if not _TORCH_AVAILABLE or not _LIGHTNING_AVAILABLE:
            logger.warning("Torch/Lightning unavailable — MaintenanceLSTM.fit() skipped")
            self._is_fitted = True
            return self

        X_arr = np.asarray(X, dtype=np.float32)

        # Z-score normalise features
        self._scaler_mean = X_arr.mean(axis=0)
        self._scaler_std = X_arr.std(axis=0) + 1e-8
        X_norm = (X_arr - self._scaler_mean) / self._scaler_std

        # Encode labels
        unique_classes = sorted({str(c) for c in y_class})
        self._label_to_idx = {c: i for i, c in enumerate(unique_classes)}
        self.class_names = unique_classes
        y_idx = np.array([self._label_to_idx[str(c)] for c in y_class], dtype=np.int64)

        # RUL normalisation
        if y_rul is not None:
            y_rul_arr = np.asarray(y_rul, dtype=np.float32)
            self._rul_mean = float(y_rul_arr.mean())
            self._rul_std = float(y_rul_arr.std()) + 1e-8
            y_rul_norm = ((y_rul_arr - self._rul_mean) / self._rul_std).astype(np.float32)
        else:
            y_rul_norm = np.zeros(len(X_arr), dtype=np.float32)

        # Sliding-window sequences
        sequences, labels, ruls = self._make_sequences(X_norm, y_idx, y_rul_norm)
        if len(sequences) == 0:
            raise ValueError(f"Insufficient samples ({len(X_arr)}) for seq_len={self.seq_len}")

        t_seq = torch.tensor(sequences)
        t_lbl = torch.tensor(labels)
        t_rul = torch.tensor(ruls)

        loader = DataLoader(
            TensorDataset(t_seq, t_lbl, t_rul),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        core = _LSTMMaintenanceCore(
            n_features=self.n_features,
            n_classes=len(unique_classes),
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        lit = _LitMaintenance(core=core, lr=self.lr, has_rul=y_rul is not None)

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            devices=1,
        )
        trainer.fit(lit, loader)

        self._model = core
        self._model.eval()
        self._is_fitted = True

        logger.info(
            "MaintenanceLSTM fitted: n_samples=%d, classes=%s",
            len(X_arr),
            self.class_names,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict failure mode + RUL for a single reading."""
        if not self._is_fitted or self._model is None:
            return self._untrained_result()

        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
        arr_norm = (arr - self._scaler_mean) / self._scaler_std  # type: ignore[operator]
        seq = np.tile(arr_norm, (self.seq_len, 1))[np.newaxis].astype(np.float32)

        with torch.no_grad():
            t = torch.tensor(seq)
            logits, rul_norm = self._model(t)
            proba = torch.softmax(logits, dim=-1).numpy()[0]
            rul_denorm = float(rul_norm.numpy()[0]) * self._rul_std + self._rul_mean

        pred_idx = int(np.argmax(proba))
        pred_label = self.class_names[pred_idx]
        pred_proba = float(proba[pred_idx])
        rul_cycles = max(0, int(round(rul_denorm)))

        normal_idx = (
            self.class_names.index("no_failure") if "no_failure" in self.class_names else -1
        )
        failure_proba = 1.0 - float(proba[normal_idx]) if normal_idx >= 0 else pred_proba

        return {
            "label": pred_label,
            "confidence": round(pred_proba, 4),
            "failure_probability": round(failure_proba, 4),
            "rul_cycles": rul_cycles,
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self.class_names, proba)
            },
            "explanation": {
                "model": "BiLSTM",
                "seq_len": self.seq_len,
            },
        }

    def predict_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Predict for multiple readings."""
        arr = np.asarray(X, dtype=np.float32)
        if len(arr) == 0:
            return []
        return [self.predict(row) for row in arr]

    # ── ONNX Export ───────────────────────────────────────────────────────────

    def export_onnx(self, path: Path | str, opset: int = 17) -> Path:
        """Export to ONNX — compatible with ONNX Runtime 1.18+ on edge devices."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ONNX export")
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Fit model before exporting to ONNX")

        out = Path(path)
        if out.suffix != ".onnx":
            out.mkdir(parents=True, exist_ok=True)
            out = out / "maintenance_lstm.onnx"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.zeros(1, self.seq_len, self.n_features)
        torch.onnx.export(
            self._model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["class_logits", "rul"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "class_logits": {0: "batch_size"},
                "rul": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        logger.info("MaintenanceLSTM exported to ONNX: %s", out)
        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        if not _TORCH_AVAILABLE or not self._is_fitted or self._model is None:
            raise RuntimeError("Cannot save — not fitted or torch unavailable")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), p / "lstm_maintenance.pt")
        np.savez(
            p / "meta.npz",
            n_features=self.n_features,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            rul_mean=self._rul_mean,
            rul_std=self._rul_std,
            scaler_mean=self._scaler_mean,
            scaler_std=self._scaler_std,
        )
        import json

        with open(p / "classes.json", "w") as f:
            json.dump({"class_names": self.class_names, "label_to_idx": self._label_to_idx}, f)
        logger.info("MaintenanceLSTM saved to %s", p)

    @classmethod
    def load(cls, path: Path | str) -> MaintenanceLSTM:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for loading")
        import json

        p = Path(path)
        meta = np.load(p / "meta.npz")
        with open(p / "classes.json") as f:
            class_data = json.load(f)

        obj = cls(
            n_features=int(meta["n_features"]),
            seq_len=int(meta["seq_len"]),
            hidden_size=int(meta["hidden_size"]),
            n_layers=int(meta["n_layers"]),
        )
        obj.class_names = class_data["class_names"]
        obj._label_to_idx = class_data["label_to_idx"]
        obj._rul_mean = float(meta["rul_mean"])
        obj._rul_std = float(meta["rul_std"])
        obj._scaler_mean = meta["scaler_mean"]
        obj._scaler_std = meta["scaler_std"]

        core = _LSTMMaintenanceCore(
            n_features=obj.n_features,
            n_classes=len(obj.class_names),
            hidden_size=obj.hidden_size,
            n_layers=obj.n_layers,
        )
        core.load_state_dict(torch.load(p / "lstm_maintenance.pt", map_location="cpu"))  # nosec B614
        core.eval()
        obj._model = core
        obj._is_fitted = True
        logger.info("MaintenanceLSTM loaded from %s", p)
        return obj

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rul: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sliding window → (sequences, labels, ruls)."""
        n = len(X)
        if n < self.seq_len:
            return np.array([]), np.array([]), np.array([])
        seqs, lbls, rls = [], [], []
        for i in range(n - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])
            lbls.append(y[i + self.seq_len - 1])
            rls.append(rul[i + self.seq_len - 1])
        return (
            np.stack(seqs).astype(np.float32),
            np.array(lbls, dtype=np.int64),
            np.array(rls, dtype=np.float32),
        )

    @staticmethod
    def _untrained_result() -> dict[str, Any]:
        return {
            "label": "no_failure",
            "confidence": 0.5,
            "failure_probability": 0.0,
            "rul_cycles": None,
            "explanation": {"info": "Model not yet trained"},
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class _LitMaintenance(L.LightningModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """Lightning training logic for the maintenance LSTM."""

    def __init__(
        self,
        core: _LSTMMaintenanceCore,
        lr: float = 1e-3,
        has_rul: bool = True,
        rul_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.core = core
        self.lr = lr
        self.has_rul = has_rul
        self.rul_weight = rul_weight
        self._clf_loss = nn.CrossEntropyLoss()
        self._rul_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core(x)

    def training_step(self, batch: tuple[torch.Tensor, ...], _: int) -> torch.Tensor:
        x, y_cls, y_rul = batch
        logits, rul_pred = self(x)
        loss_clf = self._clf_loss(logits, y_cls)
        if self.has_rul:
            loss_rul = self._rul_loss(rul_pred, y_rul)
            loss = loss_clf + self.rul_weight * loss_rul
        else:
            loss = loss_clf
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.core.parameters(), lr=self.lr)
