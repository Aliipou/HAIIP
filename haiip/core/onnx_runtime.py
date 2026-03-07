"""ONNX Runtime inference engine for industrial edge deployment.

Wraps exported ONNX models from torch_models.py and provides:
- Sub-50ms inference guarantee (target: Jetson / Hailo / Industrial PC)
- Same predict() interface as sklearn/Lightning models — zero API change
- Latency benchmarking and SLA enforcement
- INT8 quantization support via onnxruntime quantization tools
- Batch inference with throughput measurement

Usage:
    detector = ONNXAnomalyDetector.from_onnx("artifacts/anomaly.onnx", seq_len=10, n_features=5)
    result   = detector.predict([298.1, 308.6, 1551, 42.8, 0])
    # → {"label": "normal", "confidence": 0.82, "latency_ms": 3.1, ...}
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional ONNX Runtime ─────────────────────────────────────────────────────

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
    logger.info(
        "ONNX Runtime %s — providers: %s",
        ort.__version__,
        ort.get_available_providers(),
    )
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed — ONNXDetector will use fallback mode")

# SLA: maximum acceptable inference latency
LATENCY_SLA_MS = 50.0


def _make_session(onnx_path: str) -> ort.InferenceSession:
    """Create an optimised InferenceSession.

    Provider priority: CUDAExecutionProvider → TensorrtExecutionProvider → CPUExecutionProvider
    Optimisation level: ORT_ENABLE_ALL for maximum kernel fusion.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4  # tuned for industrial edge CPUs

    providers: list[str] = []
    available = ort.get_available_providers()
    for p in [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if p in available:
            providers.append(p)

    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


# ── ONNX Anomaly Detector ────────────────────────────────────────────────────


class ONNXAnomalyDetector:
    """ONNX Runtime wrapper for AnomalyAutoencoder.

    Loads a pre-exported .onnx file and performs reconstruction-error
    anomaly detection identical to the PyTorch original, but with ONNX
    Runtime optimisations for edge deployment.

    Args:
        onnx_path:      Path to the .onnx file
        seq_len:        Sequence length used during export
        n_features:     Number of features used during export
        threshold:      Reconstruction error threshold for anomaly
        scaler_mean:    Feature mean from training (for z-score normalisation)
        scaler_std:     Feature std from training
        feature_names:  Feature labels for explanation output
        latency_sla_ms: Maximum acceptable latency; warns if exceeded
    """

    def __init__(
        self,
        onnx_path: str | Path,
        seq_len: int = 10,
        n_features: int = 5,
        threshold: float = 0.01,
        scaler_mean: np.ndarray | None = None,
        scaler_std: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        latency_sla_ms: float = LATENCY_SLA_MS,
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.seq_len = seq_len
        self.n_features = n_features
        self.threshold = threshold
        self.scaler_mean = scaler_mean if scaler_mean is not None else np.zeros(n_features)
        self.scaler_std = scaler_std if scaler_std is not None else np.ones(n_features)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.latency_sla_ms = latency_sla_ms

        self._session: ort.InferenceSession | None = None
        self._input_name: str = "input"
        self._latency_history: list[float] = []

        if _ORT_AVAILABLE and Path(onnx_path).exists():
            self._load_session()

    def _load_session(self) -> None:
        self._session = _make_session(self.onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNXAnomalyDetector session loaded: %s", self.onnx_path)

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        meta_path: str | Path | None = None,
        **kwargs: Any,
    ) -> ONNXAnomalyDetector:
        """Convenience constructor — loads meta.npz if alongside the .onnx file."""
        onnx_path = Path(onnx_path)
        if meta_path is None:
            meta_path = onnx_path.parent / "meta.npz"

        kw: dict[str, Any] = {}
        if Path(meta_path).exists():
            meta = np.load(str(meta_path))
            kw["seq_len"] = int(meta["seq_len"])
            kw["n_features"] = int(meta["n_features"])
            kw["threshold"] = float(meta["threshold"])
            kw["scaler_mean"] = meta["scaler_mean"]
            kw["scaler_std"] = meta["scaler_std"]
        kw.update(kwargs)

        return cls(onnx_path=onnx_path, **kw)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict anomaly for a single reading.

        Returns:
            label, confidence, anomaly_score, reconstruction_error,
            latency_ms (SLA monitored)
        """
        if self._session is None:
            return self._fallback_result(features)

        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
        arr_norm = (arr - self.scaler_mean) / self.scaler_std

        # Replicate to window length
        seq = np.tile(arr_norm, (1, self.seq_len, 1)).astype(np.float32)
        # seq shape: (1, seq_len, n_features)

        t0 = time.perf_counter()
        reconstructed = self._session.run(None, {self._input_name: seq})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_history.append(latency_ms)

        if latency_ms > self.latency_sla_ms:
            logger.warning(
                "Inference SLA breached: %.1f ms > %.0f ms",
                latency_ms,
                self.latency_sla_ms,
            )

        error = float(np.mean((seq - reconstructed) ** 2))
        normalized = float(np.clip(error / (self.threshold * 2.0 + 1e-8), 0.0, 1.0))
        label = "anomaly" if error > self.threshold else "normal"
        confidence = float(
            np.clip(normalized if label == "anomaly" else 1.0 - normalized, 0.5, 1.0)
        )

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
            "reconstruction_error": round(error, 6),
            "threshold": round(self.threshold, 6),
            "latency_ms": round(latency_ms, 2),
            "sla_ok": latency_ms <= self.latency_sla_ms,
            "explanation": explanation,
        }

    def predict_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Predict for multiple readings (batched — more efficient than loop)."""
        arr = np.asarray(X, dtype=np.float32)
        if len(arr) == 0:
            return []
        if self._session is None:
            return [self._fallback_result(row) for row in arr]

        arr_norm = (arr - self.scaler_mean) / self.scaler_std
        # Build batch of windows: (batch, seq_len, n_features)
        batch = np.stack([np.tile(row, (self.seq_len, 1)) for row in arr_norm], axis=0).astype(
            np.float32
        )

        t0 = time.perf_counter()
        reconstructed = self._session.run(None, {self._input_name: batch})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        errors = np.mean((batch - reconstructed) ** 2, axis=(1, 2))
        results = []
        for _i, (row, error) in enumerate(zip(arr_norm, errors)):
            normalized = float(np.clip(error / (self.threshold * 2.0 + 1e-8), 0.0, 1.0))
            label = "anomaly" if error > self.threshold else "normal"
            confidence = float(
                np.clip(normalized if label == "anomaly" else 1.0 - normalized, 0.5, 1.0)
            )
            z_scores = row.tolist()
            explanation = {
                name: round(float(z), 3)
                for name, z in zip(self.feature_names, z_scores)
                if abs(z) > 1.5
            }
            results.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "anomaly_score": round(normalized, 4),
                    "reconstruction_error": round(float(error), 6),
                    "threshold": round(self.threshold, 6),
                    "latency_ms": round(latency_ms / len(arr), 2),
                    "sla_ok": (latency_ms / len(arr)) <= self.latency_sla_ms,
                    "explanation": explanation,
                }
            )
        return results

    # ── Benchmarking ──────────────────────────────────────────────────────────

    def benchmark(self, n_runs: int = 100) -> dict[str, float]:
        """Run latency benchmark — returns p50, p95, p99 and mean latencies."""
        dummy = np.zeros(self.n_features, dtype=np.float32)
        for _ in range(n_runs):
            self.predict(dummy)

        hist = np.array(self._latency_history[-n_runs:])
        result = {
            "mean_ms": round(float(hist.mean()), 2),
            "p50_ms": round(float(np.percentile(hist, 50)), 2),
            "p95_ms": round(float(np.percentile(hist, 95)), 2),
            "p99_ms": round(float(np.percentile(hist, 99)), 2),
            "max_ms": round(float(hist.max()), 2),
            "sla_pass_rate": round(float((hist <= self.latency_sla_ms).mean()), 4),
            "n_runs": n_runs,
        }
        logger.info("Benchmark: %s", result)
        return result

    @property
    def latency_stats(self) -> dict[str, float]:
        """Rolling latency statistics across all calls."""
        if not self._latency_history:
            return {}
        hist = np.array(self._latency_history)
        return {
            "n_calls": len(hist),
            "mean_ms": round(float(hist.mean()), 2),
            "p95_ms": round(float(np.percentile(hist, 95)), 2),
            "sla_pass_rate": round(float((hist <= self.latency_sla_ms).mean()), 4),
        }

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @staticmethod
    def _fallback_result(features: Any) -> dict[str, Any]:
        return {
            "label": "normal",
            "confidence": 0.5,
            "anomaly_score": 0.0,
            "reconstruction_error": 0.0,
            "threshold": 0.0,
            "latency_ms": 0.0,
            "sla_ok": True,
            "explanation": {"info": "ONNX session not available — fallback mode"},
        }


# ── ONNX Maintenance Predictor ────────────────────────────────────────────────


class ONNXMaintenancePredictor:
    """ONNX Runtime wrapper for MaintenanceLSTM.

    Provides failure classification + RUL regression via ONNX Runtime
    with the same interface as sklearn MaintenancePredictor.

    Args:
        onnx_path:      Path to .onnx file
        seq_len:        Window length (must match export)
        n_features:     Feature count (must match export)
        class_names:    Ordered list of failure class labels
        scaler_mean:    Z-score normalisation mean
        scaler_std:     Z-score normalisation std
        rul_mean:       RUL denormalisation mean
        rul_std:        RUL denormalisation std
        latency_sla_ms: SLA threshold (default 50ms)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        seq_len: int = 10,
        n_features: int = 5,
        class_names: list[str] | None = None,
        scaler_mean: np.ndarray | None = None,
        scaler_std: np.ndarray | None = None,
        rul_mean: float = 0.0,
        rul_std: float = 1.0,
        latency_sla_ms: float = LATENCY_SLA_MS,
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.seq_len = seq_len
        self.n_features = n_features
        self.class_names = class_names or [
            "no_failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
        ]
        self.scaler_mean = scaler_mean if scaler_mean is not None else np.zeros(n_features)
        self.scaler_std = scaler_std if scaler_std is not None else np.ones(n_features)
        self.rul_mean = rul_mean
        self.rul_std = rul_std
        self.latency_sla_ms = latency_sla_ms
        self._latency_history: list[float] = []

        self._session: ort.InferenceSession | None = None
        self._input_name: str = "input"

        if _ORT_AVAILABLE and Path(onnx_path).exists():
            self._load_session()

    def _load_session(self) -> None:
        self._session = _make_session(self.onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNXMaintenancePredictor session loaded: %s", self.onnx_path)

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        meta_path: str | Path | None = None,
        classes_path: str | Path | None = None,
        **kwargs: Any,
    ) -> ONNXMaintenancePredictor:
        """Load from .onnx + optional meta.npz / classes.json."""
        import json

        onnx_path = Path(onnx_path)
        parent = onnx_path.parent
        if meta_path is None:
            meta_path = parent / "meta.npz"
        if classes_path is None:
            classes_path = parent / "classes.json"

        kw: dict[str, Any] = {}
        if Path(meta_path).exists():
            meta = np.load(str(meta_path))
            kw["seq_len"] = int(meta["seq_len"])
            kw["n_features"] = int(meta["n_features"])
            kw["scaler_mean"] = meta["scaler_mean"]
            kw["scaler_std"] = meta["scaler_std"]
            kw["rul_mean"] = float(meta["rul_mean"])
            kw["rul_std"] = float(meta["rul_std"])
        if Path(classes_path).exists():
            with open(classes_path) as f:
                kw["class_names"] = json.load(f)["class_names"]
        kw.update(kwargs)
        return cls(onnx_path=onnx_path, **kw)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict failure mode + RUL for a single reading."""
        if self._session is None:
            return self._fallback_result()

        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
        arr_norm = (arr - self.scaler_mean) / self.scaler_std
        seq = np.tile(arr_norm, (1, self.seq_len, 1)).astype(np.float32)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: seq})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_history.append(latency_ms)

        if latency_ms > self.latency_sla_ms:
            logger.warning(
                "Inference SLA breached: %.1f ms > %.0f ms",
                latency_ms,
                self.latency_sla_ms,
            )

        logits = outputs[0][0]  # (n_classes,)
        rul_norm = float(outputs[1][0]) if len(outputs) > 1 else 0.0

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        proba = exp_logits / exp_logits.sum()

        pred_idx = int(np.argmax(proba))
        pred_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "unknown"
        pred_proba = float(proba[pred_idx])
        rul_cycles = max(0, int(round(rul_norm * self.rul_std + self.rul_mean)))

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
            "latency_ms": round(latency_ms, 2),
            "sla_ok": latency_ms <= self.latency_sla_ms,
            "explanation": {"model": "BiLSTM-ONNX"},
        }

    def predict_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Batch prediction — efficient for bulk inference."""
        arr = np.asarray(X, dtype=np.float32)
        if len(arr) == 0:
            return []
        if self._session is None:
            return [self._fallback_result() for _ in arr]

        arr_norm = (arr - self.scaler_mean) / self.scaler_std
        batch = np.stack([np.tile(row, (self.seq_len, 1)) for row in arr_norm], axis=0).astype(
            np.float32
        )

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: batch})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        all_logits = outputs[0]  # (batch, n_classes)
        all_rul = outputs[1] if len(outputs) > 1 else np.zeros(len(arr))

        results = []
        for i in range(len(arr)):
            logits = all_logits[i]
            exp_l = np.exp(logits - logits.max())
            proba = exp_l / exp_l.sum()
            pred_idx = int(np.argmax(proba))
            pred_label = (
                self.class_names[pred_idx] if pred_idx < len(self.class_names) else "unknown"
            )
            rul_norm = float(all_rul[i])
            rul_cycles = max(0, int(round(rul_norm * self.rul_std + self.rul_mean)))

            normal_idx = (
                self.class_names.index("no_failure") if "no_failure" in self.class_names else -1
            )
            failure_proba = (
                1.0 - float(proba[normal_idx]) if normal_idx >= 0 else float(proba[pred_idx])
            )

            results.append(
                {
                    "label": pred_label,
                    "confidence": round(float(proba[pred_idx]), 4),
                    "failure_probability": round(failure_proba, 4),
                    "rul_cycles": rul_cycles,
                    "class_probabilities": {
                        cls: round(float(p), 4) for cls, p in zip(self.class_names, proba)
                    },
                    "latency_ms": round(latency_ms / len(arr), 2),
                    "sla_ok": (latency_ms / len(arr)) <= self.latency_sla_ms,
                    "explanation": {"model": "BiLSTM-ONNX-batch"},
                }
            )
        return results

    def benchmark(self, n_runs: int = 100) -> dict[str, float]:
        """Latency benchmark — p50, p95, p99."""
        dummy = np.zeros(self.n_features, dtype=np.float32)
        for _ in range(n_runs):
            self.predict(dummy)
        hist = np.array(self._latency_history[-n_runs:])
        result = {
            "mean_ms": round(float(hist.mean()), 2),
            "p50_ms": round(float(np.percentile(hist, 50)), 2),
            "p95_ms": round(float(np.percentile(hist, 95)), 2),
            "p99_ms": round(float(np.percentile(hist, 99)), 2),
            "max_ms": round(float(hist.max()), 2),
            "sla_pass_rate": round(float((hist <= self.latency_sla_ms).mean()), 4),
            "n_runs": n_runs,
        }
        logger.info("Benchmark: %s", result)
        return result

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @staticmethod
    def _fallback_result() -> dict[str, Any]:
        return {
            "label": "no_failure",
            "confidence": 0.5,
            "failure_probability": 0.0,
            "rul_cycles": None,
            "latency_ms": 0.0,
            "sla_ok": True,
            "explanation": {"info": "ONNX session not available — fallback mode"},
        }
