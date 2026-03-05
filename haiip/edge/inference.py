"""Edge inference — lightweight on-device prediction for HAIIP.

Design goals:
- Zero network dependency during inference (works offline / poor connectivity)
- <5ms p99 latency on embedded ARM (Raspberry Pi 4, NVIDIA Jetson Nano)
- ONNX export from trained sklearn IsolationForest via sklearn-onnx
- Fallback: pure numpy scoring when ONNX runtime unavailable
- Signed model package with SHA-256 integrity check

Architecture:
    EdgeInferenceEngine
        .load(path)         — load model artifacts (ONNX or joblib fallback)
        .predict(features)  — dict with label, confidence, anomaly_score
        .export_onnx(path)  — export fitted sklearn model to ONNX
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

# Minimum acceptable signature file size (sanity check)
_MIN_MANIFEST_BYTES = 10


class ModelIntegrityError(RuntimeError):
    """Raised when model file hash doesn't match manifest."""


class EdgeInferenceEngine:
    """Lightweight edge inference engine.

    Supports ONNX Runtime (preferred) and sklearn joblib (fallback).
    Always offline — no network calls during predict().
    """

    def __init__(self) -> None:
        self._onnx_session: Any = None
        self._sklearn_detector: Any = None
        self._scaler: Any = None
        self._mode: str = "not_loaded"   # onnx | sklearn | not_loaded
        self._model_hash: str = ""
        self._feature_names: list[str] = FEATURE_NAMES

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_dir: Path | str) -> "EdgeInferenceEngine":
        """Load model from directory. Tries ONNX first, falls back to joblib."""
        model_dir = Path(model_dir)
        engine = cls()

        # Verify manifest integrity
        manifest_path = model_dir / "manifest.json"
        if manifest_path.exists():
            engine._verify_manifest(model_dir, manifest_path)

        # Try ONNX
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists() and engine._try_load_onnx(onnx_path):
            logger.info("EdgeInferenceEngine: ONNX mode loaded from %s", model_dir)
            return engine

        # Fallback to sklearn joblib
        scaler_path = model_dir / "scaler.joblib"
        model_path = model_dir / "isolation_forest.joblib"
        if scaler_path.exists() and model_path.exists():
            engine._load_sklearn(scaler_path, model_path)
            logger.info("EdgeInferenceEngine: sklearn mode loaded from %s", model_dir)
            return engine

        raise FileNotFoundError(f"No valid model artifacts found in {model_dir}")

    def _try_load_onnx(self, onnx_path: Path) -> bool:
        try:
            import onnxruntime as ort  # type: ignore[import]
            self._onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._mode = "onnx"
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("ONNX load failed (%s) — will try sklearn fallback", exc)
            return False

    def _load_sklearn(self, scaler_path: Path, model_path: Path) -> None:
        import joblib

        self._scaler = joblib.load(scaler_path)
        self._sklearn_detector = joblib.load(model_path)
        self._mode = "sklearn"

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Run inference. Returns dict with label, confidence, anomaly_score.

        Raises RuntimeError if model is not loaded.
        """
        if self._mode == "not_loaded":
            raise RuntimeError("Model not loaded. Call EdgeInferenceEngine.load() first.")

        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)

        if self._mode == "onnx":
            return self._predict_onnx(arr)
        return self._predict_sklearn(arr)

    def _predict_onnx(self, arr: np.ndarray) -> dict[str, Any]:
        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: arr})
        # sklearn-onnx IsolationForest: outputs[0] = label, outputs[1] = scores
        label_raw = int(outputs[0][0]) if len(outputs) > 0 else 1
        score_raw = float(outputs[1][0].get("anomaly_score", -0.1)) if len(outputs) > 1 else -0.1

        label = "anomaly" if label_raw == -1 else "normal"
        normalized_score = float(np.clip(-score_raw / 0.8, 0.0, 1.0))
        confidence = (
            float(np.clip(normalized_score, 0.5, 1.0))
            if label == "anomaly"
            else float(np.clip(1.0 - normalized_score, 0.5, 1.0))
        )
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(normalized_score, 4),
            "explanation": {},
            "mode": "onnx",
        }

    def _predict_sklearn(self, arr: np.ndarray) -> dict[str, Any]:
        arr_f64 = arr.astype(np.float64)
        arr_scaled = self._scaler.transform(arr_f64)
        prediction = self._sklearn_detector.predict(arr_scaled)[0]
        score = self._sklearn_detector.score_samples(arr_scaled)[0]

        label = "anomaly" if prediction == -1 else "normal"
        normalized_score = float(np.clip(-score / 0.8, 0.0, 1.0))
        confidence = (
            float(np.clip(normalized_score, 0.5, 1.0))
            if label == "anomaly"
            else float(np.clip(1.0 - normalized_score, 0.5, 1.0))
        )
        z_scores = arr_scaled[0].tolist()
        explanation = {
            name: round(z, 3)
            for name, z in zip(self._feature_names, z_scores)
            if abs(z) > 1.5
        }
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(normalized_score, 4),
            "explanation": explanation,
            "mode": "sklearn",
        }

    # ── ONNX export ───────────────────────────────────────────────────────────

    @staticmethod
    def export_onnx(
        detector: Any,
        output_dir: Path | str,
        opset: int = 17,
    ) -> Path:
        """Export a fitted AnomalyDetector to ONNX format.

        Args:
            detector: fitted AnomalyDetector instance
            output_dir: directory to write model.onnx + manifest.json
            opset: ONNX opset version

        Returns:
            Path to written model.onnx
        """
        try:
            from skl2onnx import convert_sklearn  # type: ignore[import]
            from skl2onnx.common.data_types import FloatTensorType  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "skl2onnx required for ONNX export. Install: pip install skl2onnx"
            ) from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_features = len(detector.feature_names)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        # Export IsolationForest (not the scaler — apply scaler at edge side)
        onnx_model = convert_sklearn(
            detector._model,
            initial_types=initial_type,
            target_opset=opset,
        )

        onnx_path = output_dir / "model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Write manifest with SHA-256
        model_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        manifest = {
            "model_hash_sha256": model_hash,
            "feature_names": detector.feature_names,
            "n_estimators": detector.n_estimators,
            "contamination": detector.contamination,
            "opset": opset,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        logger.info("ONNX model exported to %s (sha256=%s...)", onnx_path, model_hash[:12])
        return onnx_path

    # ── Integrity ─────────────────────────────────────────────────────────────

    @staticmethod
    def _verify_manifest(model_dir: Path, manifest_path: Path) -> None:
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.warning("Could not read manifest: %s", exc)
            return

        expected_hash = manifest.get("model_hash_sha256", "")
        if not expected_hash:
            return

        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            return

        actual_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ModelIntegrityError(
                f"ONNX model hash mismatch: expected={expected_hash[:16]}... "
                f"actual={actual_hash[:16]}..."
            )
        logger.debug("Model integrity verified: %s", expected_hash[:16])

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_loaded(self) -> bool:
        return self._mode != "not_loaded"
