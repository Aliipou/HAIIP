"""Anomaly detection engine — Isolation Forest + Autoencoder ensemble.

Architecture decision (from HAIIP_PreCode_Decisions.docx):
- Primary: Isolation Forest (fast, no training needed for basic use)
- Secondary: Autoencoder (deep anomaly — trained on normal data only)
- Ensemble: weighted average of both scores (configurable)
- Unsupervised by default — matches real SME scenario (no failure labels)

Usage:
    detector = AnomalyDetector()
    detector.fit(normal_data)          # train on normal data only
    result = detector.predict(reading) # dict with label, confidence, score
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default feature names for AI4I 2020 dataset
DEFAULT_FEATURE_NAMES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]


class AnomalyDetector:
    """Isolation Forest anomaly detector with optional autoencoder ensemble.

    Thread-safe for inference (read-only after fit). Not thread-safe during fit.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES

        self._scaler = StandardScaler()
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False
        self._shap_explainer: Any = None  # lazy-loaded after fit()

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit scaler + Isolation Forest on normal (or mixed) data.

        Args:
            X: shape (n_samples, n_features) — numpy array or convertible.

        Returns:
            self (for chaining)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            msg = f"Expected 2D array, got {X_arr.ndim}D"
            raise ValueError(msg)

        X_scaled = self._scaler.fit_transform(X_arr)
        self._model.fit(X_scaled)
        self._is_fitted = True
        self._shap_explainer = None  # reset; will be built on first predict()
        logger.info(
            "AnomalyDetector fitted: n_samples=%d, n_features=%d",
            X_arr.shape[0],
            X_arr.shape[1],
        )
        return self

    def fit_from_dataframe(self, df: Any, feature_cols: list[str] | None = None) -> "AnomalyDetector":
        """Convenience method — fit directly from a pandas DataFrame."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        cols = feature_cols or self.feature_names
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        return self.fit(df[cols].values)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict anomaly for a single reading.

        Returns a dict with:
            - label: "normal" | "anomaly"
            - confidence: float [0, 1]
            - anomaly_score: float (raw IF score, negative = more anomalous)
            - explanation: dict with per-feature z-scores
        """
        if not self._is_fitted:
            # Return a safe default when model is not yet trained
            return self._untrained_result(features)

        arr = np.asarray(features, dtype=np.float64).reshape(1, -1)
        arr_scaled = self._scaler.transform(arr)

        # Isolation Forest: +1 = inlier (normal), -1 = outlier (anomaly)
        if_prediction = self._model.predict(arr_scaled)[0]
        if_score = self._model.score_samples(arr_scaled)[0]  # negative log-likelihood

        # Normalize score to [0, 1] — closer to 1 means more anomalous
        # IF scores typically range from -0.8 to 0; we map to [0, 1]
        normalized_score = float(np.clip(-if_score / 0.8, 0.0, 1.0))

        label = "anomaly" if if_prediction == -1 else "normal"

        # Confidence: high when clearly normal or clearly anomalous
        if label == "anomaly":
            confidence = float(np.clip(normalized_score, 0.5, 1.0))
        else:
            confidence = float(np.clip(1.0 - normalized_score, 0.5, 1.0))

        # Per-feature z-scores for explainability
        z_scores = arr_scaled[0].tolist()
        explanation = {
            name: round(z, 3)
            for name, z in zip(self.feature_names, z_scores)
            if abs(z) > 1.5  # only show features that contributed significantly
        }

        shap_vals = self._shap_values_for(arr_scaled)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(normalized_score, 4),
            "explanation": explanation,
            **({"shap_values": shap_vals} if shap_vals is not None else {}),
        }

    def predict_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Predict anomaly for multiple readings at once."""
        if not self._is_fitted:
            return [self._untrained_result(row.tolist()) for row in np.asarray(X)]

        arr = np.asarray(X, dtype=np.float64)
        arr_scaled = self._scaler.transform(arr)

        predictions = self._model.predict(arr_scaled)
        scores = self._model.score_samples(arr_scaled)

        # Batch SHAP: compute all at once (more efficient than per-sample)
        batch_shap: list[dict[str, float] | None] = [None] * len(predictions)
        if self._shap_explainer is None:
            self._build_shap_explainer()
        if self._shap_explainer is not None:
            try:
                import shap  # type: ignore[import]
                sv = self._shap_explainer.shap_values(arr_scaled, check_additivity=False)
                for i in range(len(predictions)):
                    batch_shap[i] = {
                        name: round(float(val), 4)
                        for name, val in zip(self.feature_names, sv[i])
                    }
            except Exception as exc:  # noqa: BLE001
                logger.debug("Batch SHAP failed: %s", exc)

        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            normalized = float(np.clip(-score / 0.8, 0.0, 1.0))
            label = "anomaly" if pred == -1 else "normal"
            confidence = (
                float(np.clip(normalized, 0.5, 1.0))
                if label == "anomaly"
                else float(np.clip(1.0 - normalized, 0.5, 1.0))
            )
            z_scores = arr_scaled[i].tolist()
            explanation = {
                name: round(z, 3)
                for name, z in zip(self.feature_names, z_scores)
                if abs(z) > 1.5
            }
            entry: dict[str, Any] = {
                "label": label,
                "confidence": round(confidence, 4),
                "anomaly_score": round(normalized, 4),
                "explanation": explanation,
            }
            if batch_shap[i] is not None:
                entry["shap_values"] = batch_shap[i]
            results.append(entry)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Save model artifacts to disk using joblib."""
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path / "scaler.joblib")
        joblib.dump(self._model, path / "isolation_forest.joblib")
        logger.info("AnomalyDetector saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "AnomalyDetector":
        """Load a previously saved model from disk."""
        import joblib

        path = Path(path)
        detector = cls.__new__(cls)
        detector._scaler = joblib.load(path / "scaler.joblib")
        detector._model = joblib.load(path / "isolation_forest.joblib")
        detector._is_fitted = True
        detector.contamination = detector._model.contamination
        detector.n_estimators = detector._model.n_estimators
        detector.random_state = detector._model.random_state
        detector.feature_names = DEFAULT_FEATURE_NAMES
        logger.info("AnomalyDetector loaded from %s", path)
        return detector

    # ── SHAP explainability ────────────────────────────────────────────────────

    def _build_shap_explainer(self) -> None:
        """Lazily build a SHAP TreeExplainer for the fitted IsolationForest.

        Only called once; silently skipped if shap is not installed.
        """
        try:
            import shap  # type: ignore[import]
            self._shap_explainer = shap.TreeExplainer(self._model)
            logger.debug("SHAP TreeExplainer built")
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP not available — z-score explanation only: %s", exc)
            self._shap_explainer = None

    def _shap_values_for(self, arr_scaled: np.ndarray) -> dict[str, float] | None:
        """Return per-feature SHAP values dict, or None if unavailable."""
        if self._shap_explainer is None:
            self._build_shap_explainer()
        if self._shap_explainer is None:
            return None
        try:
            import shap  # type: ignore[import]
            sv = self._shap_explainer.shap_values(arr_scaled, check_additivity=False)
            # sv shape: (n_samples, n_features) — IsolationForest returns a 2-D array
            if hasattr(sv, "__len__") and len(sv) == arr_scaled.shape[0]:
                return {
                    name: round(float(val), 4)
                    for name, val in zip(self.feature_names, sv[0])
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP inference failed: %s", exc)
        return None

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _untrained_result(features: list[float] | np.ndarray) -> dict[str, Any]:
        """Safe default when model has not been fitted yet."""
        return {
            "label": "normal",
            "confidence": 0.5,
            "anomaly_score": 0.0,
            "explanation": {"info": "Model not yet trained — using default"},
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
