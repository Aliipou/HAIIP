"""Predictive maintenance engine — failure mode classification + RUL estimation.

Architecture:
- Failure classifier: RandomForest trained on AI4I 2020 failure labels
- RUL estimator: Gradient Boosted Regressor on tool wear / degradation curves
- Output: failure probability, predicted failure mode, RUL in cycles

AI4I 2020 failure modes:
    TWF  — Tool Wear Failure
    HDF  — Heat Dissipation Failure
    PWF  — Power Failure
    OSF  — Overstrain Failure
    RNF  — Random Failures (noise)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FAILURE_MODES = ["no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

DEFAULT_FEATURE_NAMES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]


class MaintenancePredictor:
    """Gradient Boosted predictive maintenance classifier and RUL regressor.

    Produces:
        - prediction_label: "no_failure" | "TWF" | "HDF" | "PWF" | "OSF" | "RNF"
        - failure_probability: float [0, 1]
        - rul_cycles: estimated remaining cycles before failure
        - confidence: model confidence in prediction
        - explanation: top contributing features
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES

        self._scaler = StandardScaler()
        self._classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._rul_regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._is_fitted = False
        self._classes: list[str] = FAILURE_MODES
        self._shap_clf_explainer: Any = None  # lazy-loaded after fit()
        self._shap_rul_explainer: Any = None  # lazy-loaded after fit()

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_rul: np.ndarray | None = None,
    ) -> MaintenancePredictor:
        """Fit the classifier (and optionally the RUL regressor).

        Args:
            X:       (n_samples, n_features) sensor readings
            y_class: (n_samples,) failure mode labels (strings from FAILURE_MODES)
            y_rul:   (n_samples,) remaining useful life in cycles (optional)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.fit_transform(X_arr)

        self._classifier.fit(X_scaled, y_class)
        self._classes = list(self._classifier.classes_)

        if y_rul is not None:
            y_rul_arr = np.asarray(y_rul, dtype=np.float64)
            self._rul_regressor.fit(X_scaled, y_rul_arr)
            self._rul_fitted = True
        else:
            self._rul_fitted = False

        self._is_fitted = True
        # Reset SHAP explainers — must rebuild after refit
        self._shap_clf_explainer = None
        self._shap_rul_explainer = None
        logger.info(
            "MaintenancePredictor fitted: n_samples=%d, classes=%s",
            X_arr.shape[0],
            self._classes,
        )
        return self

    def fit_from_dataframe(
        self,
        df: Any,
        feature_cols: list[str] | None = None,
        label_col: str = "failure_type",
        rul_col: str | None = None,
    ) -> MaintenancePredictor:
        """Convenience: fit from pandas DataFrame."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        cols = feature_cols or self.feature_names
        X = df[cols].values
        y_class = df[label_col].values
        y_rul = df[rul_col].values if rul_col and rul_col in df.columns else None
        return self.fit(X, y_class, y_rul)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict failure mode and RUL for a single reading."""
        if not self._is_fitted:
            return self._untrained_result()

        arr = np.asarray(features, dtype=np.float64).reshape(1, -1)
        arr_scaled = self._scaler.transform(arr)

        # Class probabilities
        proba = self._classifier.predict_proba(arr_scaled)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self._classes[pred_idx]
        pred_proba = float(proba[pred_idx])

        # RUL estimation
        rul_cycles: int | None = None
        if hasattr(self, "_rul_fitted") and self._rul_fitted:
            rul_raw = float(self._rul_regressor.predict(arr_scaled)[0])
            rul_cycles = max(0, int(round(rul_raw)))

        # Feature importance for top contributors
        importances = self._classifier.feature_importances_
        top_features = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
        }

        # Failure probability = sum of all non-normal class probabilities
        normal_idx = self._classes.index("no_failure") if "no_failure" in self._classes else -1
        failure_proba = 1.0 - float(proba[normal_idx]) if normal_idx >= 0 else pred_proba

        shap_clf = self._shap_for_prediction(arr_scaled, pred_idx)
        shap_rul = self._shap_rul_for(arr_scaled)

        result: dict[str, Any] = {
            "label": pred_label,
            "confidence": round(pred_proba, 4),
            "failure_probability": round(failure_proba, 4),
            "rul_cycles": rul_cycles,
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self._classes, proba)
            },
            "explanation": {
                "top_features": top_features,
                "failure_mode": pred_label,
            },
        }
        if shap_clf is not None:
            result["shap_values"] = shap_clf
        if shap_rul is not None:
            result["shap_rul_values"] = shap_rul
        return result

    def predict_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Predict failure mode for multiple readings."""
        if not self._is_fitted:
            return [self._untrained_result() for _ in range(len(X))]

        arr = np.asarray(X, dtype=np.float64)
        arr_scaled = self._scaler.transform(arr)
        probas = self._classifier.predict_proba(arr_scaled)
        importances = self._classifier.feature_importances_

        results = []
        for proba in probas:
            pred_idx = int(np.argmax(proba))
            pred_label = self._classes[pred_idx]
            normal_idx = self._classes.index("no_failure") if "no_failure" in self._classes else -1
            failure_proba = (
                1.0 - float(proba[normal_idx]) if normal_idx >= 0 else float(proba[pred_idx])
            )

            results.append(
                {
                    "label": pred_label,
                    "confidence": round(float(proba[pred_idx]), 4),
                    "failure_probability": round(failure_proba, 4),
                    "rul_cycles": None,
                    "explanation": {
                        "top_features": {
                            name: round(float(imp), 4)
                            for name, imp in sorted(
                                zip(self.feature_names, importances),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3]
                        }
                    },
                }
            )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path / "scaler.joblib")
        joblib.dump(self._classifier, path / "classifier.joblib")
        if hasattr(self, "_rul_fitted") and self._rul_fitted:
            joblib.dump(self._rul_regressor, path / "rul_regressor.joblib")
        logger.info("MaintenancePredictor saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> MaintenancePredictor:
        import joblib

        path = Path(path)
        predictor = cls.__new__(cls)
        predictor._scaler = joblib.load(path / "scaler.joblib")
        predictor._classifier = joblib.load(path / "classifier.joblib")
        predictor._is_fitted = True
        predictor.feature_names = DEFAULT_FEATURE_NAMES
        predictor._classes = list(predictor._classifier.classes_)

        rul_path = path / "rul_regressor.joblib"
        if rul_path.exists():
            predictor._rul_regressor = joblib.load(rul_path)
            predictor._rul_fitted = True
        else:
            predictor._rul_fitted = False
            predictor._rul_regressor = GradientBoostingRegressor()

        logger.info("MaintenancePredictor loaded from %s", path)
        return predictor

    # ── SHAP explainability ────────────────────────────────────────────────────

    def _build_shap_explainers(self) -> None:
        """Lazily build SHAP TreeExplainers for classifier (and RUL regressor if fitted)."""
        try:
            import shap  # type: ignore[import]

            self._shap_clf_explainer = shap.TreeExplainer(self._classifier)
            if hasattr(self, "_rul_fitted") and self._rul_fitted:
                self._shap_rul_explainer = shap.TreeExplainer(self._rul_regressor)
            logger.debug("MaintenancePredictor: SHAP explainers built")
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP not available for MaintenancePredictor: %s", exc)
            self._shap_clf_explainer = None
            self._shap_rul_explainer = None

    def _shap_for_prediction(
        self, arr_scaled: np.ndarray, pred_class_idx: int
    ) -> dict[str, float] | None:
        """Return SHAP values for the predicted failure class."""
        if self._shap_clf_explainer is None:
            self._build_shap_explainers()
        if self._shap_clf_explainer is None:
            return None
        try:
            sv = self._shap_clf_explainer.shap_values(arr_scaled, check_additivity=False)
            # Multi-class GBM: sv is list[array] where sv[class_idx] has shape (n, features)
            if isinstance(sv, list) and len(sv) > pred_class_idx:
                class_sv = sv[pred_class_idx][0]
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                class_sv = sv[0, pred_class_idx, :]
            else:
                return None
            return {name: round(float(val), 4) for name, val in zip(self.feature_names, class_sv)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP clf inference failed: %s", exc)
            return None

    def _shap_rul_for(self, arr_scaled: np.ndarray) -> dict[str, float] | None:
        """Return SHAP values for RUL regressor."""
        if not (hasattr(self, "_rul_fitted") and self._rul_fitted):
            return None
        if self._shap_rul_explainer is None:
            self._build_shap_explainers()
        if self._shap_rul_explainer is None:
            return None
        try:
            sv = self._shap_rul_explainer.shap_values(arr_scaled, check_additivity=False)
            return {name: round(float(val), 4) for name, val in zip(self.feature_names, sv[0])}
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP RUL inference failed: %s", exc)
            return None

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _untrained_result() -> dict[str, Any]:
        return {
            "label": "no_failure",
            "confidence": 0.5,
            "failure_probability": 0.0,
            "rul_cycles": None,
            "explanation": {"info": "Model not yet trained — using default"},
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
