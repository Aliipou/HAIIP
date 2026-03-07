"""Concept drift detection — statistical monitoring of incoming data distributions.

Uses:
- Page-Hinkley test: sequential drift detection on univariate streams
- Kolmogorov-Smirnov test: batch comparison of reference vs current distribution
- Population Stability Index (PSI): industry-standard drift metric

Architecture decision: drift is detected at the feature level, not the prediction
level. This gives early warning before model performance degrades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

PSI_THRESHOLDS = {
    "stable": 0.1,
    "monitoring": 0.2,
    "drift": float("inf"),
}


@dataclass
class DriftResult:
    feature_name: str
    drift_detected: bool
    psi: float
    ks_statistic: float
    ks_pvalue: float
    severity: str  # "stable" | "monitoring" | "drift"
    details: dict[str, Any] = field(default_factory=dict)


class PageHinkleyDetector:
    """Page-Hinkley sequential change-point detector for univariate streams.

    Raises a drift signal when cumulative sum of deviations exceeds threshold.

    Args:
        delta: minimum magnitude of acceptable change (0.005 default)
        threshold: detection threshold — higher = less sensitive
        alpha: forgetting factor for running mean (1.0 = no forgetting)
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1.0,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._reset()

    def _reset(self) -> None:
        self._sum = 0.0
        self._x_mean = 0.0
        self._n = 0
        self._min_sum = 0.0

    def update(self, value: float) -> bool:
        """Update with new value. Returns True if drift is detected."""
        self._n += 1
        self._x_mean = self._x_mean + (value - self._x_mean) / self._n
        self._sum += value - self._x_mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        drift = (self._sum - self._min_sum) > self.threshold
        if drift:
            logger.warning("PageHinkley drift detected after %d samples", self._n)
            self._reset()
        return drift

    def reset(self) -> None:
        self._reset()


class DriftDetector:
    """Multi-feature concept drift detector.

    Compares a reference distribution (fitted on training data) against
    incoming production data using KS test and PSI.

    Usage:
        detector = DriftDetector(feature_names=["air_temp", "torque"])
        detector.fit_reference(X_train)
        result = detector.check(X_production)
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        drift_threshold: float = 0.05,
        n_bins: int = 10,
    ) -> None:
        self.feature_names = feature_names or []
        self.drift_threshold = drift_threshold
        self.n_bins = n_bins
        self._reference: np.ndarray | None = None
        self._ph_detectors: dict[str, PageHinkleyDetector] = {}

    def fit_reference(self, X: np.ndarray, feature_names: list[str] | None = None) -> DriftDetector:
        """Store reference distribution for comparison.

        Args:
            X: (n_samples, n_features) reference dataset (training data)
            feature_names: optional override for feature names
        """
        self._reference = np.asarray(X, dtype=np.float64)
        if feature_names:
            self.feature_names = feature_names
        elif not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(self._reference.shape[1])]

        # Initialise per-feature Page-Hinkley detectors
        self._ph_detectors = {name: PageHinkleyDetector() for name in self.feature_names}
        logger.info(
            "DriftDetector reference fitted: n_samples=%d, n_features=%d",
            self._reference.shape[0],
            self._reference.shape[1],
        )
        return self

    def check(self, X_current: np.ndarray) -> list[DriftResult]:
        """Compare current data distribution against reference.

        Returns a DriftResult per feature.
        """
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before check()")

        X_cur = np.asarray(X_current, dtype=np.float64)
        results: list[DriftResult] = []

        for i, name in enumerate(self.feature_names):
            ref_col = self._reference[:, i]
            cur_col = X_cur[:, i]

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_col, cur_col)

            # Population Stability Index
            psi = self._compute_psi(ref_col, cur_col)

            # Severity classification
            if psi < PSI_THRESHOLDS["stable"]:
                severity = "stable"
            elif psi < PSI_THRESHOLDS["monitoring"]:
                severity = "monitoring"
            else:
                severity = "drift"

            drift_detected = ks_pvalue < self.drift_threshold or severity == "drift"

            results.append(
                DriftResult(
                    feature_name=name,
                    drift_detected=drift_detected,
                    psi=round(float(psi), 6),
                    ks_statistic=round(float(ks_stat), 6),
                    ks_pvalue=round(float(ks_pvalue), 6),
                    severity=severity,
                    details={
                        "ref_mean": round(float(ref_col.mean()), 4),
                        "cur_mean": round(float(cur_col.mean()), 4),
                        "ref_std": round(float(ref_col.std()), 4),
                        "cur_std": round(float(cur_col.std()), 4),
                    },
                )
            )

        return results

    def check_stream(self, values: list[float]) -> dict[str, bool]:
        """Check a single row of values against Page-Hinkley detectors.

        Returns dict of {feature_name: drift_detected}.
        """
        result: dict[str, bool] = {}
        for name, value in zip(self.feature_names, values):
            if name in self._ph_detectors:
                result[name] = self._ph_detectors[name].update(value)
            else:
                result[name] = False
        return result

    def summary(self, results: list[DriftResult]) -> dict[str, Any]:
        """Summarise drift results into a dashboard-friendly dict."""
        any_drift = any(r.drift_detected for r in results)
        return {
            "drift_detected": any_drift,
            "severity": max(
                (r.severity for r in results),
                key=lambda s: ["stable", "monitoring", "drift"].index(s),
            ),
            "affected_features": [r.feature_name for r in results if r.drift_detected],
            "feature_details": [
                {
                    "name": r.feature_name,
                    "psi": r.psi,
                    "ks_pvalue": r.ks_pvalue,
                    "severity": r.severity,
                }
                for r in results
            ],
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Population Stability Index — measures distribution shift magnitude.

        PSI < 0.1: no drift
        PSI 0.1–0.2: monitoring required
        PSI > 0.2: significant drift
        """
        # Bin on reference distribution
        _, bin_edges = np.histogram(reference, bins=self.n_bins)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions, add epsilon to avoid log(0)
        ref_pct = (ref_counts + 1e-8) / (len(reference) + 1e-8 * self.n_bins)
        cur_pct = (cur_counts + 1e-8) / (len(current) + 1e-8 * self.n_bins)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(0.0, psi)
