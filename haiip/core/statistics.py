"""
Statistical Analysis Utilities for PhD-Grade Reporting
=======================================================
All ML performance numbers must be reported with confidence intervals
and significance tests before appearing in any publication.

Rules:
- No single-split F1 numbers without CI.
- Always report effect size alongside p-value.
- McNemar's test (not paired t-test) for classifier comparison.
- Bootstrap CI is distribution-free and appropriate for F1.

Usage:
    from haiip.core.statistics import bootstrap_f1_ci, mcnemar_test, cohens_d

    ci_low, ci_high = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=1000)
    p_value = mcnemar_test(y_true, y_pred_a, y_pred_b)
    d = cohens_d(f1_scores_a, f1_scores_b)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Bootstrap confidence interval on F1
# ---------------------------------------------------------------------------


def bootstrap_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    average: str = "macro",
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Returns (ci_low, ci_high) — the bootstrap percentile CI on F1-macro.

    Distribution-free: makes no Gaussian assumption on F1 distribution.
    Recommended for imbalanced binary or multi-class classification.

    Reference: Efron & Tibshirani (1994), An Introduction to the Bootstrap.
    """
    rng = np.random.default_rng(random_state)
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    n = len(y_true_a)

    boot_scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = f1_score(y_true_a[idx], y_pred_a[idx], average=average, zero_division=0)
            boot_scores.append(score)
        except Exception:  # noqa: BLE001,S112
            continue

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_scores, 100 * alpha))
    hi = float(np.percentile(boot_scores, 100 * (1.0 - alpha)))
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# Cross-validation F1 with CI
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    mean_f1: float
    std_f1: float
    ci_low: float
    ci_high: float
    ci_level: float
    fold_f1s: list[float]

    def __str__(self) -> str:
        return (
            f"F1 = {self.mean_f1:.4f} ± {self.std_f1:.4f} "
            f"({int(self.ci_level * 100)}% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}])"
        )


def cross_validated_f1(
    estimator: object,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    average: str = "macro",
    random_state: int = 42,
    ci: float = 0.95,
) -> CVResult:
    """
    Run stratified k-fold CV and return F1 with confidence interval.

    Uses the t-distribution CI: mean ± t_{n-1, alpha/2} × SE.
    Appropriate for k-fold CV where k is small (5 or 10).

    Reference: Bengio & Grandvalet (2004) for caveats of CV CI.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring=f"f1_{average}")
    fold_f1s = scores.tolist()

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    se = std / math.sqrt(n_splits)

    # t-distribution critical value (approximate using 1.96 for n_splits >= 5)
    # For exact value: scipy.stats.t.ppf((1+ci)/2, df=n_splits-1)
    try:
        from scipy.stats import t as t_dist

        t_crit = t_dist.ppf((1 + ci) / 2, df=n_splits - 1)
    except ImportError:
        t_crit = 2.776 if n_splits == 5 else 1.96  # conservative fallback

    margin = t_crit * se
    return CVResult(
        mean_f1=round(mean, 4),
        std_f1=round(std, 4),
        ci_low=round(max(0.0, mean - margin), 4),
        ci_high=round(min(1.0, mean + margin), 4),
        ci_level=ci,
        fold_f1s=[round(f, 4) for f in fold_f1s],
    )


# ---------------------------------------------------------------------------
# McNemar's test — correct test for comparing two classifiers
# ---------------------------------------------------------------------------


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
) -> float:
    """
    McNemar's test for statistical significance of classifier difference.

    Tests whether models A and B make different errors (not whether one has
    higher accuracy). Correct for paired binary outcomes.

    Reference: McNemar (1947). Note on the sampling error of the difference
    between correlated proportions. Psychometrika, 12(2), 153–157.

    Returns p-value. p < 0.05 -> models differ significantly.
    """
    y_true_a = np.asarray(y_true)
    y_pred_a_ = np.asarray(y_pred_a)
    y_pred_b_ = np.asarray(y_pred_b)

    correct_a = y_pred_a_ == y_true_a
    correct_b = y_pred_b_ == y_true_a

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 1.0  # identical predictions — no difference

    # Mid-p McNemar (Edwards continuity correction)
    # chi2 = (|b - c| - 1)^2 / (b + c)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    try:
        from scipy.stats import chi2 as chi2_dist

        p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Approximate: chi2 with df=1, p < 0.05 when chi2 > 3.84
        p_value = 0.05 if chi2 > 3.84 else 1.0

    return round(p_value, 6)


# ---------------------------------------------------------------------------
# Cohen's d — effect size
# ---------------------------------------------------------------------------


def cohens_d(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> float:
    """
    Cohen's d effect size between two sets of F1 scores (e.g. from k-fold CV).

    Interpretation (Cohen 1988):
      |d| < 0.2  : negligible
      |d| < 0.5  : small
      |d| < 0.8  : medium
      |d| >= 0.8 : large

    Reference: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    pooled_std = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0

    return round(float((np.mean(a) - np.mean(b)) / pooled_std), 4)


# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------


def psi(
    reference: Sequence[float],
    current: Sequence[float],
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index between reference and current distributions.

    PSI < 0.10  : stable (no action)
    PSI < 0.20  : slight shift (monitor)
    PSI >= 0.20 : significant shift (investigate)
    PSI >= 0.25 : major shift (retrain)

    Reference: Yurdakul (2018) for PSI formula and thresholds.
    """
    ref_a = np.asarray(reference, dtype=float)
    cur_a = np.asarray(current, dtype=float)

    # Use reference bin edges
    _, bin_edges = np.histogram(ref_a, bins=n_bins)
    ref_counts = np.histogram(ref_a, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(cur_a, bins=bin_edges)[0].astype(float)

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    # Clip to avoid log(0)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi_value, 6)


# ---------------------------------------------------------------------------
# Calibration — Expected Calibration Error
# ---------------------------------------------------------------------------


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) via equal-width binning.

    ECE < 0.10: well-calibrated (TCS >= 0.90)
    ECE < 0.20: acceptable
    ECE >= 0.20: poorly calibrated — consider Platt scaling

    Reference: Guo et al. (2017), On Calibration of Modern Neural Networks, ICML.
    """
    conf_a = np.asarray(confidences, dtype=float)
    corr_a = np.asarray(correctness, dtype=float)
    n = len(conf_a)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf_a >= lo) & (conf_a < hi)
        if i == n_bins - 1:
            mask = (conf_a >= lo) & (conf_a <= hi)

        if mask.sum() == 0:
            continue

        bin_conf = float(conf_a[mask].mean())
        bin_acc = float(corr_a[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return round(ece, 6)
