"""A/B testing framework for ML model variants.

Enables safe model promotion by running two model variants in parallel and
comparing their performance using statistical tests.

Design:
    - Shadow deployment: Variant B runs alongside Variant A without serving traffic
    - Traffic splitting: configurable percentage routed to each variant
    - Statistical significance: Mann-Whitney U test (non-parametric)
    - Human-in-the-loop: promotes Variant B only when significance threshold met

References:
    - Kohavi et al. (2020) Trustworthy Online Controlled Experiments
    - Mann & Whitney (1947) On a Test of Whether One of Two Variables is
      Stochastically Larger than the Other

Usage:
    engine = ABTestingEngine()
    engine.register_variant("model_a", weight=70)
    engine.register_variant("model_b", weight=30)

    # Record results
    engine.record("model_a", prediction_id="p1", metric_value=0.91)
    engine.record("model_b", prediction_id="p2", metric_value=0.94)

    # Evaluate
    result = engine.evaluate(metric="confidence", min_samples=30)
    if result.winner and result.is_significant:
        print(f"Promote {result.winner}! p={result.p_value:.4f}")
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ModelVariant:
    name: str
    weight: int  # relative traffic weight (not required to sum to 100)
    observations: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.observations)

    @property
    def mean(self) -> float:
        return sum(self.observations) / len(self.observations) if self.observations else 0.0

    @property
    def variance(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        m = self.mean
        return sum((x - m) ** 2 for x in self.observations) / (len(self.observations) - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class ABTestResult:
    variant_a: str
    variant_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    relative_improvement: float  # (mean_b - mean_a) / mean_a
    p_value: float
    is_significant: bool
    winner: str | None  # name of winning variant, or None if inconclusive
    recommendation: str


# ── Engine ────────────────────────────────────────────────────────────────────

class ABTestingEngine:
    """Traffic-splitting A/B testing engine for ML model variants.

    Statistical test: Mann-Whitney U (rank-based, distribution-free).
    Falls back to a simple t-test approximation when scipy unavailable.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 30,
    ) -> None:
        self._variants: dict[str, ModelVariant] = {}
        self.significance_level = significance_level
        self.min_samples = min_samples

    # ── Variant management ────────────────────────────────────────────────────

    def register_variant(
        self,
        name: str,
        weight: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._variants[name] = ModelVariant(
            name=name,
            weight=weight,
            metadata=metadata or {},
        )
        logger.info("ab_test.variant_registered", variant=name, weight=weight)

    def select_variant(self) -> str | None:
        """Select a variant by weighted random sampling."""
        if not self._variants:
            return None
        names = list(self._variants.keys())
        weights = [self._variants[n].weight for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    def record(
        self,
        variant_name: str,
        metric_value: float,
        prediction_id: str | None = None,
    ) -> None:
        """Record an observed metric value for a variant."""
        if variant_name not in self._variants:
            raise ValueError(f"Unknown variant: {variant_name}")
        self._variants[variant_name].observations.append(float(metric_value))
        logger.debug(
            "ab_test.observation",
            variant=variant_name,
            value=metric_value,
            prediction_id=prediction_id,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        variant_a: str,
        variant_b: str,
    ) -> ABTestResult:
        """Compare two variants using Mann-Whitney U test."""
        va = self._variants.get(variant_a)
        vb = self._variants.get(variant_b)
        if va is None:
            raise ValueError(f"Unknown variant: {variant_a}")
        if vb is None:
            raise ValueError(f"Unknown variant: {variant_b}")

        n_a, n_b = va.sample_count, vb.sample_count
        mean_a, mean_b = va.mean, vb.mean
        relative_improvement = (
            (mean_b - mean_a) / mean_a if mean_a != 0 else 0.0
        )

        # Minimum sample check
        if n_a < self.min_samples or n_b < self.min_samples:
            return ABTestResult(
                variant_a=variant_a,
                variant_b=variant_b,
                n_a=n_a,
                n_b=n_b,
                mean_a=round(mean_a, 4),
                mean_b=round(mean_b, 4),
                relative_improvement=round(relative_improvement, 4),
                p_value=1.0,
                is_significant=False,
                winner=None,
                recommendation=(
                    f"Insufficient data. Need ≥{self.min_samples} samples per variant "
                    f"(A: {n_a}, B: {n_b})."
                ),
            )

        # Statistical test
        p_value = self._mann_whitney_u(va.observations, vb.observations)
        is_significant = p_value < self.significance_level

        # Determine winner
        if is_significant:
            winner = variant_b if mean_b > mean_a else variant_a
        else:
            winner = None

        recommendation = self._make_recommendation(
            variant_a, variant_b, mean_a, mean_b,
            relative_improvement, p_value, is_significant, winner,
        )

        return ABTestResult(
            variant_a=variant_a,
            variant_b=variant_b,
            n_a=n_a,
            n_b=n_b,
            mean_a=round(mean_a, 4),
            mean_b=round(mean_b, 4),
            relative_improvement=round(relative_improvement, 4),
            p_value=round(p_value, 6),
            is_significant=is_significant,
            winner=winner,
            recommendation=recommendation,
        )

    def get_variant_stats(self) -> list[dict[str, Any]]:
        """Return summary statistics for all registered variants."""
        return [
            {
                "name": v.name,
                "weight": v.weight,
                "sample_count": v.sample_count,
                "mean": round(v.mean, 4),
                "std": round(v.std, 4),
            }
            for v in self._variants.values()
        ]

    def clear_observations(self, variant_name: str | None = None) -> None:
        """Reset observations (call between test periods)."""
        if variant_name:
            if variant_name in self._variants:
                self._variants[variant_name].observations.clear()
        else:
            for v in self._variants.values():
                v.observations.clear()

    # ── Statistical tests ─────────────────────────────────────────────────────

    @staticmethod
    def _mann_whitney_u(a: list[float], b: list[float]) -> float:
        """Mann-Whitney U test — returns two-sided p-value.

        Uses scipy if available; falls back to normal approximation.
        """
        try:
            from scipy.stats import mannwhitneyu  # type: ignore[import]
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            return float(p)
        except ImportError:
            pass
        return ABTestingEngine._normal_approx_u(a, b)

    @staticmethod
    def _normal_approx_u(a: list[float], b: list[float]) -> float:
        """Normal approximation of Mann-Whitney U statistic."""
        n1, n2 = len(a), len(b)
        # Combine and rank
        combined = sorted(enumerate(a + b), key=lambda x: x[1])
        ranks = [0.0] * (n1 + n2)
        i = 0
        while i < len(combined):
            j = i
            # Handle ties
            while j + 1 < len(combined) and combined[j + 1][1] == combined[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[combined[k][0]] = avg_rank
            i = j + 1

        R1 = sum(ranks[:n1])
        U1 = R1 - n1 * (n1 + 1) / 2
        mu_U = n1 * n2 / 2
        sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if sigma_U == 0:
            return 1.0
        z = (U1 - mu_U) / sigma_U
        # Two-sided p-value via error function approximation
        p = 2 * (1 - ABTestingEngine._normal_cdf(abs(z)))
        return max(0.0, min(1.0, p))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF via Abramowitz & Stegun approximation."""
        t = 1 / (1 + 0.2316419 * x)
        poly = t * (0.319381530 + t * (
            -0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        ))
        return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly

    @staticmethod
    def _make_recommendation(
        variant_a: str,
        variant_b: str,
        mean_a: float,
        mean_b: float,
        relative_improvement: float,
        p_value: float,
        is_significant: bool,
        winner: str | None,
    ) -> str:
        if not is_significant:
            return (
                f"No statistically significant difference (p={p_value:.4f}). "
                f"Continue collecting data. Current means: A={mean_a:.4f}, B={mean_b:.4f}."
            )
        direction = "better" if mean_b > mean_a else "worse"
        pct = abs(relative_improvement * 100)
        return (
            f"**{winner}** is statistically {direction} "
            f"(p={p_value:.4f}, Δ={pct:.1f}%). "
            f"{'Promote Variant B.' if winner == variant_b else 'Keep Variant A.'} "
            "Validate with a human expert before deployment."
        )
