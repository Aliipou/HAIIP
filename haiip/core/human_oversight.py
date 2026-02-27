"""Human Oversight Quantification Engine — HIR, HOG, Trust Calibration Score.

Provides RDI-grade metrics for quantifying the value and necessity of human
oversight in AI-assisted maintenance decisions, as required by:

    - EU AI Act Article 14 — Human oversight measures
    - ISO/IEC 42001 — AI Management System (human review requirements)
    - NIST AI RMF — GOVERN 1.5 (oversight metrics)

Metrics defined:
    HIR  (Human Intervention Rate):
        Fraction of AI decisions that triggered human review.
        HIR = |interventions| / |decisions|
        High HIR → AI uncertainty is high or thresholds are too conservative.
        Target: 0.05–0.15 for a well-calibrated system.

    HOG  (Human Override Gain):
        F1 improvement when humans correct AI decisions vs accepting them.
        HOG = F1(human_corrected) − F1(ai_only)
        Positive HOG → human oversight adds measurable value.
        Target: > 0.02 (statistically meaningful improvement).

    TCS  (Trust Calibration Score):
        How well AI confidence aligns with actual accuracy.
        TCS = 1 − ECE (Expected Calibration Error)
        TCS = 1.0 → perfect calibration; TCS < 0.7 → miscalibrated.
        Computed via reliability diagram binning (Guo et al., 2017).

    HIR_by_action:
        HIR decomposed by maintenance action category (REPAIR_NOW, SCHEDULE, …)

    Risk Reduction:
        % reduction in expected cost when humans are in the loop.

References:
    - Guo et al. (2017) On Calibration of Modern Neural Networks
    - Amershi et al. (2019) Software Engineering for Machine Learning: A Case Study
    - Ashmore et al. (2021) Assuring the Machine Learning Lifecycle
    - EU AI Act Article 14 — Human oversight (published 2024)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Event records ──────────────────────────────────────────────────────────────

@dataclass
class OversightEvent:
    """One AI decision and its human oversight outcome.

    Args:
        event_id:           Unique event UUID
        decision_id:        Linked EconomicDecision / ML prediction ID
        ai_label:           AI-predicted label (e.g. 'failure', 'normal')
        ai_confidence:      AI confidence score [0, 1]
        true_label:         Ground-truth label (known post-hoc)
        human_reviewed:     Whether a human reviewed this decision
        human_overrode:     Whether the human changed the AI decision
        human_label:        Human's corrected label (None if not overridden)
        action_category:    MaintenanceAction value string
        expected_cost_ai:   E[cost] under AI decision (€)
        expected_cost_human: E[cost] under human decision (€, None if no override)
        metadata:           Arbitrary audit metadata
    """
    event_id:             str
    decision_id:          str
    ai_label:             str
    ai_confidence:        float
    true_label:           str
    human_reviewed:       bool
    human_overrode:       bool = False
    human_label:          str | None = None
    action_category:      str = "unknown"
    expected_cost_ai:     float = 0.0
    expected_cost_human:  float | None = None
    metadata:             dict[str, Any] = field(default_factory=dict)

    @property
    def ai_correct(self) -> bool:
        return self.ai_label == self.true_label

    @property
    def human_correct(self) -> bool:
        if not self.human_overrode or self.human_label is None:
            return self.ai_correct
        return self.human_label == self.true_label

    @classmethod
    def create(
        cls,
        decision_id:         str,
        ai_label:            str,
        ai_confidence:       float,
        true_label:          str,
        human_reviewed:      bool,
        human_overrode:      bool = False,
        human_label:         str | None = None,
        action_category:     str = "unknown",
        expected_cost_ai:    float = 0.0,
        expected_cost_human: float | None = None,
        metadata:            dict[str, Any] | None = None,
    ) -> "OversightEvent":
        return cls(
            event_id            = str(uuid.uuid4()),
            decision_id         = decision_id,
            ai_label            = ai_label,
            ai_confidence       = float(np.clip(ai_confidence, 0.0, 1.0)),
            true_label          = true_label,
            human_reviewed      = human_reviewed,
            human_overrode      = human_overrode,
            human_label         = human_label,
            action_category     = action_category,
            expected_cost_ai    = expected_cost_ai,
            expected_cost_human = expected_cost_human,
            metadata            = metadata or {},
        )


# ── Metric results ─────────────────────────────────────────────────────────────

@dataclass
class OversightMetrics:
    """Computed human oversight quality metrics.

    Attributes:
        hir:              Human Intervention Rate [0, 1]
        hog:              Human Override Gain (F1 delta)
        tcs:              Trust Calibration Score [0, 1]
        ece:              Expected Calibration Error (lower is better)
        hir_by_action:    HIR decomposed by action category
        ai_accuracy:      AI accuracy without human correction
        human_accuracy:   Accuracy after human corrections
        risk_reduction_pct: % reduction in expected cost from human oversight
        n_events:         Total events analysed
        n_reviewed:       Events with human review
        n_overridden:     Events where human changed AI decision
        n_correct_overrides: Overrides that improved the decision
        report:           Human-readable summary
    """
    hir:                  float
    hog:                  float
    tcs:                  float
    ece:                  float
    hir_by_action:        dict[str, float]
    ai_accuracy:          float
    human_accuracy:       float
    risk_reduction_pct:   float
    n_events:             int
    n_reviewed:           int
    n_overridden:         int
    n_correct_overrides:  int
    report:               str

    def to_dict(self) -> dict[str, Any]:
        return {
            "HIR":                  round(self.hir, 4),
            "HOG":                  round(self.hog, 4),
            "TCS":                  round(self.tcs, 4),
            "ECE":                  round(self.ece, 4),
            "HIR_by_action":        {k: round(v, 4) for k, v in self.hir_by_action.items()},
            "ai_accuracy":          round(self.ai_accuracy, 4),
            "human_accuracy":       round(self.human_accuracy, 4),
            "risk_reduction_pct":   round(self.risk_reduction_pct, 2),
            "n_events":             self.n_events,
            "n_reviewed":           self.n_reviewed,
            "n_overridden":         self.n_overridden,
            "n_correct_overrides":  self.n_correct_overrides,
        }


# ── Oversight engine ───────────────────────────────────────────────────────────

class HumanOversightEngine:
    """Records and quantifies human oversight value in AI maintenance decisions.

    Thread-safe for concurrent event recording (append-only log).

    Usage::
        engine = HumanOversightEngine()
        engine.record(OversightEvent.create(
            decision_id    = "abc-123",
            ai_label       = "failure",
            ai_confidence  = 0.72,
            true_label     = "normal",
            human_reviewed = True,
            human_overrode = True,
            human_label    = "normal",
        ))
        metrics = engine.compute_metrics()
        print(metrics.hir)   # 1.0
        print(metrics.hog)   # positive → human added value
    """

    # ECE calibration bins
    N_BINS = 10

    def __init__(self, target_hir: float = 0.10) -> None:
        """
        Args:
            target_hir: Target Human Intervention Rate (SLA threshold)
        """
        self.target_hir  = target_hir
        self._events: list[OversightEvent] = []

    # ── Event management ───────────────────────────────────────────────────────

    def record(self, event: OversightEvent) -> None:
        """Append one oversight event to the log."""
        self._events.append(event)
        logger.debug("oversight_event_recorded", extra={
            "event_id":       event.event_id,
            "human_reviewed": event.human_reviewed,
            "human_overrode": event.human_overrode,
            "ai_correct":     event.ai_correct,
        })

    def record_batch(self, events: list[OversightEvent]) -> None:
        """Record multiple events."""
        for e in events:
            self.record(e)

    def clear(self) -> None:
        """Clear all recorded events (for test isolation)."""
        self._events.clear()

    @property
    def event_count(self) -> int:
        return len(self._events)

    # ── Metric computation ─────────────────────────────────────────────────────

    def compute_metrics(self) -> OversightMetrics:
        """Compute all human oversight metrics from recorded events.

        Returns:
            OversightMetrics with HIR, HOG, TCS, ECE, and more.

        Raises:
            ValueError: if no events have been recorded.
        """
        if not self._events:
            raise ValueError("No oversight events recorded. Call record() first.")

        events = self._events

        # ── HIR ──
        hir = sum(1 for e in events if e.human_reviewed) / len(events)

        # ── HOG ──
        ai_labels    = np.array([e.ai_label    for e in events])
        true_labels  = np.array([e.true_label  for e in events])
        human_labels = np.array([
            e.human_label if (e.human_overrode and e.human_label) else e.ai_label
            for e in events
        ])
        ai_correct    = (ai_labels    == true_labels).mean()
        human_correct = (human_labels == true_labels).mean()
        hog = float(human_correct - ai_correct)

        # ── TCS / ECE ──
        confidences = np.array([e.ai_confidence for e in events])
        correctness = (ai_labels == true_labels).astype(float)
        ece = self._compute_ece(confidences, correctness)
        tcs = max(0.0, 1.0 - ece)

        # ── HIR by action ──
        action_categories = set(e.action_category for e in events)
        hir_by_action: dict[str, float] = {}
        for cat in action_categories:
            cat_events = [e for e in events if e.action_category == cat]
            if cat_events:
                hir_by_action[cat] = sum(
                    1 for e in cat_events if e.human_reviewed
                ) / len(cat_events)

        # ── Risk reduction ──
        ai_costs    = sum(e.expected_cost_ai for e in events)
        human_costs = sum(
            e.expected_cost_human if (e.human_overrode and e.expected_cost_human is not None)
            else e.expected_cost_ai
            for e in events
        )
        risk_reduction_pct = (
            (ai_costs - human_costs) / ai_costs * 100.0
            if ai_costs > 0 else 0.0
        )

        # ── Override stats ──
        overrides         = [e for e in events if e.human_overrode]
        correct_overrides = [e for e in overrides if e.human_correct and not e.ai_correct]

        report = self._generate_report(
            hir=hir, hog=hog, tcs=tcs, ece=ece,
            ai_accuracy=float(ai_correct),
            human_accuracy=float(human_correct),
            risk_reduction_pct=risk_reduction_pct,
            n_events=len(events),
            n_reviewed=sum(1 for e in events if e.human_reviewed),
            n_overridden=len(overrides),
        )

        return OversightMetrics(
            hir                = round(hir, 4),
            hog                = round(hog, 4),
            tcs                = round(tcs, 4),
            ece                = round(ece, 4),
            hir_by_action      = hir_by_action,
            ai_accuracy        = round(float(ai_correct), 4),
            human_accuracy     = round(float(human_correct), 4),
            risk_reduction_pct = round(risk_reduction_pct, 2),
            n_events           = len(events),
            n_reviewed         = sum(1 for e in events if e.human_reviewed),
            n_overridden       = len(overrides),
            n_correct_overrides= len(correct_overrides),
            report             = report,
        )

    def rolling_hir(self, window: int = 50) -> list[float]:
        """Compute rolling HIR over a sliding window (for dashboard trend)."""
        if len(self._events) < window:
            window = len(self._events)
        if window == 0:
            return []
        return [
            sum(1 for e in self._events[max(0, i - window):i] if e.human_reviewed) / window
            for i in range(window, len(self._events) + 1)
        ]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_ece(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = N_BINS,
    ) -> float:
        """Expected Calibration Error via equal-width binning (Guo et al., 2017).

        ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|
        """
        n   = len(confidences)
        ece = 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if not mask.any():
                continue
            acc  = correctness[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
        return float(ece)

    def _generate_report(
        self,
        hir: float,
        hog: float,
        tcs: float,
        ece: float,
        ai_accuracy: float,
        human_accuracy: float,
        risk_reduction_pct: float,
        n_events: int,
        n_reviewed: int,
        n_overridden: int,
    ) -> str:
        """Generate a structured EU AI Act Article 14 compliance report."""
        hir_status = "OK" if abs(hir - self.target_hir) <= 0.05 else "WARN"
        hog_status = "POSITIVE" if hog > 0 else ("NEUTRAL" if hog == 0 else "NEGATIVE")
        tcs_status = "GOOD" if tcs >= 0.8 else ("FAIR" if tcs >= 0.7 else "POOR")

        return (
            f"=== Human Oversight Report (EU AI Act Art. 14) ===\n"
            f"Events analysed       : {n_events}\n"
            f"Human reviews         : {n_reviewed} ({hir * 100:.1f}%) [{hir_status}]\n"
            f"Human overrides       : {n_overridden}\n"
            f"\n"
            f"HIR  (Intervention Rate)   : {hir:.4f}  [target: {self.target_hir:.2f}]\n"
            f"HOG  (Override Gain)       : {hog:+.4f} [{hog_status}]\n"
            f"TCS  (Calibration Score)   : {tcs:.4f}  [{tcs_status}]\n"
            f"ECE  (Calibration Error)   : {ece:.4f}\n"
            f"\n"
            f"AI accuracy (unaided)      : {ai_accuracy:.4f}\n"
            f"Human accuracy (corrected) : {human_accuracy:.4f}\n"
            f"Risk cost reduction        : {risk_reduction_pct:+.1f}%\n"
            f"\n"
            f"Assessment: Human oversight {'adds measurable value' if hog > 0.02 else 'has marginal impact'}. "
            f"Calibration is {tcs_status.lower()}. "
            f"{'Recommend reviewing confidence thresholds.' if tcs < 0.7 else 'System is operating within acceptable parameters.'}"
        )
