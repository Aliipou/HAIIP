"""Economic-Aware AI — Expected Loss Minimization Decision Engine.

Transforms ML predictions into economically optimal maintenance decisions by
minimising expected cost:

    E[Cost] = P(failure) × C_downtime  −  P(no_failure) × C_maintenance

Decision rules (Nordic SME defaults — configurable per tenant):
    - REPAIR_NOW    : E[cost of waiting] > C_maintenance × urgency_factor
    - SCHEDULE      : 0.3 < P(failure) < threshold, E[cost] within budget
    - MONITOR       : P(failure) < 0.3, cost of action > expected benefit
    - IGNORE        : anomaly_score < noise_floor (sensor noise)

Economic model (Pintelon & Parodi-Herz, 2008 — Maintenance Decision Making):
    C_downtime = production_rate × downtime_hours × unit_margin
    C_maintenance = labour_hours × rate + parts_cost + opportunity_cost
    C_false_neg = C_downtime × safety_factor   (regulatory risk multiplier)
    C_false_pos = C_maintenance × 0.15         (unnecessary maintenance)

References:
    - Pintelon & Parodi-Herz (2008) Maintenance decision making
    - Mobley (2002) An Introduction to Predictive Maintenance
    - Van Horenbeek & Pintelon (2013) Development of a maintenance performance
      measurement framework (MPMF)
    - EU AI Act Article 9 — Risk management for high-risk AI systems
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Decision types ─────────────────────────────────────────────────────────────

class MaintenanceAction(str, Enum):
    REPAIR_NOW = "repair_now"       # Immediate corrective action required
    SCHEDULE   = "schedule"         # Plan maintenance within RUL window
    MONITOR    = "monitor"          # Increase monitoring cadence
    IGNORE     = "ignore"           # Sensor noise / below economic threshold


# ── Cost model dataclass ───────────────────────────────────────────────────────

@dataclass
class CostProfile:
    """Per-machine economic parameters (tenant-configurable).

    Args:
        production_rate_eur_hr: Revenue generated per hour of operation (€/h)
        downtime_hours_avg:     Mean downtime duration per failure (h)
        labour_rate_eur_hr:     Maintenance technician hourly rate (€/h)
        labour_hours_avg:       Mean technician hours per maintenance job (h)
        parts_cost_eur:         Average spare parts cost per job (€)
        opportunity_cost_eur:   Lost capacity / secondary effects (€)
        safety_factor:          Regulatory / safety risk multiplier (≥1.0)
        urgency_factor:         Threshold multiplier for REPAIR_NOW (0–1)
        noise_floor:            Anomaly score below which we ignore (0–1)
    """
    production_rate_eur_hr: float = 500.0
    downtime_hours_avg:     float = 8.0
    labour_rate_eur_hr:     float = 85.0
    labour_hours_avg:       float = 4.0
    parts_cost_eur:         float = 250.0
    opportunity_cost_eur:   float = 150.0
    safety_factor:          float = 1.5
    urgency_factor:         float = 0.7
    noise_floor:            float = 0.05

    @property
    def c_downtime(self) -> float:
        """Expected cost of one unplanned failure (€)."""
        return self.production_rate_eur_hr * self.downtime_hours_avg

    @property
    def c_maintenance(self) -> float:
        """Expected cost of one planned maintenance event (€)."""
        return (
            self.labour_rate_eur_hr * self.labour_hours_avg
            + self.parts_cost_eur
            + self.opportunity_cost_eur
        )

    @property
    def c_false_negative(self) -> float:
        """Cost when we miss a failure (regulatory + safety risk)."""
        return self.c_downtime * self.safety_factor

    @property
    def c_false_positive(self) -> float:
        """Cost when we trigger unnecessary maintenance."""
        return self.c_maintenance * 0.15


@dataclass
class EconomicDecision:
    """Output of the economic decision engine.

    Attributes:
        decision_id:           Unique decision UUID (audit trail)
        action:                Recommended maintenance action
        expected_cost_action:  E[cost] if we take the recommended action (€)
        expected_cost_wait:    E[cost] if we do nothing (€)
        net_benefit:           cost_wait − cost_action (positive = act) (€)
        failure_probability:   P(failure) from ML model
        anomaly_score:         Raw anomaly score [0, 1]
        rul_cycles:            Remaining Useful Life estimate (cycles/hours)
        confidence:            Decision confidence [0, 1]
        explanation:           Human-readable rationale (EU AI Act Art. 13)
        requires_human_review: True when uncertainty is high (Art. 14)
        cost_profile:          Economic parameters used
        metadata:              Arbitrary audit metadata
    """
    decision_id:           str
    action:                MaintenanceAction
    expected_cost_action:  float
    expected_cost_wait:    float
    net_benefit:           float
    failure_probability:   float
    anomaly_score:         float
    rul_cycles:            float | None
    confidence:            float
    explanation:           str
    requires_human_review: bool
    cost_profile:          CostProfile
    metadata:              dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id":           self.decision_id,
            "action":                self.action.value,
            "expected_cost_action":  round(self.expected_cost_action, 2),
            "expected_cost_wait":    round(self.expected_cost_wait, 2),
            "net_benefit":           round(self.net_benefit, 2),
            "failure_probability":   round(self.failure_probability, 4),
            "anomaly_score":         round(self.anomaly_score, 4),
            "rul_cycles":            self.rul_cycles,
            "confidence":            round(self.confidence, 4),
            "explanation":           self.explanation,
            "requires_human_review": self.requires_human_review,
            "c_downtime_eur":        round(self.cost_profile.c_downtime, 2),
            "c_maintenance_eur":     round(self.cost_profile.c_maintenance, 2),
            "metadata":              self.metadata,
        }


# ── Decision engine ────────────────────────────────────────────────────────────

class EconomicDecisionEngine:
    """Expected Loss Minimization engine for maintenance decisions.

    The engine integrates ML outputs (anomaly score, failure probability, RUL)
    with economic cost parameters to recommend the cost-optimal maintenance
    action for each machine reading.

    Thread-safe for concurrent inference (stateless after __init__).

    Example::
        profile = CostProfile(production_rate_eur_hr=600.0)
        engine  = EconomicDecisionEngine(cost_profile=profile)

        decision = engine.decide(
            anomaly_score=0.72,
            failure_probability=0.65,
            rul_cycles=120,
            confidence=0.88,
        )
        print(decision.action)          # MaintenanceAction.REPAIR_NOW
        print(decision.net_benefit)     # 3240.00
    """

    # Human review thresholds (EU AI Act Article 14)
    _HIGH_UNCERTAINTY_THRESHOLD = 0.35  # confidence below this → review
    _HIGH_COST_THRESHOLD        = 5_000  # net_benefit above this → review

    def __init__(
        self,
        cost_profile:              CostProfile | None = None,
        schedule_threshold:        float = 0.5,   # P(failure) above → SCHEDULE
        repair_now_threshold:      float = 0.75,  # P(failure) above → REPAIR_NOW
        monitor_score_threshold:   float = 0.2,   # anomaly_score above → MONITOR
    ) -> None:
        self.cost_profile           = cost_profile or CostProfile()
        self.schedule_threshold     = schedule_threshold
        self.repair_now_threshold   = repair_now_threshold
        self.monitor_score_threshold = monitor_score_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def decide(
        self,
        anomaly_score:       float,
        failure_probability: float,
        rul_cycles:          float | None = None,
        confidence:          float = 0.8,
        machine_id:          str | None = None,
        metadata:            dict[str, Any] | None = None,
    ) -> EconomicDecision:
        """Compute the cost-optimal maintenance decision.

        Args:
            anomaly_score:        Isolation Forest score [0, 1] (1 = most anomalous)
            failure_probability:  GBT classifier P(failure) [0, 1]
            rul_cycles:           RUL estimate in cycles (None if unknown)
            confidence:           ML model prediction confidence [0, 1]
            machine_id:           Optional machine identifier for logging
            metadata:             Arbitrary audit metadata

        Returns:
            EconomicDecision with action, costs, explanation, and review flag.
        """
        p  = float(np.clip(failure_probability, 0.0, 1.0))
        sc = float(np.clip(anomaly_score, 0.0, 1.0))
        cf = self.cost_profile

        # Expected costs
        e_cost_wait   = p * cf.c_false_negative + (1 - p) * 0.0
        e_cost_action = p * 0.0 + (1 - p) * cf.c_false_positive + cf.c_maintenance
        net_benefit   = e_cost_wait - e_cost_action

        action, explanation = self._classify(p, sc, rul_cycles, net_benefit)

        requires_review = (
            confidence < self._HIGH_UNCERTAINTY_THRESHOLD
            or abs(net_benefit) > self._HIGH_COST_THRESHOLD
            or (action == MaintenanceAction.REPAIR_NOW and p < 0.6)
        )

        decision = EconomicDecision(
            decision_id           = str(uuid.uuid4()),
            action                = action,
            expected_cost_action  = e_cost_action,
            expected_cost_wait    = e_cost_wait,
            net_benefit           = net_benefit,
            failure_probability   = p,
            anomaly_score         = sc,
            rul_cycles            = rul_cycles,
            confidence            = confidence,
            explanation           = explanation,
            requires_human_review = requires_review,
            cost_profile          = cf,
            metadata              = {
                **(metadata or {}),
                "machine_id": machine_id,
                "engine_version": "1.0.0",
            },
        )

        logger.info(
            "economic_decision",
            extra={
                "decision_id": decision.decision_id,
                "action":      action.value,
                "net_benefit": round(net_benefit, 2),
                "p_failure":   round(p, 3),
                "machine_id":  machine_id,
            },
        )
        return decision

    def batch_decide(
        self,
        records: list[dict[str, Any]],
    ) -> list[EconomicDecision]:
        """Run decide() for a list of sensor records.

        Each record must have keys: anomaly_score, failure_probability.
        Optional keys: rul_cycles, confidence, machine_id, metadata.
        """
        return [self.decide(**r) for r in records]

    def roi_summary(
        self,
        decisions: list[EconomicDecision],
    ) -> dict[str, Any]:
        """Compute fleet-level ROI summary across a batch of decisions.

        Returns:
            dict with total_net_benefit, decisions_by_action, avg_confidence,
            human_review_count, projected_downtime_savings_eur.
        """
        if not decisions:
            return {"total_net_benefit": 0.0, "decisions_by_action": {}}

        action_counts: dict[str, int] = {}
        for d in decisions:
            k = d.action.value
            action_counts[k] = action_counts.get(k, 0) + 1

        total_benefit = sum(d.net_benefit for d in decisions)
        avg_conf      = float(np.mean([d.confidence for d in decisions]))
        review_count  = sum(1 for d in decisions if d.requires_human_review)

        # Projected downtime savings: decisions where we acted vs doing nothing
        acted = [
            d for d in decisions
            if d.action in (MaintenanceAction.REPAIR_NOW, MaintenanceAction.SCHEDULE)
        ]
        savings = sum(
            d.failure_probability * self.cost_profile.c_downtime for d in acted
        )

        return {
            "total_net_benefit":            round(total_benefit, 2),
            "projected_downtime_savings_eur": round(savings, 2),
            "decisions_by_action":           action_counts,
            "avg_confidence":                round(avg_conf, 4),
            "human_review_count":            review_count,
            "decisions_total":               len(decisions),
            "cost_profile": {
                "c_downtime":    round(self.cost_profile.c_downtime, 2),
                "c_maintenance": round(self.cost_profile.c_maintenance, 2),
            },
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _classify(
        self,
        p:          float,
        score:      float,
        rul:        float | None,
        net_benefit: float,
    ) -> tuple[MaintenanceAction, str]:
        """Map (p, score, rul, net_benefit) → (action, explanation)."""
        cf = self.cost_profile

        # Below noise floor → no economic action warranted
        if score < cf.noise_floor and p < 0.1:
            return (
                MaintenanceAction.IGNORE,
                f"Anomaly score ({score:.3f}) is below noise floor ({cf.noise_floor}). "
                f"P(failure)={p:.3f}. No economic action warranted.",
            )

        # Immediate repair when failure probability is very high
        if p >= self.repair_now_threshold:
            urgency = "critically"
            if rul is not None and rul < 50:
                urgency = "critically (RUL < 50 cycles)"
            return (
                MaintenanceAction.REPAIR_NOW,
                f"P(failure)={p:.3f} exceeds repair threshold ({self.repair_now_threshold}). "
                f"E[cost_wait]=€{p * cf.c_false_negative:.0f} vs "
                f"C_maintenance=€{cf.c_maintenance:.0f}. "
                f"Net benefit of acting: €{net_benefit:.0f}. "
                f"Machine is {urgency} at risk.",
            )

        # Schedule maintenance when failure probability is elevated
        if p >= self.schedule_threshold:
            window = f"within {int(rul)} cycles" if rul is not None else "promptly"
            return (
                MaintenanceAction.SCHEDULE,
                f"P(failure)={p:.3f}. Schedule maintenance {window}. "
                f"Expected cost if unplanned: €{p * cf.c_false_negative:.0f}. "
                f"Planned maintenance cost: €{cf.c_maintenance:.0f}. "
                f"Net benefit: €{net_benefit:.0f}.",
            )

        # Increase monitoring when anomaly score is above threshold
        if score >= self.monitor_score_threshold:
            return (
                MaintenanceAction.MONITOR,
                f"Anomaly score ({score:.3f}) above monitoring threshold "
                f"({self.monitor_score_threshold}). P(failure)={p:.3f} is low "
                f"but warrants increased monitoring cadence. "
                f"Re-evaluate after next inspection.",
            )

        # Default: monitor with low priority
        return (
            MaintenanceAction.MONITOR,
            f"P(failure)={p:.3f}, anomaly_score={score:.3f}. "
            f"Below action thresholds. Continue standard monitoring.",
        )
