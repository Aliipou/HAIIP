"""
Operator Simulation Model — Honest Simulation for Human Oversight Metrics
==========================================================================
Simulates operator decision behaviour for testing HIR / HOG / TCS metrics.

IMPORTANT: This is a simulation. Every assumption is a named constant with
a citation or explicit "Source: model assumption, no citation" marker.
A field study with real operators is required before reporting HOG/TCS
to external stakeholders.

All oversight reports generated using this model carry:
  simulation_confidence: 'LOW'
  field_study_required: True

This is hardcoded and cannot be overridden without changing the source.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation assumptions — NAMED CONSTANTS, every one cited or flagged.
# ---------------------------------------------------------------------------

# Expert operator accepts 82% of true positive alerts.
# Source: Kaasinen et al. 2022, "Human Robot Interaction in Industrial Settings",
#         Table 3, expert group (N=24 operators, automotive sector).
# Testable: compare to real operator logs when available.
# Confidence: MEDIUM — single study, automotive context not directly Nordic SME.
ASSUMPTION_ACCEPT_RATE_EXPERT: float = 0.82

# Novice accepts 54% — lower trust in AI, more overrides.
# Source: extrapolated from Kaasinen et al. 2022, no direct novice measurement.
# Confidence: LOW — must be validated with real operators.
ASSUMPTION_ACCEPT_RATE_NOVICE: float = 0.54

# Alert acceptance rate multiplier per 4-hour shift block (fatigue effect).
# After 4h: rate * 0.91; after 8h: rate * 0.91^2 = 0.828.
# Source: general HCI fatigue literature (Mackworth 1948, Warm et al. 2008),
#         not industrial-specific. No direct Nordic SME citation.
# Confidence: LOW — direction is correct, magnitude is model assumption.
ASSUMPTION_FATIGUE_FACTOR: float = 0.91

# Each prior false positive reduces acceptance rate for that fault type by 3pp.
# Source: model assumption, no citation.
# Confidence: VERY LOW — key uncertainty in the simulation.
ASSUMPTION_FALSE_POSITIVE_LEARNING: float = 0.03

# Showing SHAP explanation increases acceptance rate by 8 percentage points.
# Source: RQ3 hypothesis (not yet measured). This is what RQ3 is designed to measure.
# Confidence: NONE — unvalidated hypothesis.
ASSUMPTION_EXPLANATION_BOOST: float = 0.08


ASSUMPTION_METADATA: dict[str, dict[str, str]] = {
    "ASSUMPTION_ACCEPT_RATE_EXPERT": {
        "value": str(ASSUMPTION_ACCEPT_RATE_EXPERT),
        "source": "Kaasinen et al. 2022, HRI in industrial settings, Table 3",
        "confidence": "MEDIUM",
        "testable": "Compare to real operator logs when available",
    },
    "ASSUMPTION_ACCEPT_RATE_NOVICE": {
        "value": str(ASSUMPTION_ACCEPT_RATE_NOVICE),
        "source": "Extrapolated from Kaasinen et al. 2022, no direct citation",
        "confidence": "LOW",
        "testable": "Requires field study with novice operators",
    },
    "ASSUMPTION_FATIGUE_FACTOR": {
        "value": str(ASSUMPTION_FATIGUE_FACTOR),
        "source": "Mackworth 1948, Warm et al. 2008; general HCI literature, not industrial-specific",
        "confidence": "LOW",
        "testable": "Requires longitudinal shift study at real SME",
    },
    "ASSUMPTION_FALSE_POSITIVE_LEARNING": {
        "value": str(ASSUMPTION_FALSE_POSITIVE_LEARNING),
        "source": "Model assumption, no citation",
        "confidence": "VERY LOW",
        "testable": "Track operator rejection rate over time correlated with FP history",
    },
    "ASSUMPTION_EXPLANATION_BOOST": {
        "value": str(ASSUMPTION_EXPLANATION_BOOST),
        "source": "RQ3 hypothesis — not yet measured",
        "confidence": "NONE",
        "testable": "A/B test: same alert with/without SHAP explanation, measure accept rate",
    },
}

# Assumptions where confidence is LOW, VERY LOW, or NONE.
# Any report using these must include explicit warnings.
LOW_CONFIDENCE_ASSUMPTIONS = [
    "ASSUMPTION_ACCEPT_RATE_NOVICE",
    "ASSUMPTION_FATIGUE_FACTOR",
    "ASSUMPTION_FALSE_POSITIVE_LEARNING",
    "ASSUMPTION_EXPLANATION_BOOST",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class OperatorRole(str, Enum):
    EXPERT = "expert"
    NOVICE = "novice"
    TRAINEE = "trainee"


@dataclass
class OperatorProfile:
    role: OperatorRole
    experience_years: float  # 0 = first day; 10+ = expert
    has_seen_explanation: bool = False


@dataclass
class AlertStub:
    """Minimal alert representation for simulation (avoids DB dependency)."""

    alert_id: str
    machine_id: str
    fault_type: str  # e.g. 'HDF', 'TWF'
    is_true_positive: bool
    ai_confidence: float  # [0, 1]
    has_explanation: bool = False  # was SHAP shown?


@dataclass
class SimulationDecision:
    alert_id: str
    decision: str  # 'accept' | 'reject' | 'escalate'
    simulated_confidence: float
    accept_probability_used: float
    parameters_logged: dict[str, float]


# ---------------------------------------------------------------------------
# Simulation model
# ---------------------------------------------------------------------------


class OperatorSimulationModel:
    """
    Simulates operator decision behaviour for testing oversight metrics.

    IMPORTANT: This is a simulation. Every assumption listed above is a
    testable hypothesis. Field study required to validate before publication.

    Simulation confidence: LOW (hardcoded — cannot be upgraded without field data).
    """

    LOW_CONFIDENCE_ASSUMPTIONS = LOW_CONFIDENCE_ASSUMPTIONS

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def simulate_operator_decision(
        self,
        alert: AlertStub,
        operator_profile: OperatorProfile,
        shift_hour: int,  # 0-11; 0 = start of shift, 11 = end of 12h shift
        prior_false_positives: int,  # FP count for this fault type in this shift
    ) -> SimulationDecision:
        """
        Returns SimulationDecision with decision + simulated_confidence.
        decision: 'accept' | 'reject' | 'escalate'

        Every parameter used is logged in parameters_logged for reproducibility.
        """
        # Base accept rate from role
        if operator_profile.role == OperatorRole.EXPERT:
            base_rate = ASSUMPTION_ACCEPT_RATE_EXPERT
        elif operator_profile.role == OperatorRole.NOVICE:
            base_rate = ASSUMPTION_ACCEPT_RATE_NOVICE
        else:
            # Trainee: interpolate between novice and expert based on experience
            base_rate = ASSUMPTION_ACCEPT_RATE_NOVICE + (
                ASSUMPTION_ACCEPT_RATE_EXPERT - ASSUMPTION_ACCEPT_RATE_NOVICE
            ) * min(1.0, operator_profile.experience_years / 5.0)

        # Fatigue: multiply by factor for each 4-hour block elapsed
        fatigue_blocks = shift_hour // 4
        fatigue_mult = ASSUMPTION_FATIGUE_FACTOR**fatigue_blocks

        # False positive history reduces trust
        fp_penalty = prior_false_positives * ASSUMPTION_FALSE_POSITIVE_LEARNING

        # Explanation boost
        explanation_boost = ASSUMPTION_EXPLANATION_BOOST if alert.has_explanation else 0.0

        # Final accept probability, clamped to [0.05, 0.99]
        accept_prob = max(
            0.05, min(0.99, base_rate * fatigue_mult - fp_penalty + explanation_boost)
        )

        params = {
            "base_rate": base_rate,
            "fatigue_factor": ASSUMPTION_FATIGUE_FACTOR,
            "fatigue_blocks": float(fatigue_blocks),
            "fatigue_mult": round(fatigue_mult, 4),
            "fp_penalty": round(fp_penalty, 4),
            "explanation_boost": explanation_boost,
            "final_accept_prob": round(accept_prob, 4),
        }

        roll = self._rng.random()
        if roll < accept_prob:
            decision = "accept"
        elif roll < accept_prob + (1.0 - accept_prob) * 0.15:
            decision = "escalate"
        else:
            decision = "reject"

        return SimulationDecision(
            alert_id=alert.alert_id,
            decision=decision,
            simulated_confidence=round(accept_prob, 4),
            accept_probability_used=accept_prob,
            parameters_logged=params,
        )

    def simulate_session(
        self,
        alerts: list[AlertStub],
        operator_profile: OperatorProfile,
        shift_start_hour: int = 0,
    ) -> list[SimulationDecision]:
        """Simulate a full session of alert reviews for one operator."""
        decisions: list[SimulationDecision] = []
        fp_history: dict[str, int] = {}

        for i, alert in enumerate(alerts):
            shift_hour = shift_start_hour + (i // 5)  # ~5 alerts per hour
            fp_count = fp_history.get(alert.fault_type, 0)

            decision = self.simulate_operator_decision(
                alert=alert,
                operator_profile=operator_profile,
                shift_hour=shift_hour,
                prior_false_positives=fp_count,
            )
            decisions.append(decision)

            if not alert.is_true_positive and decision.decision == "reject":
                fp_history[alert.fault_type] = fp_count + 1

        return decisions

    def get_confidence_report(self) -> dict[str, dict[str, str]]:
        """
        Returns per-assumption confidence level and citation.
        Called before any HOG/TCS report is generated.
        Always emits a warning when low-confidence assumptions are active.
        """
        low = [k for k in LOW_CONFIDENCE_ASSUMPTIONS if k in ASSUMPTION_METADATA]
        if low:
            logger.warning(
                "oversight_simulation_low_confidence: %s — "
                "field study required before reporting to external stakeholders.",
                low,
            )
        return dict(ASSUMPTION_METADATA)


# ---------------------------------------------------------------------------
# Oversight report dataclass (used by ComplianceEngine)
# ---------------------------------------------------------------------------


@dataclass
class OversightReport:
    hir: float
    hog: float
    tcs: float
    ece: float
    simulation_confidence: str  # always 'LOW' until field study runs
    field_study_required: bool  # always True until field study runs
    low_confidence_assumptions: list[str]
    n_events: int
    n_reviewed: int
    n_overridden: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hir": round(self.hir, 4),
            "hog": round(self.hog, 4),
            "tcs": round(self.tcs, 4),
            "ece": round(self.ece, 4),
            "simulation_confidence": self.simulation_confidence,
            "field_study_required": self.field_study_required,
            "low_confidence_assumptions": self.low_confidence_assumptions,
            "n_events": self.n_events,
            "n_reviewed": self.n_reviewed,
            "n_overridden": self.n_overridden,
            "warning": (
                "These metrics are based on unvalidated simulation assumptions. "
                "Field study required before reporting to external stakeholders."
            ),
        }


def generate_oversight_report(
    hir: float,
    hog: float,
    tcs: float,
    ece: float,
    n_events: int,
    n_reviewed: int,
    n_overridden: int,
) -> OversightReport:
    """
    Build an OversightReport with simulation_confidence hardcoded to LOW.
    Emits warnings for low-confidence assumptions.
    """
    sim = OperatorSimulationModel()
    confidence_report = sim.get_confidence_report()

    low_confidence = [
        k for k, v in confidence_report.items() if v["confidence"] in ("LOW", "VERY LOW", "NONE")
    ]

    if low_confidence:
        logger.warning(
            "oversight_report_low_confidence_assumptions=%s field_study_required=True",
            low_confidence,
        )

    return OversightReport(
        hir=hir,
        hog=hog,
        tcs=tcs,
        ece=ece,
        simulation_confidence="LOW",  # hardcoded until field study runs
        field_study_required=True,  # hardcoded until field study runs
        low_confidence_assumptions=low_confidence,
        n_events=n_events,
        n_reviewed=n_reviewed,
        n_overridden=n_overridden,
    )
