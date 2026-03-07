"""IEC 61508 SIL-2 safety alignment layer for HAIIP.

Implements:
- Safety Integrity Level (SIL) checks aligned to IEC 61508-3
- Failure Mode and Effects Analysis (FMEA) helper
- Diagnostic Coverage (DC) metric computation
- Safe-state enforcement: if uncertainty > threshold, escalate to human
- Probabilistic Failure on Demand (PFD) estimation

References:
    IEC 61508:2010 Functional Safety of E/E/EP Safety-related Systems
    IEC 62061:2021 Safety of Machinery
    EN ISO 13849-1:2015 Safety of Machinery — Control Systems

Design principle: AI recommendations are ADVISORY only at SIL-2.
A human operator MUST confirm any REPAIR_NOW decision on safety-critical assets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# ── SIL definitions (IEC 61508 Table 2) ──────────────────────────────────────


class SILLevel(IntEnum):
    """Safety Integrity Level — higher = stricter."""

    SIL_0 = 0  # non-safety-critical
    SIL_1 = 1  # PFD_avg: 10e-2 to 10e-1
    SIL_2 = 2  # PFD_avg: 10e-3 to 10e-2  ← HAIIP target
    SIL_3 = 3  # PFD_avg: 10e-4 to 10e-3
    SIL_4 = 4  # PFD_avg: 10e-5 to 10e-4


# SIL_2 PFD bounds per IEC 61508 Table 2
SIL_2_PFD_MAX = 1e-2
SIL_2_PFD_MIN = 1e-3

# Confidence threshold below which AI MUST escalate to human
ESCALATION_CONFIDENCE_THRESHOLD = 0.70

# AI is advisory only — humans confirm safety-critical actions
AI_IS_ADVISORY = True


@dataclass
class FMEAEntry:
    """Single row of a Failure Mode and Effects Analysis table."""

    component: str
    failure_mode: str
    effect: str
    severity: int  # 1–10
    occurrence: int  # 1–10
    detectability: int  # 1–10 (10 = hard to detect)

    @property
    def rpn(self) -> int:
        """Risk Priority Number = S × O × D. Range 1–1000."""
        return self.severity * self.occurrence * self.detectability

    @property
    def sil_required(self) -> SILLevel:
        """Derive minimum required SIL from RPN (simplified mapping)."""
        if self.rpn >= 500:
            return SILLevel.SIL_3
        if self.rpn >= 200:
            return SILLevel.SIL_2
        if self.rpn >= 80:
            return SILLevel.SIL_1
        return SILLevel.SIL_0


@dataclass
class SafetyDecision:
    """Result of a safety check on an AI prediction."""

    original_label: str
    original_confidence: float
    safe_label: str  # may differ if overridden
    safe_confidence: float
    escalate_to_human: bool
    sil_level: SILLevel
    reason: str
    timestamp: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticCoverage:
    """IEC 61508 Diagnostic Coverage computation."""

    total_failure_rate: float  # λ_total (per hour)
    detected_failure_rate: float  # λ_detected (per hour)

    @property
    def dc(self) -> float:
        """DC = λ_detected / λ_total. Range [0, 1]."""
        if self.total_failure_rate <= 0:
            return 0.0
        return min(self.detected_failure_rate / self.total_failure_rate, 1.0)

    @property
    def dc_category(self) -> str:
        """IEC 61508 Table A.14 DC categories."""
        d = self.dc
        if d < 0.60:
            return "None (<60%)"
        if d < 0.90:
            return "Low (60–90%)"
        if d < 0.99:
            return "Medium (90–99%)"
        return "High (≥99%)"


class SafetyLayer:
    """Wraps AI predictions with IEC 61508 SIL-2 safety checks.

    Every prediction passes through this layer before reaching the operator.
    If confidence < threshold or anomaly_score > safe_limit, escalation fires.

    Thread-safe: all methods are stateless (pure functions over inputs).
    """

    def __init__(
        self,
        sil_level: SILLevel = SILLevel.SIL_2,
        confidence_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
        anomaly_score_limit: float = 0.80,
        require_human_for_repair: bool = True,
    ) -> None:
        self.sil_level = sil_level
        self.confidence_threshold = confidence_threshold
        self.anomaly_score_limit = anomaly_score_limit
        self.require_human_for_repair = require_human_for_repair

    def check(
        self,
        prediction: dict[str, Any],
        economic_action: str | None = None,
    ) -> SafetyDecision:
        """Apply safety checks to an AI prediction.

        Args:
            prediction: dict from AnomalyDetector.predict() or
                        MaintenancePredictor.predict()
            economic_action: optional EconomicAI action string
                             (REPAIR_NOW / SCHEDULE / MONITOR / IGNORE)

        Returns:
            SafetyDecision with safe_label and escalate_to_human flag.
        """
        label = prediction.get("label", "normal")
        confidence = float(prediction.get("confidence", 0.5))
        anomaly_score = float(prediction.get("anomaly_score", 0.0))

        escalate = False
        reasons: list[str] = []

        # Rule 1: Low confidence → escalate
        if confidence < self.confidence_threshold:
            escalate = True
            reasons.append(f"confidence={confidence:.3f} < threshold={self.confidence_threshold}")

        # Rule 2: High anomaly score → escalate regardless of label
        if anomaly_score > self.anomaly_score_limit:
            escalate = True
            reasons.append(f"anomaly_score={anomaly_score:.3f} > limit={self.anomaly_score_limit}")

        # Rule 3: REPAIR_NOW at SIL-2 always requires human confirmation
        if economic_action == "REPAIR_NOW" and self.require_human_for_repair:
            escalate = True
            reasons.append("REPAIR_NOW requires human confirmation at SIL-2")

        # Rule 4: Anomaly predicted with low confidence → treat as anomaly (fail-safe)
        safe_label = label
        if label == "normal" and anomaly_score > 0.60:
            safe_label = "anomaly"
            reasons.append(f"fail-safe override: score={anomaly_score:.3f} despite label=normal")

        reason = "; ".join(reasons) if reasons else "all safety checks passed"

        decision = SafetyDecision(
            original_label=label,
            original_confidence=confidence,
            safe_label=safe_label,
            safe_confidence=confidence,
            escalate_to_human=escalate,
            sil_level=self.sil_level,
            reason=reason,
        )

        if escalate:
            logger.warning(
                "SafetyLayer escalation: label=%s → safe=%s | %s",
                label,
                safe_label,
                reason,
            )
        else:
            logger.debug("SafetyLayer OK: label=%s confidence=%.3f", label, confidence)

        return decision

    def check_batch(
        self,
        predictions: list[dict[str, Any]],
        economic_actions: list[str | None] | None = None,
    ) -> list[SafetyDecision]:
        """Apply safety checks to a batch of predictions."""
        actions = economic_actions or [None] * len(predictions)
        return [self.check(p, a) for p, a in zip(predictions, actions)]

    # ── FMEA helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_fmea(entries: list[FMEAEntry]) -> dict[str, Any]:
        """Summarise FMEA table: top risks, overall SIL requirement."""
        if not entries:
            return {"entries": [], "max_rpn": 0, "required_sil": SILLevel.SIL_0}

        sorted_entries = sorted(entries, key=lambda e: e.rpn, reverse=True)
        max_rpn = sorted_entries[0].rpn
        required_sil = max(e.sil_required for e in entries)

        return {
            "entries": [
                {
                    "component": e.component,
                    "failure_mode": e.failure_mode,
                    "effect": e.effect,
                    "rpn": e.rpn,
                    "sil_required": e.sil_required.name,
                }
                for e in sorted_entries
            ],
            "max_rpn": max_rpn,
            "required_sil": required_sil,
            "required_sil_name": required_sil.name,
        }

    # ── PFD estimation ────────────────────────────────────────────────────────

    @staticmethod
    def estimate_pfd(
        lambda_d: float,
        proof_test_interval_h: float,
        beta: float = 0.02,
    ) -> float:
        """Simplified PFD_avg formula for 1oo1 architecture (IEC 61508-6 B.3.1).

        Args:
            lambda_d: dangerous failure rate (per hour)
            proof_test_interval_h: test interval in hours (e.g. 8760 = 1 year)
            beta: common cause failure fraction (typically 0.02–0.10)

        Returns:
            PFD_avg estimate
        """
        if lambda_d <= 0 or proof_test_interval_h <= 0:
            return 0.0
        # 1oo1: PFD_avg ≈ λ_D × T_I / 2
        pfd = lambda_d * proof_test_interval_h / 2.0
        # Common cause adjustment
        pfd_cc = beta * lambda_d * proof_test_interval_h / 2.0
        return min(pfd + pfd_cc, 1.0)

    @staticmethod
    def sil_achieved(pfd: float) -> SILLevel:
        """Return SIL level achieved given a PFD_avg."""
        if pfd < 1e-4:
            return SILLevel.SIL_4
        if pfd < 1e-3:
            return SILLevel.SIL_3
        if pfd < 1e-2:
            return SILLevel.SIL_2
        if pfd < 1e-1:
            return SILLevel.SIL_1
        return SILLevel.SIL_0


# ── Predefined HAIIP FMEA table (CNC machine context) ────────────────────────

HAIIP_FMEA: list[FMEAEntry] = [
    FMEAEntry(
        component="AnomalyDetector",
        failure_mode="False negative (missed anomaly)",
        effect="Undetected machine failure → unplanned downtime",
        severity=8,
        occurrence=3,
        detectability=4,
    ),
    FMEAEntry(
        component="AnomalyDetector",
        failure_mode="False positive (false alarm)",
        effect="Unnecessary maintenance stop → lost production",
        severity=4,
        occurrence=4,
        detectability=2,
    ),
    FMEAEntry(
        component="EconomicAI",
        failure_mode="REPAIR_NOW issued incorrectly",
        effect="Machine stopped unnecessarily — production loss",
        severity=5,
        occurrence=2,
        detectability=3,
    ),
    FMEAEntry(
        component="SensorPipeline",
        failure_mode="Sensor dropout / stale data",
        effect="Model operates on stale inputs → degraded accuracy",
        severity=7,
        occurrence=3,
        detectability=3,
    ),
    FMEAEntry(
        component="RAGEngine",
        failure_mode="Hallucinated maintenance instruction",
        effect="Wrong repair procedure followed by technician",
        severity=9,
        occurrence=2,
        detectability=6,
    ),
    FMEAEntry(
        component="DriftDetector",
        failure_mode="Missed distribution shift",
        effect="Model predictions unreliable — undetected degradation",
        severity=8,
        occurrence=2,
        detectability=5,
    ),
]
