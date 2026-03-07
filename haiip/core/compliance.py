"""EU AI Act Compliance Module.

The EU AI Act became fully applicable August 2026. Every AI system deployed
by EU companies must comply. HAIIP serves Nordic SMEs — compliance is not
optional, it is a legal and contractual requirement.

This module covers:
1. Risk classification (Annex III — Limited Risk)
2. Transparency requirements (Article 52)
3. Human oversight evidence (Article 14)
4. Audit trail generation (Article 12)
5. Incident detection and reporting (Article 73)
6. GDPR data minimisation alignment

Usage:
    engine = ComplianceEngine(system_name="HAIIP", tenant_id="sme-001")
    report = engine.generate_transparency_report(predictions)
    engine.log_decision(prediction, user_id="usr-001", explanation={...})
    risk = engine.classify_risk()
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Risk Classification ───────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    UNACCEPTABLE = "unacceptable"  # Article 5 — prohibited
    HIGH = "high"  # Annex III — strict requirements
    LIMITED = "limited"  # Article 52 — transparency obligations
    MINIMAL = "minimal"  # No specific obligations


@dataclass
class RiskAssessment:
    system_name: str
    risk_level: RiskLevel
    justification: str
    applicable_articles: list[str]
    transparency_required: bool
    human_oversight_required: bool
    conformity_assessment_required: bool
    assessed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Transparency Declaration ──────────────────────────────────────────────────


@dataclass
class TransparencyReport:
    """Article 52 transparency report — must be available to end users."""

    system_name: str
    tenant_id: str
    report_period_start: str
    report_period_end: str
    total_decisions: int
    human_reviewed_count: int
    human_review_rate: float
    anomaly_rate: float
    average_confidence: float
    model_types_used: list[str]
    training_datasets: list[str]
    data_sources: list[str]
    limitations: list[str]
    human_oversight_mechanism: str
    complaint_procedure: str
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return f"""# EU AI Act Transparency Report
## {self.system_name} — Tenant: {self.tenant_id}
**Report period**: {self.report_period_start[:10]} to {self.report_period_end[:10]}
**Generated**: {self.generated_at[:10]}
**System version**: {self.version}

## AI System Description
HAIIP (Human-Aligned Industrial Intelligence Platform) is an AI-powered predictive
maintenance and anomaly detection system for industrial SMEs. It analyses sensor data
from manufacturing machines to predict failures and detect anomalies.

**Risk classification**: Limited Risk (Article 52, EU AI Act)

## Decision Statistics
| Metric | Value |
|--------|-------|
| Total AI decisions | {self.total_decisions:,} |
| Human-reviewed decisions | {self.human_reviewed_count:,} |
| Human review rate | {self.human_review_rate * 100:.1f}% |
| Anomaly detection rate | {self.anomaly_rate * 100:.1f}% |
| Average model confidence | {self.average_confidence * 100:.1f}% |

## Model Information
**Model types**: {", ".join(self.model_types_used)}
**Training datasets**: {", ".join(self.training_datasets)}
**Data sources**: {", ".join(self.data_sources)}

## Limitations
{chr(10).join(f"- {item}" for item in self.limitations)}

## Human Oversight
{self.human_oversight_mechanism}

## Complaints
{self.complaint_procedure}
"""


# ── Audit Event ───────────────────────────────────────────────────────────────


@dataclass
class ComplianceEvent:
    """Single auditable event — Article 12 record keeping."""

    event_id: str
    tenant_id: str
    event_type: str  # "prediction", "human_override", "alert", "retrain"
    resource_id: str | None
    user_id: str | None
    input_hash: str  # SHA-256 of input features (privacy-preserving)
    output_label: str
    confidence: float
    human_reviewed: bool
    explanation_available: bool
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Compliance Engine ─────────────────────────────────────────────────────────


class ComplianceEngine:
    """Central EU AI Act compliance engine for HAIIP.

    Responsibilities:
    1. Classify system risk level
    2. Log every AI decision with privacy-preserving input hash
    3. Generate transparency reports on demand
    4. Detect incidents (low confidence, drift, systematic errors)
    5. Verify GDPR data minimisation
    """

    # HAIIP-specific compliance constants
    TRAINING_DATASETS = [
        "AI4I 2020 Predictive Maintenance (UCI ML Repository, CC BY 4.0)",
        "NASA CMAPSS Turbofan Engine Degradation (Public Domain)",
        "CWRU Bearing Dataset (Public Domain)",
    ]

    SYSTEM_LIMITATIONS = [
        "Model accuracy may degrade if production conditions drift significantly from training data.",
        "Anomaly detection is unsupervised — false positive rate depends on contamination setting.",
        "RUL prediction accuracy decreases beyond the training distribution of tool wear cycles.",
        "The system requires human review for all critical (severity > 0.7) anomaly detections.",
        "Models trained on European SME equipment may not generalise to non-European machinery.",
        "Real-time OPC UA/MQTT latency may delay anomaly detection by up to 5 seconds.",
    ]

    HUMAN_OVERSIGHT = (
        "All predictions are presented to human operators via the HAIIP dashboard. "
        "Operators can accept, reject, or override any AI decision. "
        "Feedback is logged and used for model retraining. "
        "Critical alerts require explicit human acknowledgement before the system "
        "resets the alert state. No automated actuator control — human confirmation required."
    )

    COMPLAINT_PROCEDURE = (
        "Users may submit complaints about AI decisions via: "
        "(1) The HAIIP feedback interface (in-app), "
        "(2) Email to the designated AI system contact, "
        "(3) The national data protection authority (e.g., Tietosuojavaltuutettu in Finland). "
        "All complaints are investigated within 30 working days."
    )

    def __init__(
        self,
        system_name: str = "HAIIP",
        tenant_id: str = "default",
        min_confidence_threshold: float = 0.6,
        incident_low_confidence_rate: float = 0.2,
    ) -> None:
        self.system_name = system_name
        self.tenant_id = tenant_id
        self.min_confidence_threshold = min_confidence_threshold
        self.incident_low_confidence_rate = incident_low_confidence_rate
        self._events: list[ComplianceEvent] = []

    # ── Risk classification ───────────────────────────────────────────────────

    def classify_risk(self) -> RiskAssessment:
        """Classify HAIIP under EU AI Act risk framework.

        HAIIP is LIMITED RISK:
        - Not used in high-risk domains (Article 6 / Annex III high-risk list)
        - Interacts with humans (operators) → transparency obligation (Article 52)
        - No autonomous control of safety-critical actuators
        - Human oversight maintained throughout
        """
        return RiskAssessment(
            system_name=self.system_name,
            risk_level=RiskLevel.LIMITED,
            justification=(
                "HAIIP provides decision support for industrial maintenance — "
                "it does not autonomously control machines or make safety-critical decisions. "
                "Human operators review all alerts and predictions. "
                "Classified as Limited Risk under Article 52 EU AI Act."
            ),
            applicable_articles=[
                "Article 52 — Transparency obligations for certain AI systems",
                "Article 14 — Human oversight",
                "Article 12 — Record keeping",
                "Article 13 — Transparency and provision of information",
                "Recital 47 — Industrial AI systems",
            ],
            transparency_required=True,
            human_oversight_required=True,
            conformity_assessment_required=False,
        )

    # ── Decision logging ──────────────────────────────────────────────────────

    def log_decision(
        self,
        prediction_id: str,
        input_features: list[float] | dict[str, Any],
        output_label: str,
        confidence: float,
        human_reviewed: bool = False,
        user_id: str | None = None,
        explanation: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComplianceEvent:
        """Record an AI decision in the compliance audit trail.

        Input features are hashed (SHA-256) — the raw sensor values are NOT
        stored in the compliance log (GDPR data minimisation, Article 5(1)(c)).
        """
        input_data = (
            json.dumps(input_features, sort_keys=True)
            if isinstance(input_features, dict)
            else json.dumps(sorted(input_features))
        )
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()

        event = ComplianceEvent(
            event_id=f"evt-{len(self._events):06d}",
            tenant_id=self.tenant_id,
            event_type="prediction",
            resource_id=prediction_id,
            user_id=user_id,
            input_hash=input_hash,
            output_label=output_label,
            confidence=confidence,
            human_reviewed=human_reviewed,
            explanation_available=explanation is not None,
            metadata=metadata or {},
        )
        self._events.append(event)

        # Warn if confidence below threshold
        if confidence < self.min_confidence_threshold:
            logger.warning(
                "compliance.low_confidence: prediction=%s confidence=%.3f threshold=%.3f",
                prediction_id,
                confidence,
                self.min_confidence_threshold,
            )

        return event

    def log_human_override(
        self,
        prediction_id: str,
        original_label: str,
        corrected_label: str,
        user_id: str,
        reason: str | None = None,
    ) -> ComplianceEvent:
        """Record a human operator overriding an AI decision (Article 14)."""
        event = ComplianceEvent(
            event_id=f"evt-{len(self._events):06d}",
            tenant_id=self.tenant_id,
            event_type="human_override",
            resource_id=prediction_id,
            user_id=user_id,
            input_hash="",
            output_label=corrected_label,
            confidence=1.0,  # human decision = 100% confidence
            human_reviewed=True,
            explanation_available=reason is not None,
            metadata={
                "original_label": original_label,
                "corrected_label": corrected_label,
                "reason": reason,
            },
        )
        self._events.append(event)
        logger.info(
            "compliance.human_override: prediction=%s %s→%s by=%s",
            prediction_id,
            original_label,
            corrected_label,
            user_id,
        )
        return event

    # ── Incident detection ────────────────────────────────────────────────────

    def detect_incidents(self) -> list[dict[str, Any]]:
        """Scan recent events for compliance incidents.

        Incidents:
        - Low confidence rate > threshold
        - No human reviews in last N decisions
        - Systematic label bias (same label > 90% of time)
        """
        incidents: list[dict[str, Any]] = []
        if not self._events:
            return incidents

        recent = self._events[-100:]

        # Low confidence incident
        low_conf = [e for e in recent if e.confidence < self.min_confidence_threshold]
        low_conf_rate = len(low_conf) / len(recent)
        if low_conf_rate > self.incident_low_confidence_rate:
            incidents.append(
                {
                    "type": "low_confidence",
                    "severity": "high",
                    "description": (
                        f"Low confidence rate {low_conf_rate * 100:.1f}% exceeds "
                        f"threshold {self.incident_low_confidence_rate * 100:.1f}%"
                    ),
                    "count": len(low_conf),
                    "recommendation": "Trigger model retraining. Check for distribution shift.",
                }
            )

        # Human review gap
        unreviewed = [e for e in recent if not e.human_reviewed]
        if len(unreviewed) > 50:
            incidents.append(
                {
                    "type": "human_review_gap",
                    "severity": "medium",
                    "description": f"{len(unreviewed)} decisions in last 100 not human-reviewed.",
                    "recommendation": "Increase operator review frequency. Check dashboard engagement.",
                }
            )

        # Label bias
        labels = [e.output_label for e in recent]
        if labels:
            most_common = max(set(labels), key=labels.count)
            bias_rate = labels.count(most_common) / len(labels)
            if bias_rate > 0.95:
                incidents.append(
                    {
                        "type": "label_bias",
                        "severity": "low",
                        "description": (
                            f"Label '{most_common}' appears in {bias_rate * 100:.1f}% of recent decisions."
                        ),
                        "recommendation": "Review class imbalance. Consider rebalancing training data.",
                    }
                )

        return incidents

    # ── Transparency report ───────────────────────────────────────────────────

    def generate_transparency_report(
        self,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> TransparencyReport:
        """Generate Article 52 transparency report for the current tenant."""
        now = datetime.now(UTC)
        start = period_start or datetime(now.year, now.month, 1, tzinfo=UTC)
        end = period_end or now

        events_in_period = [
            e for e in self._events if start.isoformat() <= e.timestamp <= end.isoformat()
        ]

        total = len(events_in_period)
        reviewed = sum(1 for e in events_in_period if e.human_reviewed)
        anomalies = sum(1 for e in events_in_period if e.output_label == "anomaly")
        avg_conf = sum(e.confidence for e in events_in_period) / total if total > 0 else 0.0

        model_types = list(
            {e.metadata.get("model_type", "anomaly_detection") for e in events_in_period}
        ) or ["anomaly_detection", "predictive_maintenance"]

        return TransparencyReport(
            system_name=self.system_name,
            tenant_id=self.tenant_id,
            report_period_start=start.isoformat(),
            report_period_end=end.isoformat(),
            total_decisions=total,
            human_reviewed_count=reviewed,
            human_review_rate=reviewed / total if total > 0 else 0.0,
            anomaly_rate=anomalies / total if total > 0 else 0.0,
            average_confidence=round(avg_conf, 4),
            model_types_used=model_types,
            training_datasets=self.TRAINING_DATASETS,
            data_sources=["OPC UA sensors", "MQTT IoT gateway", "HAIIP simulator"],
            limitations=self.SYSTEM_LIMITATIONS,
            human_oversight_mechanism=self.HUMAN_OVERSIGHT,
            complaint_procedure=self.COMPLAINT_PROCEDURE,
        )

    # ── GDPR helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def hash_personal_data(data: str) -> str:
        """One-way hash of personal/sensitive data (GDPR Art. 25 — privacy by design)."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def anonymise_features(features: dict[str, float]) -> dict[str, str]:
        """Replace raw sensor values with bucketed ranges (k-anonymity)."""
        buckets: dict[str, str] = {}
        for name, value in features.items():
            # Round to 2 significant figures
            magnitude = 10 ** (len(str(int(abs(value) + 1))) - 2) if value != 0 else 1
            bucket_val = round(value / magnitude) * magnitude
            buckets[name] = f"~{bucket_val}"
        return buckets

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def human_review_rate(self) -> float:
        if not self._events:
            return 0.0
        return sum(1 for e in self._events if e.human_reviewed) / len(self._events)

    def get_events(self) -> list[ComplianceEvent]:
        return list(self._events)

    def clear_events(self) -> None:
        """Clear in-memory events — called after persisting to DB."""
        self._events.clear()
