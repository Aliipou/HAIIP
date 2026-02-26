"""Tests for EU AI Act compliance engine.

Covers: risk classification, decision logging, incident detection,
transparency reports, GDPR helpers, audit event structure.

EU AI Act references: Article 12, 13, 14, 52, Annex III.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from haiip.core.compliance import (
    ComplianceEngine,
    ComplianceEvent,
    RiskAssessment,
    RiskLevel,
    TransparencyReport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> ComplianceEngine:
    return ComplianceEngine(system_name="HAIIP-TEST", tenant_id="sme-test-001")


@pytest.fixture
def engine_low_threshold() -> ComplianceEngine:
    """Engine with tight incident thresholds for testing detection."""
    return ComplianceEngine(
        system_name="HAIIP-TEST",
        tenant_id="sme-low-threshold",
        min_confidence_threshold=0.8,
        incident_low_confidence_rate=0.05,
    )


# ── Risk classification ───────────────────────────────────────────────────────

class TestRiskClassification:
    def test_classify_risk_returns_assessment(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        assert isinstance(assessment, RiskAssessment)

    def test_haiip_is_limited_risk(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        assert assessment.risk_level == RiskLevel.LIMITED

    def test_not_high_risk(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        assert assessment.risk_level != RiskLevel.HIGH
        assert assessment.risk_level != RiskLevel.UNACCEPTABLE

    def test_transparency_required(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        assert assessment.transparency_required is True

    def test_human_oversight_required(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        assert assessment.human_oversight_required is True

    def test_no_conformity_assessment_needed(self, engine: ComplianceEngine):
        """Limited risk does not require formal conformity assessment."""
        assessment = engine.classify_risk()
        assert assessment.conformity_assessment_required is False

    def test_applicable_articles_include_52(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        articles_text = " ".join(assessment.applicable_articles)
        assert "52" in articles_text

    def test_applicable_articles_include_14(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        articles_text = " ".join(assessment.applicable_articles)
        assert "14" in articles_text

    def test_applicable_articles_include_12(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        articles_text = " ".join(assessment.applicable_articles)
        assert "12" in articles_text

    def test_assessment_has_timestamp(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        assert assessment.assessed_at
        # Must be parseable ISO format
        datetime.fromisoformat(assessment.assessed_at)

    def test_assessment_to_dict(self, engine: ComplianceEngine):
        assessment = engine.classify_risk()
        d = assessment.to_dict()
        assert isinstance(d, dict)
        assert "risk_level" in d
        assert "justification" in d

    def test_risk_level_enum_values(self):
        assert RiskLevel.UNACCEPTABLE == "unacceptable"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.LIMITED == "limited"
        assert RiskLevel.MINIMAL == "minimal"


# ── Decision logging ──────────────────────────────────────────────────────────

class TestDecisionLogging:
    def test_log_decision_returns_event(self, engine: ComplianceEngine):
        event = engine.log_decision(
            prediction_id="pred-001",
            input_features=[1.0, 2.0, 3.0],
            output_label="no_failure",
            confidence=0.95,
        )
        assert isinstance(event, ComplianceEvent)

    def test_event_has_correct_label(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-002", [1.0], "anomaly", 0.87)
        assert event.output_label == "anomaly"

    def test_event_has_correct_confidence(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-003", [1.0], "no_failure", 0.72)
        assert abs(event.confidence - 0.72) < 1e-6

    def test_input_is_hashed_not_stored_raw(self, engine: ComplianceEngine):
        """GDPR data minimisation — raw sensor values must not appear in event."""
        features = [42.5, 1337.0, 0.001]
        event = engine.log_decision("pred-004", features, "no_failure", 0.9)
        # SHA-256 hex is 64 chars
        assert len(event.input_hash) == 64
        # Raw values not in input_hash field
        assert "42.5" not in event.input_hash

    def test_input_hash_is_sha256(self, engine: ComplianceEngine):
        features = [1.0, 2.0]
        event = engine.log_decision("pred-005", features, "no_failure", 0.9)
        # Recompute expected hash
        expected = hashlib.sha256(json.dumps(sorted(features)).encode()).hexdigest()
        assert event.input_hash == expected

    def test_dict_features_hashed_deterministically(self, engine: ComplianceEngine):
        features = {"temp": 60.0, "vibration": 0.3}
        e1 = engine.log_decision("pred-006a", features, "normal", 0.88)
        e2 = engine.log_decision("pred-006b", features, "normal", 0.88)
        assert e1.input_hash == e2.input_hash

    def test_event_type_is_prediction(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-007", [1.0], "no_failure", 0.9)
        assert event.event_type == "prediction"

    def test_event_tenant_id(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-008", [1.0], "no_failure", 0.9)
        assert event.tenant_id == "sme-test-001"

    def test_human_reviewed_default_false(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-009", [1.0], "no_failure", 0.9)
        assert event.human_reviewed is False

    def test_human_reviewed_can_be_true(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-010", [1.0], "no_failure", 0.9, human_reviewed=True)
        assert event.human_reviewed is True

    def test_explanation_available_flag(self, engine: ComplianceEngine):
        event_no_exp = engine.log_decision("pred-011", [1.0], "no_failure", 0.9)
        event_with_exp = engine.log_decision(
            "pred-012", [1.0], "no_failure", 0.9, explanation={"feat1": 0.5}
        )
        assert event_no_exp.explanation_available is False
        assert event_with_exp.explanation_available is True

    def test_event_ids_increment(self, engine: ComplianceEngine):
        e1 = engine.log_decision("pred-013", [1.0], "no_failure", 0.9)
        e2 = engine.log_decision("pred-014", [1.0], "no_failure", 0.9)
        assert e1.event_id != e2.event_id

    def test_event_count_increments(self, engine: ComplianceEngine):
        assert engine.event_count == 0
        engine.log_decision("pred-015", [1.0], "no_failure", 0.9)
        assert engine.event_count == 1
        engine.log_decision("pred-016", [1.0], "no_failure", 0.9)
        assert engine.event_count == 2

    def test_event_has_timestamp(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-017", [1.0], "no_failure", 0.9)
        assert event.timestamp
        datetime.fromisoformat(event.timestamp)

    def test_event_to_dict(self, engine: ComplianceEngine):
        event = engine.log_decision("pred-018", [1.0], "no_failure", 0.9)
        d = event.to_dict()
        assert isinstance(d, dict)
        assert "input_hash" in d
        assert "confidence" in d

    def test_metadata_stored(self, engine: ComplianceEngine):
        meta = {"model_type": "isolation_forest", "version": "1.0"}
        event = engine.log_decision("pred-019", [1.0], "anomaly", 0.75, metadata=meta)
        assert event.metadata["model_type"] == "isolation_forest"


# ── Human override logging ────────────────────────────────────────────────────

class TestHumanOverrideLogging:
    def test_log_human_override_returns_event(self, engine: ComplianceEngine):
        event = engine.log_human_override(
            prediction_id="pred-100",
            original_label="no_failure",
            corrected_label="TWF",
            user_id="usr-001",
        )
        assert isinstance(event, ComplianceEvent)

    def test_override_event_type(self, engine: ComplianceEngine):
        event = engine.log_human_override("pred-101", "no_failure", "HDF", "usr-002")
        assert event.event_type == "human_override"

    def test_override_confidence_is_one(self, engine: ComplianceEngine):
        """Human decisions have full confidence."""
        event = engine.log_human_override("pred-102", "anomaly", "no_failure", "usr-003")
        assert event.confidence == 1.0

    def test_override_human_reviewed_true(self, engine: ComplianceEngine):
        event = engine.log_human_override("pred-103", "anomaly", "no_failure", "usr-004")
        assert event.human_reviewed is True

    def test_override_stores_original_label(self, engine: ComplianceEngine):
        event = engine.log_human_override("pred-104", "TWF", "no_failure", "usr-005")
        assert event.metadata["original_label"] == "TWF"

    def test_override_stores_corrected_label(self, engine: ComplianceEngine):
        event = engine.log_human_override("pred-105", "no_failure", "PWF", "usr-006")
        assert event.metadata["corrected_label"] == "PWF"

    def test_override_with_reason(self, engine: ComplianceEngine):
        event = engine.log_human_override(
            "pred-106", "no_failure", "OSF", "usr-007", reason="Vibration spike confirmed"
        )
        assert event.explanation_available is True
        assert "Vibration spike" in event.metadata["reason"]

    def test_override_increments_count(self, engine: ComplianceEngine):
        before = engine.event_count
        engine.log_human_override("pred-107", "no_failure", "RNF", "usr-008")
        assert engine.event_count == before + 1


# ── Incident detection ────────────────────────────────────────────────────────

class TestIncidentDetection:
    def test_no_incidents_with_no_events(self, engine: ComplianceEngine):
        assert engine.detect_incidents() == []

    def test_no_incidents_with_good_data(self, engine: ComplianceEngine):
        for i in range(20):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.95)
        incidents = engine.detect_incidents()
        assert incidents == []

    def test_low_confidence_incident_detected(self, engine_low_threshold: ComplianceEngine):
        # Make 50% of events low confidence
        for i in range(10):
            engine_low_threshold.log_decision(f"p{i}", [1.0], "anomaly", 0.3)
        incidents = engine_low_threshold.detect_incidents()
        types = [inc["type"] for inc in incidents]
        assert "low_confidence" in types

    def test_low_confidence_incident_severity(self, engine_low_threshold: ComplianceEngine):
        for i in range(20):
            engine_low_threshold.log_decision(f"p{i}", [1.0], "anomaly", 0.2)
        incidents = engine_low_threshold.detect_incidents()
        lc = next(inc for inc in incidents if inc["type"] == "low_confidence")
        assert lc["severity"] == "high"

    def test_human_review_gap_detected(self, engine: ComplianceEngine):
        # 60 decisions, none human-reviewed
        for i in range(60):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9)
        incidents = engine.detect_incidents()
        types = [inc["type"] for inc in incidents]
        assert "human_review_gap" in types

    def test_human_review_gap_threshold_is_50(self, engine: ComplianceEngine):
        # 50 unreviewed — right at threshold, should NOT trigger
        for i in range(50):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9)
        incidents = engine.detect_incidents()
        types = [inc["type"] for inc in incidents]
        # <= 50 does not trigger gap (> 50)
        assert "human_review_gap" not in types

    def test_label_bias_detected(self, engine: ComplianceEngine):
        # 100% same label
        for i in range(20):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9)
        incidents = engine.detect_incidents()
        types = [inc["type"] for inc in incidents]
        assert "label_bias" in types

    def test_label_bias_severity_is_low(self, engine: ComplianceEngine):
        for i in range(20):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9)
        incidents = engine.detect_incidents()
        lb = next((inc for inc in incidents if inc["type"] == "label_bias"), None)
        assert lb is not None
        assert lb["severity"] == "low"

    def test_incident_has_recommendation(self, engine_low_threshold: ComplianceEngine):
        for i in range(20):
            engine_low_threshold.log_decision(f"p{i}", [1.0], "anomaly", 0.2)
        incidents = engine_low_threshold.detect_incidents()
        for inc in incidents:
            assert "recommendation" in inc


# ── Transparency report ───────────────────────────────────────────────────────

class TestTransparencyReport:
    def test_empty_report_generated(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert isinstance(report, TransparencyReport)

    def test_report_tenant_id(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert report.tenant_id == "sme-test-001"

    def test_report_zero_decisions_when_empty(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert report.total_decisions == 0
        assert report.human_review_rate == 0.0
        assert report.anomaly_rate == 0.0

    def test_report_counts_decisions(self, engine: ComplianceEngine):
        for i in range(10):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9)
        report = engine.generate_transparency_report()
        assert report.total_decisions == 10

    def test_report_human_review_rate(self, engine: ComplianceEngine):
        for i in range(8):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9)
        for i in range(8, 10):
            engine.log_decision(f"p{i}", [1.0], "no_failure", 0.9, human_reviewed=True)
        report = engine.generate_transparency_report()
        assert abs(report.human_review_rate - 0.2) < 0.01

    def test_report_anomaly_rate(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "anomaly", 0.9)
        engine.log_decision("p2", [1.0], "no_failure", 0.9)
        engine.log_decision("p3", [1.0], "anomaly", 0.9)
        engine.log_decision("p4", [1.0], "no_failure", 0.9)
        report = engine.generate_transparency_report()
        assert abs(report.anomaly_rate - 0.5) < 0.01

    def test_report_average_confidence(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "no_failure", 0.8)
        engine.log_decision("p2", [1.0], "no_failure", 1.0)
        report = engine.generate_transparency_report()
        assert abs(report.average_confidence - 0.9) < 0.01

    def test_report_training_datasets_populated(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert len(report.training_datasets) >= 1
        assert any("AI4I" in ds or "NASA" in ds or "CWRU" in ds for ds in report.training_datasets)

    def test_report_limitations_populated(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert len(report.limitations) >= 3

    def test_report_human_oversight_text(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert len(report.human_oversight_mechanism) > 50

    def test_report_complaint_procedure(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        assert "30" in report.complaint_procedure  # 30 working days

    def test_report_to_dict(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "total_decisions" in d

    def test_report_to_markdown(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "no_failure", 0.9)
        md = engine.generate_transparency_report().to_markdown()
        assert "EU AI Act" in md
        assert "Limited Risk" in md
        assert "Article 52" in md

    def test_report_period_defaults_to_current_month(self, engine: ComplianceEngine):
        report = engine.generate_transparency_report()
        now = datetime.now(timezone.utc)
        start = datetime.fromisoformat(report.report_period_start)
        assert start.year == now.year
        assert start.month == now.month

    def test_model_types_extracted_from_metadata(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "no_failure", 0.9, metadata={"model_type": "isolation_forest"})
        engine.log_decision("p2", [1.0], "anomaly", 0.8, metadata={"model_type": "gradient_boosting"})
        report = engine.generate_transparency_report()
        assert "isolation_forest" in report.model_types_used or "gradient_boosting" in report.model_types_used


# ── GDPR helpers ──────────────────────────────────────────────────────────────

class TestGDPRHelpers:
    def test_hash_personal_data_returns_hex(self):
        h = ComplianceEngine.hash_personal_data("user@example.com")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_personal_data_deterministic(self):
        h1 = ComplianceEngine.hash_personal_data("test@test.fi")
        h2 = ComplianceEngine.hash_personal_data("test@test.fi")
        assert h1 == h2

    def test_hash_personal_data_different_inputs(self):
        h1 = ComplianceEngine.hash_personal_data("alice@test.fi")
        h2 = ComplianceEngine.hash_personal_data("bob@test.fi")
        assert h1 != h2

    def test_anonymise_features_returns_dict(self):
        features = {"temperature": 65.3, "vibration": 0.42}
        anon = ComplianceEngine.anonymise_features(features)
        assert isinstance(anon, dict)

    def test_anonymise_features_same_keys(self):
        features = {"temp": 65.0, "pressure": 1013.0}
        anon = ComplianceEngine.anonymise_features(features)
        assert set(anon.keys()) == set(features.keys())

    def test_anonymise_features_uses_tilde_prefix(self):
        features = {"temp": 100.0}
        anon = ComplianceEngine.anonymise_features(features)
        assert anon["temp"].startswith("~")

    def test_anonymise_features_no_raw_values(self):
        """Raw precision values should not appear verbatim."""
        features = {"temp": 65.789}
        anon = ComplianceEngine.anonymise_features(features)
        # Exact value 65.789 should not appear
        assert "65.789" not in anon["temp"]


# ── Engine properties ─────────────────────────────────────────────────────────

class TestEngineProperties:
    def test_event_count_starts_zero(self, engine: ComplianceEngine):
        assert engine.event_count == 0

    def test_human_review_rate_starts_zero(self, engine: ComplianceEngine):
        assert engine.human_review_rate == 0.0

    def test_human_review_rate_calculation(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "no_failure", 0.9, human_reviewed=True)
        engine.log_decision("p2", [1.0], "no_failure", 0.9)
        engine.log_decision("p3", [1.0], "no_failure", 0.9)
        assert abs(engine.human_review_rate - 1 / 3) < 0.01

    def test_get_events_returns_copy(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "no_failure", 0.9)
        events = engine.get_events()
        events.clear()
        assert engine.event_count == 1  # original not affected

    def test_clear_events(self, engine: ComplianceEngine):
        engine.log_decision("p1", [1.0], "no_failure", 0.9)
        engine.clear_events()
        assert engine.event_count == 0
