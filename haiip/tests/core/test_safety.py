"""Tests for IEC 61508 SIL-2 safety layer — 100% branch coverage."""

from __future__ import annotations

import pytest
from haiip.core.safety import (
    DiagnosticCoverage,
    FMEAEntry,
    HAIIP_FMEA,
    SafetyLayer,
    SafetyDecision,
    SILLevel,
    ESCALATION_CONFIDENCE_THRESHOLD,
    AI_IS_ADVISORY,
)


# ── SILLevel ─────────────────────────────────────────────────────────────────

class TestSILLevel:

    def test_sil_levels_ordered(self):
        assert SILLevel.SIL_0 < SILLevel.SIL_1 < SILLevel.SIL_2 < SILLevel.SIL_3 < SILLevel.SIL_4

    def test_sil2_is_haiip_target(self):
        assert SILLevel.SIL_2 == 2


# ── FMEAEntry ─────────────────────────────────────────────────────────────────

class TestFMEAEntry:

    def _entry(self, s=5, o=5, d=5):
        return FMEAEntry(
            component="Test",
            failure_mode="test fault",
            effect="test effect",
            severity=s, occurrence=o, detectability=d,
        )

    def test_rpn_calculation(self):
        e = self._entry(5, 4, 3)
        assert e.rpn == 60

    def test_rpn_max(self):
        e = self._entry(10, 10, 10)
        assert e.rpn == 1000

    def test_sil_required_0_low_rpn(self):
        e = self._entry(2, 2, 2)   # rpn=8
        assert e.sil_required == SILLevel.SIL_0

    def test_sil_required_1_mid_rpn(self):
        e = self._entry(4, 4, 5)   # rpn=80
        assert e.sil_required == SILLevel.SIL_1

    def test_sil_required_2_high_rpn(self):
        e = self._entry(5, 5, 9)   # rpn=225
        assert e.sil_required == SILLevel.SIL_2

    def test_sil_required_3_critical_rpn(self):
        e = self._entry(10, 10, 6)  # rpn=600
        assert e.sil_required == SILLevel.SIL_3

    def test_rpn_boundary_80(self):
        e = self._entry(4, 4, 5)   # rpn=80 exactly → SIL_1
        assert e.sil_required == SILLevel.SIL_1

    def test_rpn_boundary_200(self):
        e = self._entry(5, 5, 8)   # rpn=200 → SIL_2
        assert e.sil_required == SILLevel.SIL_2

    def test_rpn_boundary_500(self):
        e = self._entry(10, 10, 5)  # rpn=500 → SIL_3
        assert e.sil_required == SILLevel.SIL_3


# ── DiagnosticCoverage ────────────────────────────────────────────────────────

class TestDiagnosticCoverage:

    def test_dc_calculation(self):
        dc = DiagnosticCoverage(total_failure_rate=1.0, detected_failure_rate=0.95)
        assert abs(dc.dc - 0.95) < 1e-9

    def test_dc_zero_total(self):
        dc = DiagnosticCoverage(total_failure_rate=0.0, detected_failure_rate=0.5)
        assert dc.dc == 0.0

    def test_dc_capped_at_1(self):
        dc = DiagnosticCoverage(total_failure_rate=1.0, detected_failure_rate=2.0)
        assert dc.dc == 1.0

    def test_dc_none_category(self):
        dc = DiagnosticCoverage(total_failure_rate=1.0, detected_failure_rate=0.3)
        assert dc.dc_category.startswith("None")

    def test_dc_low_category(self):
        dc = DiagnosticCoverage(total_failure_rate=1.0, detected_failure_rate=0.75)
        assert "Low" in dc.dc_category

    def test_dc_medium_category(self):
        dc = DiagnosticCoverage(total_failure_rate=1.0, detected_failure_rate=0.95)
        assert "Medium" in dc.dc_category

    def test_dc_high_category(self):
        dc = DiagnosticCoverage(total_failure_rate=1.0, detected_failure_rate=0.995)
        assert "High" in dc.dc_category


# ── SafetyLayer ───────────────────────────────────────────────────────────────

class TestSafetyLayer:

    @pytest.fixture
    def layer(self):
        return SafetyLayer()

    def _prediction(self, label="normal", confidence=0.90, anomaly_score=0.1):
        return {"label": label, "confidence": confidence, "anomaly_score": anomaly_score}

    def test_normal_high_confidence_passes(self, layer):
        decision = layer.check(self._prediction())
        assert not decision.escalate_to_human
        assert decision.safe_label == "normal"

    def test_low_confidence_escalates(self, layer):
        decision = layer.check(self._prediction(confidence=0.50))
        assert decision.escalate_to_human
        assert "confidence" in decision.reason

    def test_high_anomaly_score_escalates(self, layer):
        decision = layer.check(self._prediction(anomaly_score=0.85))
        assert decision.escalate_to_human
        assert "anomaly_score" in decision.reason

    def test_repair_now_escalates(self, layer):
        decision = layer.check(self._prediction(), economic_action="REPAIR_NOW")
        assert decision.escalate_to_human
        assert "REPAIR_NOW" in decision.reason

    def test_schedule_does_not_escalate(self, layer):
        decision = layer.check(self._prediction(), economic_action="SCHEDULE")
        assert not decision.escalate_to_human

    def test_monitor_does_not_escalate(self, layer):
        decision = layer.check(self._prediction(), economic_action="MONITOR")
        assert not decision.escalate_to_human

    def test_failsafe_override_label(self, layer):
        """Normal label but high anomaly score → upgrade to anomaly (fail-safe)."""
        decision = layer.check(self._prediction(label="normal", anomaly_score=0.65))
        assert decision.safe_label == "anomaly"
        assert "fail-safe" in decision.reason

    def test_anomaly_label_stays_anomaly(self, layer):
        decision = layer.check(self._prediction(label="anomaly", confidence=0.90, anomaly_score=0.7))
        assert decision.safe_label == "anomaly"

    def test_decision_has_timestamp(self, layer):
        decision = layer.check(self._prediction())
        assert decision.timestamp > 0

    def test_sil_level_stored(self, layer):
        decision = layer.check(self._prediction())
        assert decision.sil_level == SILLevel.SIL_2

    def test_batch_check_length(self, layer):
        preds = [self._prediction() for _ in range(5)]
        decisions = layer.check_batch(preds)
        assert len(decisions) == 5

    def test_batch_check_with_actions(self, layer):
        preds = [self._prediction() for _ in range(3)]
        actions = ["REPAIR_NOW", "SCHEDULE", None]
        decisions = layer.check_batch(preds, actions)
        assert decisions[0].escalate_to_human is True
        assert decisions[1].escalate_to_human is False

    def test_batch_check_no_actions(self, layer):
        preds = [self._prediction() for _ in range(3)]
        decisions = layer.check_batch(preds)
        assert len(decisions) == 3

    def test_no_require_human_for_repair_now(self):
        layer = SafetyLayer(require_human_for_repair=False)
        decision = layer.check(
            {"label": "normal", "confidence": 0.95, "anomaly_score": 0.1},
            economic_action="REPAIR_NOW",
        )
        assert not decision.escalate_to_human

    def test_custom_thresholds(self):
        layer = SafetyLayer(confidence_threshold=0.99, anomaly_score_limit=0.99)
        decision = layer.check(self._prediction(confidence=0.95, anomaly_score=0.95))
        # Both below custom thresholds → should escalate on confidence
        assert decision.escalate_to_human

    def test_missing_prediction_keys_default(self, layer):
        """Partial dict doesn't crash — defaults apply."""
        decision = layer.check({})
        assert decision.original_label == "normal"
        assert decision.original_confidence == 0.5


# ── FMEA helpers ──────────────────────────────────────────────────────────────

class TestFMEAHelpers:

    def test_compute_fmea_empty(self):
        result = SafetyLayer.compute_fmea([])
        assert result["max_rpn"] == 0
        assert result["required_sil"] == SILLevel.SIL_0

    def test_compute_fmea_sorts_by_rpn(self):
        entries = [
            FMEAEntry("A", "fault1", "effect1", 2, 2, 2),  # rpn=8
            FMEAEntry("B", "fault2", "effect2", 9, 9, 9),  # rpn=729
        ]
        result = SafetyLayer.compute_fmea(entries)
        assert result["entries"][0]["component"] == "B"

    def test_compute_fmea_max_rpn(self):
        entries = [
            FMEAEntry("A", "f", "e", 10, 10, 10),
        ]
        result = SafetyLayer.compute_fmea(entries)
        assert result["max_rpn"] == 1000

    def test_compute_fmea_required_sil_is_max(self):
        entries = [
            FMEAEntry("A", "f", "e", 2, 2, 2),  # SIL_0
            FMEAEntry("B", "f", "e", 9, 9, 8),  # SIL_3
        ]
        result = SafetyLayer.compute_fmea(entries)
        assert result["required_sil"] == SILLevel.SIL_3

    def test_haiip_fmea_has_entries(self):
        assert len(HAIIP_FMEA) >= 4

    def test_haiip_fmea_rag_hallucination_severity(self):
        rag = next(e for e in HAIIP_FMEA if "RAG" in e.component)
        assert rag.severity >= 7  # hallucinated instructions are high severity


# ── PFD estimation ────────────────────────────────────────────────────────────

class TestPFDEstimation:

    def test_pfd_sil2_target(self):
        # λ_D = 1e-6 /h, T_I = 8760h (annual test) → PFD_avg ≈ 4.4e-3 (SIL-2 band)
        pfd = SafetyLayer.estimate_pfd(lambda_d=1e-6, proof_test_interval_h=8760)
        assert 1e-3 <= pfd <= 1e-2, f"Expected SIL-2 PFD, got {pfd}"

    def test_pfd_zero_rate(self):
        assert SafetyLayer.estimate_pfd(0.0, 8760) == 0.0

    def test_pfd_zero_interval(self):
        assert SafetyLayer.estimate_pfd(1e-5, 0.0) == 0.0

    def test_pfd_capped_at_1(self):
        pfd = SafetyLayer.estimate_pfd(1.0, 1e9)
        assert pfd <= 1.0

    def test_sil_achieved_sil4(self):
        assert SafetyLayer.sil_achieved(5e-5) == SILLevel.SIL_4

    def test_sil_achieved_sil3(self):
        assert SafetyLayer.sil_achieved(5e-4) == SILLevel.SIL_3

    def test_sil_achieved_sil2(self):
        assert SafetyLayer.sil_achieved(5e-3) == SILLevel.SIL_2

    def test_sil_achieved_sil1(self):
        assert SafetyLayer.sil_achieved(5e-2) == SILLevel.SIL_1

    def test_sil_achieved_sil0(self):
        assert SafetyLayer.sil_achieved(0.5) == SILLevel.SIL_0


# ── Module-level constants ────────────────────────────────────────────────────

def test_ai_is_advisory():
    assert AI_IS_ADVISORY is True

def test_escalation_threshold_reasonable():
    assert 0.5 <= ESCALATION_CONFIDENCE_THRESHOLD <= 0.9
