"""
Oversight Simulation Tests
===========================
Tests for OperatorSimulationModel — assumption citations, confidence reporting,
determinism, and behavioural properties.

Rules:
- No test mocks the thing it is testing.
- Every assert must be able to fail.
- Every assert tests a property that can meaningfully be wrong.
"""

from __future__ import annotations

import pytest

from haiip.core.oversight_simulation import (
    ASSUMPTION_METADATA,
    LOW_CONFIDENCE_ASSUMPTIONS,
    AlertStub,
    OperatorProfile,
    OperatorRole,
    OperatorSimulationModel,
    OversightReport,
    generate_oversight_report,
)


# ---------------------------------------------------------------------------
# Assumption documentation tests
# ---------------------------------------------------------------------------

class TestAssumptionDocumentation:

    def test_all_assumptions_have_citations(self):
        """Every assumption entry in ASSUMPTION_METADATA has a non-empty 'source' field."""
        for name, meta in ASSUMPTION_METADATA.items():
            assert "source" in meta, f"Assumption {name} missing 'source'"
            assert len(meta["source"]) > 0, f"Assumption {name} has empty 'source'"

    def test_all_assumptions_have_confidence_level(self):
        """Every assumption has a 'confidence' field with a non-empty value."""
        valid_levels = {"HIGH", "MEDIUM", "LOW", "VERY LOW", "NONE"}
        for name, meta in ASSUMPTION_METADATA.items():
            assert "confidence" in meta, f"Assumption {name} missing 'confidence'"
            assert meta["confidence"] in valid_levels, (
                f"Assumption {name} confidence '{meta['confidence']}' not in {valid_levels}"
            )

    def test_all_assumptions_have_testable_field(self):
        """Every assumption has a 'testable' description."""
        for name, meta in ASSUMPTION_METADATA.items():
            assert "testable" in meta, f"Assumption {name} missing 'testable'"
            assert len(meta["testable"]) > 0, f"Assumption {name} has empty 'testable'"

    def test_low_confidence_assumptions_are_named(self):
        """LOW_CONFIDENCE_ASSUMPTIONS list is non-empty and all entries are in ASSUMPTION_METADATA."""
        assert len(LOW_CONFIDENCE_ASSUMPTIONS) > 0
        for name in LOW_CONFIDENCE_ASSUMPTIONS:
            assert name in ASSUMPTION_METADATA, (
                f"LOW_CONFIDENCE_ASSUMPTIONS entry '{name}' not in ASSUMPTION_METADATA"
            )

    def test_explanation_boost_has_confidence_none(self):
        """ASSUMPTION_EXPLANATION_BOOST is confidence=NONE (not yet measured by RQ3)."""
        meta = ASSUMPTION_METADATA.get("ASSUMPTION_EXPLANATION_BOOST", {})
        assert meta.get("confidence") == "NONE", (
            "ASSUMPTION_EXPLANATION_BOOST should be confidence=NONE until RQ3 is measured"
        )

    def test_false_positive_learning_is_very_low_confidence(self):
        """ASSUMPTION_FALSE_POSITIVE_LEARNING is flagged as VERY LOW confidence."""
        meta = ASSUMPTION_METADATA.get("ASSUMPTION_FALSE_POSITIVE_LEARNING", {})
        assert meta.get("confidence") == "VERY LOW"


# ---------------------------------------------------------------------------
# Oversight report properties
# ---------------------------------------------------------------------------

class TestOversightReportProperties:

    def test_oversight_report_always_includes_confidence_level(self):
        """generate_oversight_report() always sets simulation_confidence."""
        report = generate_oversight_report(
            hir=0.10, hog=0.03, tcs=0.82, ece=0.08,
            n_events=100, n_reviewed=10, n_overridden=3,
        )
        assert report.simulation_confidence is not None
        assert len(report.simulation_confidence) > 0

    def test_oversight_report_simulation_confidence_is_low(self):
        """simulation_confidence is hardcoded 'LOW' until field study runs."""
        report = generate_oversight_report(
            hir=0.10, hog=0.05, tcs=0.85, ece=0.05,
            n_events=200, n_reviewed=20, n_overridden=5,
        )
        assert report.simulation_confidence == "LOW", (
            "simulation_confidence must be 'LOW' until a field study validates the assumptions"
        )

    def test_oversight_report_field_study_required_is_true(self):
        """field_study_required is always True until real operator data is collected."""
        report = generate_oversight_report(
            hir=0.10, hog=0.03, tcs=0.80, ece=0.10,
            n_events=50, n_reviewed=5, n_overridden=1,
        )
        assert report.field_study_required is True

    def test_oversight_report_includes_low_confidence_list(self):
        """OversightReport.low_confidence_assumptions is a non-empty list."""
        report = generate_oversight_report(
            hir=0.10, hog=0.03, tcs=0.82, ece=0.08,
            n_events=100, n_reviewed=10, n_overridden=3,
        )
        assert isinstance(report.low_confidence_assumptions, list)
        assert len(report.low_confidence_assumptions) > 0

    def test_oversight_report_to_dict_includes_warning(self):
        """to_dict() includes a warning key alerting consumers."""
        report = generate_oversight_report(
            hir=0.10, hog=0.03, tcs=0.82, ece=0.08,
            n_events=100, n_reviewed=10, n_overridden=3,
        )
        d = report.to_dict()
        assert "warning" in d
        assert "simulation" in d["warning"].lower() or "field study" in d["warning"].lower()

    def test_warning_logged_when_low_confidence_assumptions_used(self, caplog):
        """generate_oversight_report() emits a warning log for low-confidence assumptions."""
        import logging
        with caplog.at_level(logging.WARNING):
            generate_oversight_report(
                hir=0.10, hog=0.03, tcs=0.82, ece=0.08,
                n_events=100, n_reviewed=10, n_overridden=3,
            )
        # At least one warning should reference low confidence
        warning_texts = " ".join(caplog.messages)
        assert any(
            kw in warning_texts.lower()
            for kw in ["low_confidence", "field study", "low confidence", "simulation"]
        ), f"No expected warning in logs. Messages: {caplog.messages}"


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

class TestOperatorBehaviour:

    def _make_alert(self, is_tp: bool = True, has_explanation: bool = False) -> AlertStub:
        return AlertStub(
            alert_id="alert-001",
            machine_id="pump-01",
            fault_type="HDF",
            is_true_positive=is_tp,
            ai_confidence=0.80,
            has_explanation=has_explanation,
        )

    def test_expert_operator_higher_acceptance_than_novice(self):
        """Expert accept rate > novice accept rate on same alerts (averaged over many)."""
        expert_profile = OperatorProfile(role=OperatorRole.EXPERT, experience_years=8.0)
        novice_profile = OperatorProfile(role=OperatorRole.NOVICE, experience_years=0.5)

        alert = self._make_alert()

        expert_accepts = 0
        novice_accepts = 0
        n = 500

        for seed in range(n):
            model_e = OperatorSimulationModel(seed=seed)
            model_n = OperatorSimulationModel(seed=seed)
            if model_e.simulate_operator_decision(alert, expert_profile, 0, 0).decision == "accept":
                expert_accepts += 1
            if model_n.simulate_operator_decision(alert, novice_profile, 0, 0).decision == "accept":
                novice_accepts += 1

        expert_rate = expert_accepts / n
        novice_rate = novice_accepts / n

        assert expert_rate > novice_rate, (
            f"Expert rate {expert_rate:.3f} not > novice rate {novice_rate:.3f}"
        )

    def test_fatigue_reduces_acceptance_rate(self):
        """Alert acceptance rate at shift_hour=8 < rate at shift_hour=0."""
        expert_profile = OperatorProfile(role=OperatorRole.EXPERT, experience_years=8.0)
        alert          = self._make_alert()

        fresh_accepts  = 0
        tired_accepts  = 0
        n = 500

        for seed in range(n):
            m = OperatorSimulationModel(seed=seed)
            if m.simulate_operator_decision(alert, expert_profile, 0, 0).decision == "accept":
                fresh_accepts += 1
            m2 = OperatorSimulationModel(seed=seed)
            if m2.simulate_operator_decision(alert, expert_profile, 8, 0).decision == "accept":
                tired_accepts += 1

        fresh_rate = fresh_accepts / n
        tired_rate = tired_accepts / n

        assert fresh_rate > tired_rate, (
            f"Fatigue not reducing acceptance: fresh={fresh_rate:.3f}, tired={tired_rate:.3f}"
        )

    def test_false_positive_history_reduces_trust(self):
        """Prior false positives reduce alert acceptance rate."""
        expert_profile = OperatorProfile(role=OperatorRole.EXPERT, experience_years=8.0)
        alert          = self._make_alert()

        no_fp_accepts  = 0
        many_fp_accepts = 0
        n = 500

        for seed in range(n):
            m = OperatorSimulationModel(seed=seed)
            if m.simulate_operator_decision(alert, expert_profile, 0, 0).decision == "accept":
                no_fp_accepts += 1
            m2 = OperatorSimulationModel(seed=seed)
            if m2.simulate_operator_decision(alert, expert_profile, 0, 10).decision == "accept":
                many_fp_accepts += 1

        no_fp_rate   = no_fp_accepts / n
        many_fp_rate = many_fp_accepts / n

        assert no_fp_rate > many_fp_rate, (
            f"FP history not reducing trust: no_fp={no_fp_rate:.3f}, many_fp={many_fp_rate:.3f}"
        )

    def test_explanation_boost_increases_acceptance_rate(self):
        """Showing explanation increases acceptance rate (ASSUMPTION_EXPLANATION_BOOST > 0)."""
        expert_profile = OperatorProfile(role=OperatorRole.EXPERT, experience_years=8.0)
        alert_no_exp   = self._make_alert(has_explanation=False)
        alert_with_exp = self._make_alert(has_explanation=True)

        no_exp_accepts  = 0
        with_exp_accepts = 0
        n = 500

        for seed in range(n):
            m = OperatorSimulationModel(seed=seed)
            if m.simulate_operator_decision(alert_no_exp,  expert_profile, 0, 0).decision == "accept":
                no_exp_accepts += 1
            m2 = OperatorSimulationModel(seed=seed)
            if m2.simulate_operator_decision(alert_with_exp, expert_profile, 0, 0).decision == "accept":
                with_exp_accepts += 1

        no_exp_rate  = no_exp_accepts / n
        with_exp_rate = with_exp_accepts / n

        # Note: ASSUMPTION_EXPLANATION_BOOST = 0.08 (CONFIDENCE: NONE)
        # This test documents the simulation assumption — not a validated result
        assert with_exp_rate >= no_exp_rate, (
            f"Explanation boost not working: no_exp={no_exp_rate:.3f}, with_exp={with_exp_rate:.3f}. "
            "Check ASSUMPTION_EXPLANATION_BOOST constant."
        )

    def test_simulation_is_deterministic_given_seed(self):
        """Same seed produces identical simulation results."""
        profile = OperatorProfile(role=OperatorRole.EXPERT, experience_years=5.0)
        alert   = self._make_alert()

        m1 = OperatorSimulationModel(seed=123)
        m2 = OperatorSimulationModel(seed=123)

        d1 = m1.simulate_operator_decision(alert, profile, 2, 1)
        d2 = m2.simulate_operator_decision(alert, profile, 2, 1)

        assert d1.decision == d2.decision
        assert d1.simulated_confidence == d2.simulated_confidence

    def test_different_seeds_can_produce_different_results(self):
        """Different seeds can produce different decisions (simulation is stochastic)."""
        profile = OperatorProfile(role=OperatorRole.NOVICE, experience_years=0.5)
        alert   = self._make_alert()

        decisions = set()
        for seed in range(20):
            m = OperatorSimulationModel(seed=seed)
            decisions.add(m.simulate_operator_decision(alert, profile, 0, 0).decision)

        # With a novice, we should see at least 2 different outcomes over 20 seeds
        assert len(decisions) >= 2, (
            f"Simulation appears deterministic regardless of seed: {decisions}"
        )

    def test_get_confidence_report_returns_all_assumptions(self):
        """get_confidence_report() returns an entry for every assumption in ASSUMPTION_METADATA."""
        model  = OperatorSimulationModel()
        report = model.get_confidence_report()
        for name in ASSUMPTION_METADATA:
            assert name in report, f"Missing assumption in confidence report: {name}"
