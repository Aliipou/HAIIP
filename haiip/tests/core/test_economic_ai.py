"""Brutal tests for EconomicDecisionEngine.

Coverage:
- All 4 decision branches (REPAIR_NOW, SCHEDULE, MONITOR, IGNORE)
- Boundary conditions (exact thresholds, noise floor)
- Cost formula correctness
- Batch decisions + ROI summary
- Human review triggers
- CostProfile properties
- Extreme values (p=0, p=1, score=0, score=1)
- Negative net_benefit (don't act)
- RUL integration in explanations
- Metadata passthrough
- Thread-safety (concurrent calls)
"""

from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest

from haiip.core.economic_ai import (
    CostProfile,
    EconomicDecision,
    EconomicDecisionEngine,
    MaintenanceAction,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def default_profile() -> CostProfile:
    return CostProfile()


@pytest.fixture
def engine(default_profile: CostProfile) -> EconomicDecisionEngine:
    return EconomicDecisionEngine(cost_profile=default_profile)


@pytest.fixture
def custom_engine() -> EconomicDecisionEngine:
    profile = CostProfile(
        production_rate_eur_hr=1000.0,
        downtime_hours_avg=12.0,
        labour_rate_eur_hr=100.0,
        labour_hours_avg=6.0,
        parts_cost_eur=500.0,
        opportunity_cost_eur=300.0,
        safety_factor=2.0,
        urgency_factor=0.6,
        noise_floor=0.03,
    )
    return EconomicDecisionEngine(
        cost_profile=profile,
        schedule_threshold=0.45,
        repair_now_threshold=0.70,
        monitor_score_threshold=0.15,
    )


# ── CostProfile ────────────────────────────────────────────────────────────────

class TestCostProfile:
    def test_c_downtime_formula(self) -> None:
        p = CostProfile(production_rate_eur_hr=500.0, downtime_hours_avg=8.0)
        assert p.c_downtime == pytest.approx(4000.0)

    def test_c_maintenance_formula(self) -> None:
        p = CostProfile(
            labour_rate_eur_hr=85.0,
            labour_hours_avg=4.0,
            parts_cost_eur=250.0,
            opportunity_cost_eur=150.0,
        )
        assert p.c_maintenance == pytest.approx(85 * 4 + 250 + 150)

    def test_c_false_negative_includes_safety(self) -> None:
        p = CostProfile(
            production_rate_eur_hr=500.0,
            downtime_hours_avg=8.0,
            safety_factor=2.0,
        )
        assert p.c_false_negative == pytest.approx(4000.0 * 2.0)

    def test_c_false_positive_is_15pct_maintenance(self) -> None:
        p = CostProfile(
            labour_rate_eur_hr=85.0,
            labour_hours_avg=4.0,
            parts_cost_eur=250.0,
            opportunity_cost_eur=150.0,
        )
        assert p.c_false_positive == pytest.approx(p.c_maintenance * 0.15)

    def test_defaults_are_reasonable(self) -> None:
        p = CostProfile()
        assert p.c_downtime > 0
        assert p.c_maintenance > 0
        assert p.noise_floor > 0
        assert p.safety_factor >= 1.0


# ── Decision correctness ───────────────────────────────────────────────────────

class TestDecisionBranches:
    def test_repair_now_high_probability(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.9, failure_probability=0.85, confidence=0.9)
        assert d.action == MaintenanceAction.REPAIR_NOW

    def test_repair_now_at_exact_threshold(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.8, failure_probability=0.75, confidence=0.9)
        assert d.action == MaintenanceAction.REPAIR_NOW

    def test_schedule_above_schedule_threshold(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.5, failure_probability=0.60, confidence=0.8)
        assert d.action == MaintenanceAction.SCHEDULE

    def test_schedule_at_exact_schedule_threshold(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.5, failure_probability=0.50, confidence=0.85)
        assert d.action == MaintenanceAction.SCHEDULE

    def test_monitor_elevated_score_low_probability(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.35, failure_probability=0.15, confidence=0.75)
        assert d.action == MaintenanceAction.MONITOR

    def test_monitor_default_low_probability(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.10, failure_probability=0.20, confidence=0.9)
        assert d.action == MaintenanceAction.MONITOR

    def test_ignore_below_noise_floor(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.02, failure_probability=0.05, confidence=0.95)
        assert d.action == MaintenanceAction.IGNORE

    def test_ignore_requires_both_low_score_and_low_prob(
        self, engine: EconomicDecisionEngine
    ) -> None:
        # Low score but elevated probability → not IGNORE
        d = engine.decide(anomaly_score=0.02, failure_probability=0.20, confidence=0.9)
        assert d.action != MaintenanceAction.IGNORE

    def test_custom_thresholds_respected(self, custom_engine: EconomicDecisionEngine) -> None:
        # repair_now_threshold=0.70
        d = engine_decide_helper(custom_engine, anomaly_score=0.8, failure_probability=0.72)
        assert d.action == MaintenanceAction.REPAIR_NOW

    def test_rul_in_explanation_repair_now(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.9, failure_probability=0.85, rul_cycles=30)
        assert "RUL" in d.explanation or "30" in d.explanation

    def test_rul_in_explanation_schedule(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.5, failure_probability=0.60, rul_cycles=200)
        assert "200" in d.explanation


def engine_decide_helper(eng: EconomicDecisionEngine, **kwargs: float) -> EconomicDecision:
    return eng.decide(**kwargs)  # type: ignore


# ── Cost arithmetic ────────────────────────────────────────────────────────────

class TestCostArithmetic:
    def test_expected_cost_wait_formula(self, engine: EconomicDecisionEngine) -> None:
        p  = 0.8
        cf = engine.cost_profile
        d  = engine.decide(anomaly_score=0.9, failure_probability=p)
        assert d.expected_cost_wait == pytest.approx(p * cf.c_false_negative, rel=1e-3)

    def test_net_benefit_sign_repair_now(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.9, failure_probability=0.9, confidence=0.9)
        # Net benefit should be positive for high P(failure)
        assert d.net_benefit > 0

    def test_net_benefit_near_zero_for_ignore(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.01, failure_probability=0.02)
        # Very low P(failure): waiting costs less than acting
        assert d.net_benefit < engine.cost_profile.c_maintenance

    def test_decision_id_is_uuid(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.5, failure_probability=0.5)
        assert uuid.UUID(d.decision_id)  # should not raise

    def test_to_dict_round_trip(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.6, failure_probability=0.65)
        dct = d.to_dict()
        assert "decision_id" in dct
        assert "action" in dct
        assert "net_benefit" in dct
        assert "requires_human_review" in dct


# ── Human review triggers ──────────────────────────────────────────────────────

class TestHumanReview:
    def test_low_confidence_triggers_review(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(
            anomaly_score=0.5, failure_probability=0.6, confidence=0.20
        )
        assert d.requires_human_review

    def test_high_confidence_no_review_for_monitor(
        self, engine: EconomicDecisionEngine
    ) -> None:
        d = engine.decide(
            anomaly_score=0.3, failure_probability=0.25, confidence=0.95
        )
        # Monitor with high confidence → no human review required
        assert not d.requires_human_review

    def test_high_net_benefit_triggers_review(self, engine: EconomicDecisionEngine) -> None:
        # Artificially set huge downtime cost
        big = EconomicDecisionEngine(
            cost_profile=CostProfile(production_rate_eur_hr=5000.0, downtime_hours_avg=20.0)
        )
        d = big.decide(anomaly_score=0.9, failure_probability=0.95, confidence=0.9)
        assert d.requires_human_review


# ── Extreme values ─────────────────────────────────────────────────────────────

class TestExtremeValues:
    def test_p_zero(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.0, failure_probability=0.0)
        assert d.failure_probability == 0.0
        assert d.action in list(MaintenanceAction)

    def test_p_one(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=1.0, failure_probability=1.0)
        assert d.action == MaintenanceAction.REPAIR_NOW

    def test_clamping_above_one(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=2.0, failure_probability=1.5)
        assert d.failure_probability == pytest.approx(1.0)
        assert d.anomaly_score == pytest.approx(1.0)

    def test_clamping_below_zero(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=-0.5, failure_probability=-0.1)
        assert d.failure_probability == pytest.approx(0.0)
        assert d.anomaly_score == pytest.approx(0.0)


# ── Batch + ROI ────────────────────────────────────────────────────────────────

class TestBatchAndROI:
    def test_batch_returns_all_results(self, engine: EconomicDecisionEngine) -> None:
        records = [
            {"anomaly_score": 0.9, "failure_probability": 0.85},
            {"anomaly_score": 0.3, "failure_probability": 0.25},
            {"anomaly_score": 0.01, "failure_probability": 0.02},
        ]
        results = engine.batch_decide(records)
        assert len(results) == 3

    def test_roi_summary_keys(self, engine: EconomicDecisionEngine) -> None:
        decisions = engine.batch_decide([
            {"anomaly_score": 0.9, "failure_probability": 0.85},
            {"anomaly_score": 0.5, "failure_probability": 0.55},
            {"anomaly_score": 0.1, "failure_probability": 0.10},
        ])
        roi = engine.roi_summary(decisions)
        for key in ("total_net_benefit", "decisions_by_action", "avg_confidence",
                    "human_review_count", "projected_downtime_savings_eur"):
            assert key in roi

    def test_roi_summary_empty(self, engine: EconomicDecisionEngine) -> None:
        roi = engine.roi_summary([])
        assert roi["total_net_benefit"] == 0.0

    def test_roi_action_counts(self, engine: EconomicDecisionEngine) -> None:
        records = [{"anomaly_score": 0.9, "failure_probability": 0.85}] * 5
        decisions = engine.batch_decide(records)
        roi = engine.roi_summary(decisions)
        assert roi["decisions_by_action"].get("repair_now", 0) == 5

    def test_metadata_passthrough(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(
            anomaly_score=0.5,
            failure_probability=0.55,
            machine_id="M-007",
            metadata={"sensor": "vibration", "tenant": "acme"},
        )
        assert d.metadata["machine_id"] == "M-007"
        assert d.metadata["sensor"] == "vibration"


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_decisions(self, engine: EconomicDecisionEngine) -> None:
        results: list[EconomicDecision] = []
        lock = threading.Lock()

        def worker() -> None:
            d = engine.decide(anomaly_score=0.5, failure_probability=0.55)
            with lock:
                results.append(d)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        ids = {r.decision_id for r in results}
        assert len(ids) == 20  # all unique
