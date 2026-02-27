"""BDD-style Phase 6 user journey tests.

Scenarios:
1. Operator receives economic decision for failing machine
2. Engineer runs federated learning experiment and compares to baseline
3. Compliance officer audits human oversight metrics for EU AI Act
4. Admin reviews cost model ROI for fleet of 100 machines
5. Security: economic endpoint rejects unauthorised access
6. Engineer uses agentic RAG to diagnose machine with context
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from haiip.core.economic_ai import (
    CostProfile,
    EconomicDecisionEngine,
    MaintenanceAction,
)
from haiip.core.federated import FederatedLearner
from haiip.core.human_oversight import HumanOversightEngine, OversightEvent
from haiip.observability.cost_model import PredictionCostModel


# ── Scenario 1: Operator receives REPAIR_NOW for critical machine ──────────────

class TestOperatorDecisionJourney:
    """
    GIVEN: Machine M-003 shows anomaly_score=0.92, failure_probability=0.87
    WHEN:  Operator queries the economic decision engine
    THEN:  Decision is REPAIR_NOW with positive net_benefit
    AND:   Explanation mentions failure probability
    AND:   Human review flag set for high-cost decision
    """

    def test_operator_gets_repair_now(self) -> None:
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score=0.92, failure_probability=0.87,
            confidence=0.91, machine_id="M-003",
        )
        assert d.action == MaintenanceAction.REPAIR_NOW

    def test_positive_net_benefit_on_critical(self) -> None:
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score=0.92, failure_probability=0.87, confidence=0.91
        )
        assert d.net_benefit > 0

    def test_explanation_is_informative(self) -> None:
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score=0.92, failure_probability=0.87, confidence=0.91
        )
        assert len(d.explanation) > 50
        assert "0.87" in d.explanation or "0.870" in d.explanation

    def test_rul_in_explanation_when_provided(self) -> None:
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score=0.92, failure_probability=0.87,
            confidence=0.91, rul_cycles=25,
        )
        assert "25" in d.explanation

    def test_machine_id_in_metadata(self) -> None:
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score=0.9, failure_probability=0.85,
            machine_id="M-007",
        )
        assert d.metadata["machine_id"] == "M-007"

    def test_batch_for_fleet_of_10_machines(self) -> None:
        engine = EconomicDecisionEngine()
        rng    = np.random.default_rng(42)
        records = [
            {
                "anomaly_score":       float(rng.uniform(0, 1)),
                "failure_probability": float(rng.uniform(0, 1)),
                "machine_id":          f"M-{i:03d}",
            }
            for i in range(10)
        ]
        decisions = engine.batch_decide(records)
        assert len(decisions) == 10
        actions = [d.action.value for d in decisions]
        assert len(set(actions)) >= 2  # at least two different decision types


# ── Scenario 2: Engineer runs federated experiment ─────────────────────────────

class TestEngineerFederatedJourney:
    """
    GIVEN: 3 Nordic SME nodes with non-IID sensor data
    WHEN:  Engineer runs 5-round FedAvg experiment
    THEN:  Federated F1 is within 15% of centralized baseline
    AND:   Privacy is preserved (no raw data in result)
    AND:   Per-node metrics available for each round
    """

    @pytest.fixture(scope="class")
    def result(self) -> "FederatedResult":  # noqa: F821
        from haiip.core.federated import FederatedResult
        learner = FederatedLearner(random_state=42)
        return learner.run(n_rounds=5, local_epochs=2)

    def test_f1_within_15pct_of_centralized(self, result) -> None:
        assert result.federated_gap <= 0.15, (
            f"Gap: {result.federated_gap:.4f}"
        )

    def test_privacy_preserved(self, result) -> None:
        assert result.privacy_preserved is True

    def test_per_node_f1s_in_every_round(self, result) -> None:
        for rnd in result.rounds:
            assert "SME_FI" in rnd.node_f1s
            assert "SME_SE" in rnd.node_f1s
            assert "SME_NO" in rnd.node_f1s

    def test_experiment_id_in_result(self, result) -> None:
        uuid.UUID(result.experiment_id)

    def test_node_profiles_documented(self, result) -> None:
        for node_id in ("SME_FI", "SME_SE", "SME_NO"):
            assert node_id in result.node_profiles
            profile = result.node_profiles[node_id]
            assert "n_samples" in profile
            assert "failure_rate" in profile


# ── Scenario 3: Compliance officer audits human oversight ─────────────────────

class TestComplianceOfficerJourney:
    """
    GIVEN: 100 AI decisions were made last month
    WHEN:  Compliance officer requests oversight metrics
    THEN:  HIR, HOG, TCS are all computable
    AND:   Report contains EU AI Act Article 14 reference
    AND:   Risk reduction is non-negative when humans correct errors
    """

    @pytest.fixture
    def oversight_with_data(self) -> HumanOversightEngine:
        eng = HumanOversightEngine(target_hir=0.10)
        rng = np.random.default_rng(42)
        for i in range(100):
            ai_correct = rng.random() > 0.15  # 85% AI accuracy
            ai_label   = "failure" if rng.random() > 0.7 else "normal"
            true_label = ai_label if ai_correct else ("normal" if ai_label == "failure" else "failure")
            reviewed   = rng.random() < 0.12  # 12% human review rate
            overrode   = reviewed and not ai_correct and rng.random() > 0.3
            eng.record(OversightEvent.create(
                decision_id        = str(uuid.uuid4()),
                ai_label           = ai_label,
                ai_confidence      = float(rng.uniform(0.6, 0.99)),
                true_label         = true_label,
                human_reviewed     = reviewed,
                human_overrode     = overrode,
                human_label        = true_label if overrode else None,
                action_category    = "repair_now" if ai_label == "failure" else "monitor",
                expected_cost_ai   = float(rng.uniform(500, 5000)),
                expected_cost_human= float(rng.uniform(100, 2000)) if overrode else None,
            ))
        return eng

    def test_metrics_computable(self, oversight_with_data: HumanOversightEngine) -> None:
        m = oversight_with_data.compute_metrics()
        assert 0.0 <= m.hir <= 1.0
        assert -1.0 <= m.hog <= 1.0
        assert 0.0 <= m.tcs <= 1.0

    def test_report_contains_eu_ai_act(
        self, oversight_with_data: HumanOversightEngine
    ) -> None:
        m = oversight_with_data.compute_metrics()
        assert "EU AI Act" in m.report

    def test_to_dict_all_numeric_keys(
        self, oversight_with_data: HumanOversightEngine
    ) -> None:
        m = oversight_with_data.compute_metrics()
        dct = m.to_dict()
        for key in ("HIR", "HOG", "TCS", "ECE"):
            assert isinstance(dct[key], float)

    def test_hir_by_action_populated(
        self, oversight_with_data: HumanOversightEngine
    ) -> None:
        m = oversight_with_data.compute_metrics()
        assert len(m.hir_by_action) >= 1


# ── Scenario 4: Admin reviews fleet ROI ───────────────────────────────────────

class TestAdminFleetROIJourney:
    """
    GIVEN: Fleet of 100 machines making predictions daily
    WHEN:  Admin queries the cost model for 30-day ROI
    THEN:  Total net benefit and projections are available
    AND:   Average inference time is tracked
    AND:   Storage cost is included
    """

    def test_fleet_roi_100_machines(self) -> None:
        model  = PredictionCostModel(
            downtime_cost_eur    = 4000.0,
            maintenance_cost_eur = 590.0,
        )
        rng    = np.random.default_rng(42)
        # 100 machines × 30 predictions/day × 30 days = 90,000 predictions
        reports = [
            model.compute(
                inference_time_ms   = float(rng.uniform(10, 150)),
                failure_probability = float(rng.uniform(0, 1)),
                was_correct         = rng.random() > 0.1,
            )
            for _ in range(90_000)
        ]
        roi = model.fleet_roi(reports, period_days=30)
        assert roi["predictions_total"] == 90_000
        assert roi["avg_inference_ms"] > 0
        assert "storage_cost_eur" in roi
        assert "roi_ratio" in roi

    def test_high_accuracy_model_positive_roi(self) -> None:
        model = PredictionCostModel(downtime_cost_eur=4000.0)
        rng   = np.random.default_rng(0)
        reports = [
            model.compute(
                inference_time_ms   = 50.0,
                failure_probability = 0.85,  # high p(failure) → high avoided cost
                was_correct         = True,
            )
            for _ in range(1000)
        ]
        roi = model.fleet_roi(reports)
        # Each report avoids significant downtime → positive overall ROI
        assert roi["total_avoided_downtime_eur"] > roi["total_compute_cost_eur"]
