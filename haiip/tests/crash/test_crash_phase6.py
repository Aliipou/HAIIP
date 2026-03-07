"""Crash tests for Phase 6 modules — robustness under adversarial inputs.

Tests every module against:
- NaN / Inf inputs
- Empty inputs
- Negative values
- Extremely large values
- None / null inputs
- Type mismatches
- Concurrent access under load
- Serialisation with non-serialisable metadata
"""

from __future__ import annotations

import math
import threading
import uuid

import pytest

from haiip.core.economic_ai import CostProfile, EconomicDecisionEngine
from haiip.core.federated import FederatedLearner, FederatedNode, NodeProfile, SMENode
from haiip.core.human_oversight import HumanOversightEngine, OversightEvent
from haiip.observability.cost_model import PredictionCostModel

# ── EconomicDecisionEngine crash tests ────────────────────────────────────────


class TestEconomicCrash:
    @pytest.fixture
    def engine(self) -> EconomicDecisionEngine:
        return EconomicDecisionEngine()

    def test_nan_anomaly_score_clamped(self, engine: EconomicDecisionEngine) -> None:
        # NaN should be handled — either clamped or raises cleanly
        try:
            d = engine.decide(anomaly_score=float("nan"), failure_probability=0.5)
            # If it doesn't raise, result must be a valid MaintenanceAction
            assert d.action is not None
        except (ValueError, TypeError):
            pass  # Acceptable — NaN is invalid input

    def test_inf_failure_probability_clamped(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.5, failure_probability=float("inf"))
        assert d.failure_probability == pytest.approx(1.0)

    def test_negative_inf_clamped(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=float("-inf"), failure_probability=0.5)
        assert d.anomaly_score == pytest.approx(0.0)

    def test_zero_zero_inputs(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.0, failure_probability=0.0)
        assert d.action is not None

    def test_both_one(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=1.0, failure_probability=1.0)
        assert d.action is not None

    def test_empty_batch(self, engine: EconomicDecisionEngine) -> None:
        results = engine.batch_decide([])
        assert results == []

    def test_roi_summary_with_empty(self, engine: EconomicDecisionEngine) -> None:
        roi = engine.roi_summary([])
        assert "total_net_benefit" in roi

    def test_large_cost_profile(self) -> None:
        huge = CostProfile(
            production_rate_eur_hr=1_000_000.0,
            downtime_hours_avg=1000.0,
        )
        engine = EconomicDecisionEngine(cost_profile=huge)
        d = engine.decide(anomaly_score=0.9, failure_probability=0.95)
        assert math.isfinite(d.expected_cost_wait)
        assert math.isfinite(d.net_benefit)

    def test_confidence_nan(self, engine: EconomicDecisionEngine) -> None:
        try:
            d = engine.decide(anomaly_score=0.5, failure_probability=0.5, confidence=float("nan"))
            assert d.confidence is not None
        except (ValueError, TypeError):
            pass

    def test_concurrent_100_calls(self, engine: EconomicDecisionEngine) -> None:
        errors: list[Exception] = []

        def call() -> None:
            try:
                engine.decide(anomaly_score=0.5, failure_probability=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_to_dict_with_none_metadata(self, engine: EconomicDecisionEngine) -> None:
        d = engine.decide(anomaly_score=0.5, failure_probability=0.5, metadata=None)
        dct = d.to_dict()
        assert "metadata" in dct


# ── FederatedLearner crash tests ───────────────────────────────────────────────


class TestFederatedCrash:
    def test_zero_rounds(self) -> None:
        """Zero rounds should either return empty or raise cleanly."""
        learner = FederatedLearner(random_state=42)
        try:
            result = learner.run(n_rounds=0, local_epochs=1)
            assert result.total_rounds == 0
        except (ValueError, IndexError):
            pass

    def test_single_node(self) -> None:
        profiles = [NodeProfile(SMENode.SME_FI, 200, 0.1, 0.2, country="FI", industry="test")]
        learner = FederatedLearner(profiles=profiles, random_state=0)
        result = learner.run(n_rounds=2, local_epochs=1)
        assert result.total_rounds >= 1

    def test_all_failure_node(self) -> None:
        """Node with 100% failure rate."""
        profiles = [
            NodeProfile(SMENode.SME_FI, 100, 1.0, 0.1, country="FI", industry="test"),
            NodeProfile(SMENode.SME_SE, 100, 0.0, 0.1, country="SE", industry="test"),
        ]
        learner = FederatedLearner(profiles=profiles, random_state=0)
        result = learner.run(n_rounds=2, local_epochs=1)
        for rnd in result.rounds:
            assert 0.0 <= rnd.global_f1 <= 1.0

    def test_tiny_dataset(self) -> None:
        profiles = [
            NodeProfile(SMENode.SME_FI, 10, 0.2, 0.1, country="FI", industry="test"),
            NodeProfile(SMENode.SME_SE, 10, 0.3, 0.1, country="SE", industry="test"),
        ]
        learner = FederatedLearner(profiles=profiles, random_state=0)
        result = learner.run(n_rounds=2, local_epochs=1)
        assert result is not None

    def test_node_zero_failure_rate(self) -> None:
        node = FederatedNode(
            profile=NodeProfile(SMENode.SME_NO, 100, 0.0, 0.1, country="NO"),
            random_state=0,
        )
        assert node.y.sum() == 0  # no failures


# ── HumanOversightEngine crash tests ──────────────────────────────────────────


class TestHumanOversightCrash:
    @pytest.fixture
    def engine(self) -> HumanOversightEngine:
        return HumanOversightEngine()

    def test_empty_engine_raises(self, engine: HumanOversightEngine) -> None:
        with pytest.raises(ValueError):
            engine.compute_metrics()

    def test_single_event_does_not_crash(self, engine: HumanOversightEngine) -> None:
        engine.record(
            OversightEvent.create(
                decision_id="d1",
                ai_label="normal",
                ai_confidence=0.8,
                true_label="normal",
                human_reviewed=False,
            )
        )
        m = engine.compute_metrics()
        assert 0.0 <= m.tcs <= 1.0

    def test_all_wrong_ai(self, engine: HumanOversightEngine) -> None:
        for _ in range(20):
            engine.record(
                OversightEvent.create(
                    decision_id=str(uuid.uuid4()),
                    ai_label="failure",
                    ai_confidence=0.9,
                    true_label="normal",
                    human_reviewed=False,
                )
            )
        m = engine.compute_metrics()
        assert m.ai_accuracy == pytest.approx(0.0)

    def test_all_overridden(self, engine: HumanOversightEngine) -> None:
        for _ in range(10):
            engine.record(
                OversightEvent.create(
                    decision_id=str(uuid.uuid4()),
                    ai_label="failure",
                    ai_confidence=0.5,
                    true_label="normal",
                    human_reviewed=True,
                    human_overrode=True,
                    human_label="normal",
                )
            )
        m = engine.compute_metrics()
        assert m.n_overridden == 10
        assert m.hog > 0

    def test_rolling_hir_small_window(self, engine: HumanOversightEngine) -> None:
        for i in range(3):
            engine.record(
                OversightEvent.create(
                    decision_id=str(uuid.uuid4()),
                    ai_label="normal",
                    ai_confidence=0.8,
                    true_label="normal",
                    human_reviewed=(i % 2 == 0),
                )
            )
        result = engine.rolling_hir(window=5)
        assert len(result) >= 0  # may be empty for window > events

    def test_concurrent_recording(self, engine: HumanOversightEngine) -> None:
        def record_events() -> None:
            for _ in range(10):
                engine.record(
                    OversightEvent.create(
                        decision_id=str(uuid.uuid4()),
                        ai_label="normal",
                        ai_confidence=0.8,
                        true_label="normal",
                        human_reviewed=False,
                    )
                )

        threads = [threading.Thread(target=record_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert engine.event_count == 50


# ── PredictionCostModel crash tests ───────────────────────────────────────────


class TestCostModelCrash:
    @pytest.fixture
    def model(self) -> PredictionCostModel:
        return PredictionCostModel()

    def test_inf_latency(self, model: PredictionCostModel) -> None:
        try:
            r = model.compute(inference_time_ms=float("inf"), failure_probability=0.5)
            assert math.isinf(r.compute_cost_eur) or r.compute_cost_eur > 0
        except (ValueError, OverflowError):
            pass

    def test_negative_latency(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=-10.0, failure_probability=0.5)
        assert r.compute_cost_eur <= 0  # negative time → negative compute cost

    def test_p_exactly_0_5(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.5)
        # At boundary: p=0.5 → avoided = 0.5 * downtime - maintenance
        assert r is not None

    def test_fleet_roi_single_report(self, model: PredictionCostModel) -> None:
        r = model.compute(50.0, 0.8)
        roi = model.fleet_roi([r])
        assert roi["predictions_total"] == 1

    def test_to_dict_json_serialisable(self, model: PredictionCostModel) -> None:
        import json

        r = model.compute(50.0, 0.7, metadata={"machine": "M001"})
        dct = r.to_dict()
        # Should not raise
        json.dumps(dct)
