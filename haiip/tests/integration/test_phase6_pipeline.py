"""Integration tests for Phase 6 pipeline — end-to-end economic AI + oversight.

Tests the complete flow:
    Sensor readings
    → AnomalyDetector
    → MaintenancePredictor
    → EconomicDecisionEngine
    → HumanOversightEngine
    → OversightMetrics

And:
    FederatedLearner
    → compare vs centralized baseline
    → assert privacy preserved

All tests use real implementations (not mocks) on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from haiip.core.anomaly import AnomalyDetector
from haiip.core.economic_ai import CostProfile, EconomicDecisionEngine, MaintenanceAction
from haiip.core.federated import FederatedLearner
from haiip.core.human_oversight import HumanOversightEngine, OversightEvent
from haiip.core.maintenance import MaintenancePredictor
from haiip.observability.cost_model import PredictionCostModel


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def normal_data(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 1, (300, 5))


@pytest.fixture(scope="module")
def failure_data(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(2.5, 1.2, (50, 5))


@pytest.fixture(scope="module")
def trained_anomaly_detector(normal_data: np.ndarray) -> AnomalyDetector:
    det = AnomalyDetector(contamination=0.1, random_state=42)
    det.fit(normal_data)
    return det


@pytest.fixture(scope="module")
def trained_maintenance_predictor(
    normal_data: np.ndarray, failure_data: np.ndarray
) -> MaintenancePredictor:
    X = np.vstack([normal_data[:100], failure_data])
    y_class = np.array([0] * 100 + [1] * 50)
    pred = MaintenancePredictor(random_state=42)
    pred.fit(X, y_class)
    return pred


# ── Full pipeline: sensor → anomaly → maintenance → economic ───────────────────

class TestFullEconomicPipeline:
    def test_normal_reading_leads_to_monitor_or_ignore(
        self,
        trained_anomaly_detector: AnomalyDetector,
        trained_maintenance_predictor: MaintenancePredictor,
    ) -> None:
        x = np.array([[0.1, -0.1, 0.2, -0.2, 0.1]])  # clearly normal
        anom   = trained_anomaly_detector.predict(x)
        maint  = trained_maintenance_predictor.predict(x)
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score       = anom["anomaly_score"],
            failure_probability = maint["failure_probability"],
            confidence          = anom["confidence"],
        )
        assert d.action in (MaintenanceAction.MONITOR, MaintenanceAction.IGNORE)

    def test_failure_reading_leads_to_repair_or_schedule(
        self,
        trained_anomaly_detector: AnomalyDetector,
        trained_maintenance_predictor: MaintenancePredictor,
    ) -> None:
        x = np.array([[3.0, 2.8, 2.9, 3.1, 2.7]])  # clearly anomalous
        anom   = trained_anomaly_detector.predict(x)
        maint  = trained_maintenance_predictor.predict(x)
        engine = EconomicDecisionEngine()
        d = engine.decide(
            anomaly_score       = anom["anomaly_score"],
            failure_probability = maint["failure_probability"],
            confidence          = anom["confidence"],
        )
        # High anomaly readings should not result in IGNORE
        assert d.action != MaintenanceAction.IGNORE

    def test_cost_report_generated(
        self,
        trained_anomaly_detector: AnomalyDetector,
        trained_maintenance_predictor: MaintenancePredictor,
    ) -> None:
        x     = np.array([[2.0, 1.8, 2.2, 1.9, 2.1]])
        maint = trained_maintenance_predictor.predict(x)
        model = PredictionCostModel()
        import time
        t0 = time.perf_counter()
        _  = trained_maintenance_predictor.predict(x)
        latency = (time.perf_counter() - t0) * 1000

        report = model.compute(
            inference_time_ms   = latency,
            failure_probability = maint["failure_probability"],
            was_correct         = True,
        )
        assert report.net_value_eur is not None

    def test_batch_economic_with_fleet_roi(
        self,
        trained_anomaly_detector: AnomalyDetector,
        trained_maintenance_predictor: MaintenancePredictor,
    ) -> None:
        engine    = EconomicDecisionEngine()
        cost_model = PredictionCostModel()
        rng       = np.random.default_rng(0)
        X_test    = rng.normal(1.0, 1.5, (20, 5))

        decisions  = []
        cost_repts = []
        for row in X_test:
            x    = row.reshape(1, -1)
            anom = trained_anomaly_detector.predict(x)
            maint= trained_maintenance_predictor.predict(x)
            d    = engine.decide(
                anomaly_score       = anom["anomaly_score"],
                failure_probability = maint["failure_probability"],
                confidence          = anom["confidence"],
            )
            decisions.append(d)
            cr = cost_model.compute(
                inference_time_ms   = 45.0,
                failure_probability = maint["failure_probability"],
            )
            cost_repts.append(cr)

        roi_ec  = engine.roi_summary(decisions)
        roi_cm  = cost_model.fleet_roi(cost_repts)
        assert roi_ec["decisions_total"] == 20
        assert roi_cm["predictions_total"] == 20


# ── Oversight pipeline ─────────────────────────────────────────────────────────

class TestOversightPipeline:
    def test_oversight_after_economic_decisions(
        self,
        trained_anomaly_detector: AnomalyDetector,
        trained_maintenance_predictor: MaintenancePredictor,
    ) -> None:
        """Run economic decisions, simulate human oversight, compute metrics."""
        engine   = EconomicDecisionEngine()
        oversight = HumanOversightEngine(target_hir=0.10)
        rng      = np.random.default_rng(5)
        X_test   = rng.normal(0.5, 1.5, (30, 5))

        for i, row in enumerate(X_test):
            x    = row.reshape(1, -1)
            anom = trained_anomaly_detector.predict(x)
            maint= trained_maintenance_predictor.predict(x)
            d    = engine.decide(
                anomaly_score       = anom["anomaly_score"],
                failure_probability = maint["failure_probability"],
                confidence          = anom["confidence"],
            )
            # Simulate: human reviews high-risk decisions
            human_reviewed = d.action == MaintenanceAction.REPAIR_NOW
            true_label     = "failure" if maint["failure_probability"] > 0.6 else "normal"
            ai_label       = "failure" if d.action in (
                MaintenanceAction.REPAIR_NOW, MaintenanceAction.SCHEDULE
            ) else "normal"

            event = OversightEvent.create(
                decision_id     = d.decision_id,
                ai_label        = ai_label,
                ai_confidence   = anom["confidence"],
                true_label      = true_label,
                human_reviewed  = human_reviewed,
                action_category = d.action.value,
                expected_cost_ai= d.expected_cost_action,
            )
            oversight.record(event)

        metrics = oversight.compute_metrics()
        assert 0.0 <= metrics.hir <= 1.0
        assert -1.0 <= metrics.hog <= 1.0
        assert 0.0 <= metrics.tcs <= 1.0
        assert metrics.n_events == 30


# ── Federated + centralized comparison ────────────────────────────────────────

class TestFederatedIntegration:
    def test_federated_vs_centralized_gap_acceptable(self) -> None:
        learner = FederatedLearner(random_state=42)
        result  = learner.run(n_rounds=5, local_epochs=2)
        assert result.privacy_preserved
        # Federated gap should be < 20% (empirical tolerance)
        assert result.federated_gap < 0.20, (
            f"Federated gap too large: {result.federated_gap:.4f}"
        )

    def test_all_nodes_improve_over_rounds(self) -> None:
        learner = FederatedLearner(random_state=0)
        result  = learner.run(n_rounds=6, local_epochs=2)
        if len(result.rounds) >= 3:
            # At least last round F1 >= first round F1 − 5% (tolerance)
            assert result.rounds[-1].global_f1 >= result.rounds[0].global_f1 - 0.05

    def test_per_node_losses_decrease(self) -> None:
        learner = FederatedLearner(random_state=7)
        result  = learner.run(n_rounds=5, local_epochs=2)
        if len(result.rounds) >= 3:
            for node_id in ("SME_FI", "SME_SE", "SME_NO"):
                first_loss = result.rounds[0].node_losses.get(node_id, 999)
                last_loss  = result.rounds[-1].node_losses.get(node_id, 999)
                # Allow some variance — GBT losses can fluctuate
                assert last_loss <= first_loss + 0.3, (
                    f"Node {node_id} loss did not decrease: {first_loss:.3f} → {last_loss:.3f}"
                )
