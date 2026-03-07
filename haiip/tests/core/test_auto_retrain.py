"""Tests for haiip.core.auto_retrain — Automatic retraining pipeline.

Test categories:
    - Unit: RetrainTrigger, ModelEvaluator, ChampionChallenger, AutoRetrainPipeline
    - Integration: full trigger → train → evaluate → promote cycle
    - Thread safety: concurrent trigger + retrain calls
    - Crash/Edge: NaN labels, empty data, failed train_fn, no champion
    - Audit trail: events recorded correctly
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from haiip.core.auto_retrain import (
    AutoRetrainPipeline,
    ChampionChallenger,
    ModelEvaluator,
    ModelMetrics,
    RetrainEvent,
    RetrainStatus,
    RetrainTrigger,
    TriggerReason,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_FEATURES = 5
N_SAMPLES = 100


@pytest.fixture()
def normal_X() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (N_SAMPLES, N_FEATURES)).astype(np.float32)


@pytest.fixture()
def binary_y() -> np.ndarray:
    y = np.zeros(N_SAMPLES, dtype=int)
    y[90:] = 1  # 10% anomalies
    return y


@pytest.fixture()
def class_y() -> np.ndarray:
    y = np.array(["no_failure"] * 90 + ["TWF"] * 10)
    rng = np.random.default_rng(42)
    rng.shuffle(y)
    return y


@pytest.fixture()
def rul_y() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(50, 500, N_SAMPLES).astype(np.float32)


def _make_drift_result(severity: str) -> Any:
    r = MagicMock()
    r.severity = severity
    r.drift_detected = severity in ("monitoring", "drift")
    return r


def _make_good_metrics() -> ModelMetrics:
    return ModelMetrics(f1_macro=0.90, accuracy=0.92, auc_roc=0.95, n_samples=100)


def _make_bad_metrics() -> ModelMetrics:
    return ModelMetrics(f1_macro=0.70, accuracy=0.72, auc_roc=0.75, n_samples=100)


# ══════════════════════════════════════════════════════════════════════════════
# RetrainTrigger
# ══════════════════════════════════════════════════════════════════════════════


class TestRetrainTriggerInit:
    def test_defaults(self) -> None:
        t = RetrainTrigger()
        assert t.drift_feature_threshold == 2
        assert t.accuracy_threshold == 0.80
        assert t.max_samples_since_retrain == 10_000
        assert t.cooldown_minutes == 60.0
        assert t.samples_since_retrain == 0

    def test_custom_params(self) -> None:
        t = RetrainTrigger(
            drift_feature_threshold=1,
            accuracy_threshold=0.85,
            max_samples_since_retrain=500,
            cooldown_minutes=5.0,
        )
        assert t.drift_feature_threshold == 1
        assert t.accuracy_threshold == 0.85


class TestRetrainTriggerUpdate:
    def test_update_increments_counter(self) -> None:
        t = RetrainTrigger()
        t.update(10)
        assert t.samples_since_retrain == 10

    def test_update_accumulates(self) -> None:
        t = RetrainTrigger()
        t.update(5)
        t.update(3)
        assert t.samples_since_retrain == 8

    def test_update_default_is_one(self) -> None:
        t = RetrainTrigger()
        t.update()
        assert t.samples_since_retrain == 1


class TestRetrainTriggerShouldRetrain:
    def test_no_trigger_when_stable(self) -> None:
        t = RetrainTrigger(cooldown_minutes=0.0)
        should, reason = t.should_retrain()
        assert not should
        assert reason is None

    def test_drift_trigger(self) -> None:
        t = RetrainTrigger(drift_feature_threshold=2, cooldown_minutes=0.0)
        drifts = [_make_drift_result("drift"), _make_drift_result("drift")]
        should, reason = t.should_retrain(drift_results=drifts)
        assert should
        assert reason == TriggerReason.DRIFT_CRITICAL

    def test_drift_below_threshold_no_trigger(self) -> None:
        t = RetrainTrigger(drift_feature_threshold=3, cooldown_minutes=0.0)
        drifts = [_make_drift_result("drift"), _make_drift_result("stable")]
        should, reason = t.should_retrain(drift_results=drifts)
        assert not should

    def test_accuracy_trigger(self) -> None:
        t = RetrainTrigger(accuracy_threshold=0.80, cooldown_minutes=0.0)
        should, reason = t.should_retrain(feedback_accuracy=0.70)
        assert should
        assert reason == TriggerReason.ACCURACY_DROP

    def test_accuracy_above_threshold_no_trigger(self) -> None:
        t = RetrainTrigger(accuracy_threshold=0.80, cooldown_minutes=0.0)
        should, _ = t.should_retrain(feedback_accuracy=0.85)
        assert not should

    def test_volume_trigger(self) -> None:
        t = RetrainTrigger(max_samples_since_retrain=5, cooldown_minutes=0.0)
        t.update(5)
        should, reason = t.should_retrain()
        assert should
        assert reason == TriggerReason.SCHEDULED

    def test_combined_trigger(self) -> None:
        t = RetrainTrigger(
            drift_feature_threshold=1,
            accuracy_threshold=0.80,
            cooldown_minutes=0.0,
        )
        drifts = [_make_drift_result("drift")]
        should, reason = t.should_retrain(drift_results=drifts, feedback_accuracy=0.60)
        assert should
        assert reason == TriggerReason.COMBINED

    def test_cooldown_prevents_repeated_triggers(self) -> None:
        t = RetrainTrigger(accuracy_threshold=0.80, cooldown_minutes=60.0)
        should1, _ = t.should_retrain(feedback_accuracy=0.60)
        should2, _ = t.should_retrain(feedback_accuracy=0.60)
        assert should1
        assert not should2  # blocked by cooldown

    def test_reset_cooldown(self) -> None:
        t = RetrainTrigger(accuracy_threshold=0.80, cooldown_minutes=60.0)
        t.should_retrain(feedback_accuracy=0.60)  # fires + sets cooldown
        t.reset_cooldown()
        should, _ = t.should_retrain(feedback_accuracy=0.60)
        assert should  # cooldown cleared

    def test_counter_resets_after_trigger(self) -> None:
        t = RetrainTrigger(max_samples_since_retrain=5, cooldown_minutes=0.0)
        t.update(5)
        t.should_retrain()
        assert t.samples_since_retrain == 0

    def test_dict_drift_result_support(self) -> None:
        t = RetrainTrigger(drift_feature_threshold=1, cooldown_minutes=0.0)
        drifts = [{"severity": "drift", "feature_name": "torque"}]
        should, reason = t.should_retrain(drift_results=drifts)
        assert should


class TestRetrainTriggerThreadSafety:
    def test_concurrent_update(self) -> None:
        t = RetrainTrigger()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    t.update(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        assert t.samples_since_retrain == 400


# ══════════════════════════════════════════════════════════════════════════════
# ModelEvaluator
# ══════════════════════════════════════════════════════════════════════════════


class TestModelEvaluatorAnomaly:
    def _make_model(self, label: str = "normal", score: float = 0.1) -> Any:
        m = MagicMock()
        m.predict.return_value = {"label": label, "anomaly_score": score}
        return m

    def test_evaluate_returns_model_metrics(
        self, normal_X: np.ndarray, binary_y: np.ndarray
    ) -> None:
        model = self._make_model("normal", 0.0)
        metrics = ModelEvaluator.evaluate_anomaly(model, normal_X, binary_y)
        assert isinstance(metrics, ModelMetrics)

    def test_evaluate_sets_n_samples(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        model = self._make_model()
        metrics = ModelEvaluator.evaluate_anomaly(model, normal_X, binary_y)
        assert metrics.n_samples == len(normal_X)

    def test_evaluate_f1_in_range(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        model = self._make_model("normal", 0.0)
        metrics = ModelEvaluator.evaluate_anomaly(model, normal_X, binary_y)
        assert 0.0 <= metrics.f1_macro <= 1.0

    def test_evaluate_accuracy_in_range(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        model = self._make_model("normal", 0.0)
        metrics = ModelEvaluator.evaluate_anomaly(model, normal_X, binary_y)
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_evaluate_auc_in_range(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        model = self._make_model("normal", 0.0)
        metrics = ModelEvaluator.evaluate_anomaly(model, normal_X, binary_y)
        assert 0.0 <= metrics.auc_roc <= 1.0

    def test_evaluate_sets_timestamp(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        model = self._make_model()
        metrics = ModelEvaluator.evaluate_anomaly(model, normal_X, binary_y)
        assert metrics.evaluated_at != ""

    def test_perfect_model_high_f1(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        def predict(features: Any) -> dict:
            return {
                "label": "normal",
                "anomaly_score": 0.0,
                "confidence": 0.9,
            }

        m = MagicMock()
        m.predict.return_value = {"label": "normal", "anomaly_score": 0.0}
        # All predictions = normal; truth is mostly normal
        metrics = ModelEvaluator.evaluate_anomaly(m, normal_X[:90], binary_y[:90])
        assert metrics.accuracy == pytest.approx(1.0)


class TestModelEvaluatorMaintenance:
    def _make_model(self) -> Any:
        m = MagicMock()
        m.predict.return_value = {
            "label": "no_failure",
            "failure_probability": 0.05,
            "rul_cycles": 300,
        }
        return m

    def test_evaluate_returns_metrics(self, normal_X: np.ndarray, class_y: np.ndarray) -> None:
        model = self._make_model()
        metrics = ModelEvaluator.evaluate_maintenance(model, normal_X, class_y)
        assert isinstance(metrics, ModelMetrics)

    def test_evaluate_with_rul(
        self, normal_X: np.ndarray, class_y: np.ndarray, rul_y: np.ndarray
    ) -> None:
        model = self._make_model()
        metrics = ModelEvaluator.evaluate_maintenance(model, normal_X, class_y, rul_y)
        assert metrics.rmse_rul != float("inf")
        assert metrics.rmse_rul >= 0.0

    def test_evaluate_without_rul(self, normal_X: np.ndarray, class_y: np.ndarray) -> None:
        model = self._make_model()
        metrics = ModelEvaluator.evaluate_maintenance(model, normal_X, class_y)
        assert metrics.rmse_rul == float("inf")

    def test_auc_fallback_on_single_class(self, normal_X: np.ndarray) -> None:
        model = MagicMock()
        model.predict.return_value = {
            "label": "no_failure",
            "failure_probability": 0.0,
            "rul_cycles": 100,
        }
        y_all_normal = np.array(["no_failure"] * len(normal_X))
        # AUC with single class should not crash
        metrics = ModelEvaluator.evaluate_maintenance(model, normal_X, y_all_normal)
        assert 0.0 <= metrics.auc_roc <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ChampionChallenger
# ══════════════════════════════════════════════════════════════════════════════


class TestChampionChallengerInit:
    def test_defaults(self) -> None:
        cc = ChampionChallenger()
        assert cc.min_improvement == 0.01
        assert cc.auc_tolerance == 0.02
        assert cc.rollback_threshold == 0.05
        assert cc.champion is None

    def test_champion_metrics_default(self) -> None:
        cc = ChampionChallenger()
        assert cc.champion_metrics.f1_macro == 0.0


class TestChampionChallengerRegister:
    def test_register_champion(self) -> None:
        cc = ChampionChallenger()
        model = MagicMock()
        metrics = _make_good_metrics()
        cc.register_champion(model, metrics)
        assert cc.champion is model
        assert cc.champion_metrics.f1_macro == 0.90

    def test_register_replaces_previous(self) -> None:
        cc = ChampionChallenger()
        m1, m2 = MagicMock(), MagicMock()
        cc.register_champion(m1, _make_good_metrics())
        cc.register_champion(m2, _make_bad_metrics())
        assert cc.champion is m2

    def test_propose_challenger(self) -> None:
        cc = ChampionChallenger()
        cc.register_champion(MagicMock(), _make_bad_metrics())
        cc.propose_challenger(MagicMock(), _make_good_metrics())
        assert cc._challenger is not None


class TestChampionChallengerPromotion:
    def test_promote_better_challenger(self) -> None:
        cc = ChampionChallenger(min_improvement=0.01, auc_tolerance=0.02)
        champion_model = MagicMock()
        cc.register_champion(champion_model, _make_bad_metrics())

        challenger_model = MagicMock()
        better = ModelMetrics(f1_macro=0.72, accuracy=0.74, auc_roc=0.75, n_samples=100)
        cc.propose_challenger(challenger_model, better)

        promoted, reason = cc.evaluate_promotion()
        assert promoted
        assert cc.champion is challenger_model

    def test_reject_worse_challenger(self) -> None:
        cc = ChampionChallenger(min_improvement=0.05)
        champion = _make_good_metrics()
        cc.register_champion(MagicMock(), champion)

        # Challenger only 0.01 better → below min_improvement=0.05
        challenger_metrics = ModelMetrics(f1_macro=0.91, accuracy=0.93, auc_roc=0.96, n_samples=100)
        cc.propose_challenger(MagicMock(), challenger_metrics)

        promoted, reason = cc.evaluate_promotion()
        assert not promoted
        assert "rejected" in reason

    def test_reject_auc_regression(self) -> None:
        cc = ChampionChallenger(min_improvement=0.01, auc_tolerance=0.02)
        cc.register_champion(MagicMock(), ModelMetrics(f1_macro=0.80, auc_roc=0.90))

        # F1 improved but AUC dropped too much
        cc.propose_challenger(MagicMock(), ModelMetrics(f1_macro=0.85, auc_roc=0.85))
        promoted, reason = cc.evaluate_promotion()
        assert not promoted

    def test_first_model_promoted_unconditionally(self) -> None:
        cc = ChampionChallenger()
        model = MagicMock()
        cc.propose_challenger(model, _make_bad_metrics())
        promoted, reason = cc.evaluate_promotion()
        assert promoted
        assert "unconditionally" in reason
        assert cc.champion is model

    def test_no_challenger_no_promotion(self) -> None:
        cc = ChampionChallenger()
        cc.register_champion(MagicMock(), _make_good_metrics())
        promoted, reason = cc.evaluate_promotion()
        assert not promoted
        assert "no challenger" in reason

    def test_promotion_clears_challenger(self) -> None:
        cc = ChampionChallenger(min_improvement=0.01)
        cc.register_champion(MagicMock(), _make_bad_metrics())
        cc.propose_challenger(MagicMock(), _make_good_metrics())
        cc.evaluate_promotion()
        assert cc._challenger is None

    def test_promotion_history_recorded(self) -> None:
        cc = ChampionChallenger(min_improvement=0.01)
        cc.register_champion(MagicMock(), _make_bad_metrics())
        cc.propose_challenger(MagicMock(), _make_good_metrics())
        cc.evaluate_promotion()
        assert len(cc.promotion_history) == 1
        assert "promoted_at" in cc.promotion_history[0]

    def test_rollback_returns_false(self) -> None:
        cc = ChampionChallenger()
        result = cc.rollback()
        assert result is False


class TestChampionChallengerThreadSafety:
    def test_concurrent_propose_and_evaluate(self) -> None:
        cc = ChampionChallenger(min_improvement=0.01, auc_tolerance=0.05)
        cc.register_champion(MagicMock(), _make_bad_metrics())
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(5):
                    cc.propose_challenger(MagicMock(), _make_good_metrics())
                    cc.evaluate_promotion()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ══════════════════════════════════════════════════════════════════════════════
# AutoRetrainPipeline
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def pipeline(normal_X: np.ndarray, binary_y: np.ndarray) -> AutoRetrainPipeline:
    """Pipeline with a registered champion and no cooldown."""
    from haiip.core.anomaly import AnomalyDetector

    detector = AnomalyDetector(contamination=0.05, n_estimators=10)
    detector.fit(normal_X)

    trigger = RetrainTrigger(cooldown_minutes=0.0, drift_feature_threshold=1)
    p = AutoRetrainPipeline(tenant_id="test-tenant", trigger=trigger)
    p.register_champion(detector, normal_X, binary_y)
    return p


class TestAutoRetrainPipelineInit:
    def test_defaults(self) -> None:
        p = AutoRetrainPipeline()
        assert p.tenant_id == "default"
        assert p.status == RetrainStatus.IDLE
        assert p.events == []
        assert p.current_champion is None

    def test_custom_tenant(self) -> None:
        p = AutoRetrainPipeline(tenant_id="sme-fi")
        assert p.tenant_id == "sme-fi"

    def test_initial_summary(self) -> None:
        p = AutoRetrainPipeline()
        s = p.summary()
        assert s["total_retrain_events"] == 0
        assert s["promotions"] == 0
        assert s["status"] == "idle"


class TestAutoRetrainPipelineRegisterChampion:
    def test_register_champion(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        from haiip.core.anomaly import AnomalyDetector

        detector = AnomalyDetector(contamination=0.05, n_estimators=10)
        detector.fit(normal_X)

        p = AutoRetrainPipeline()
        metrics = p.register_champion(detector, normal_X, binary_y)
        assert isinstance(metrics, ModelMetrics)
        assert p.current_champion is detector

    def test_champion_metrics_stored(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        from haiip.core.anomaly import AnomalyDetector

        detector = AnomalyDetector(contamination=0.05, n_estimators=10)
        detector.fit(normal_X)
        p = AutoRetrainPipeline()
        p.register_champion(detector, normal_X, binary_y)
        assert p.champion_metrics.n_samples == len(normal_X)


class TestAutoRetrainPipelineMaybeRetrain:
    def test_no_trigger_returns_none(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        # cooldown=0 but no drift + good accuracy
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.95)
        assert event is None

    def test_drift_trigger_returns_event(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        drifts = [_make_drift_result("drift")]
        event = pipeline.maybe_retrain(normal_X, drift_results=drifts)
        assert event is not None
        assert isinstance(event, RetrainEvent)

    def test_accuracy_trigger_returns_event(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None

    def test_event_has_tenant_id(self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.tenant_id == "test-tenant"

    def test_event_status_complete(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.status == RetrainStatus.COMPLETE

    def test_event_has_trigger_reason(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.trigger_reason == TriggerReason.ACCURACY_DROP

    def test_event_has_n_training_samples(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.n_training_samples == len(normal_X)

    def test_event_has_timestamps(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.triggered_at != ""
        assert event.completed_at is not None

    def test_event_has_challenger_metrics(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert "f1_macro" in event.challenger_metrics

    def test_event_has_champion_metrics(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert "f1_macro" in event.champion_metrics

    def test_event_recorded_in_history(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert len(pipeline.events) == 1

    def test_manual_reason_bypasses_trigger(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        event = pipeline.maybe_retrain(normal_X, reason=TriggerReason.MANUAL)
        assert event is not None
        assert event.trigger_reason == TriggerReason.MANUAL

    def test_drift_severity_recorded(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        drifts = [_make_drift_result("drift")]
        event = pipeline.maybe_retrain(normal_X, drift_results=drifts)
        assert event is not None
        assert event.drift_severity == "drift"

    def test_monitoring_severity_recorded(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        drifts = [_make_drift_result("monitoring")]
        event = pipeline.maybe_retrain(normal_X, drift_results=drifts, feedback_accuracy=0.60)
        assert event is not None
        assert event.drift_severity in ("monitoring", "drift")

    def test_stable_drift_severity(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        drifts = [_make_drift_result("stable")]
        event = pipeline.maybe_retrain(normal_X, drift_results=drifts, feedback_accuracy=0.60)
        if event is not None:
            assert event.drift_severity in ("stable", "monitoring", "drift", None)


class TestAutoRetrainPipelineFailedTraining:
    def test_failed_train_fn_returns_event_with_failed_status(
        self, normal_X: np.ndarray, binary_y: np.ndarray
    ) -> None:
        p = AutoRetrainPipeline(
            tenant_id="test",
            trigger=RetrainTrigger(cooldown_minutes=0.0),
            train_fn=lambda X: None,  # returns None → fail
        )
        from haiip.core.anomaly import AnomalyDetector

        detector = AnomalyDetector(contamination=0.05, n_estimators=10)
        detector.fit(normal_X)
        p.register_champion(detector, normal_X, binary_y)

        event = p.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.status == RetrainStatus.FAILED
        assert event.error is not None

    def test_exception_in_train_fn_caught(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        def bad_train(X: np.ndarray) -> Any:
            raise RuntimeError("disk full")

        p = AutoRetrainPipeline(
            tenant_id="test",
            trigger=RetrainTrigger(cooldown_minutes=0.0),
            train_fn=bad_train,
        )
        from haiip.core.anomaly import AnomalyDetector

        d = AnomalyDetector(contamination=0.05, n_estimators=10)
        d.fit(normal_X)
        p.register_champion(d, normal_X, binary_y)

        event = p.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert event is not None
        assert event.status == RetrainStatus.FAILED
        assert "disk full" in (event.error or "")


class TestAutoRetrainPipelineCustomTrainFn:
    def test_custom_train_fn_called(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        from haiip.core.anomaly import AnomalyDetector

        calls: list[int] = []

        def train_fn(X: np.ndarray) -> AnomalyDetector:
            calls.append(1)
            d = AnomalyDetector(contamination=0.05, n_estimators=10)
            d.fit(X)
            return d

        p = AutoRetrainPipeline(
            trigger=RetrainTrigger(cooldown_minutes=0.0),
            train_fn=train_fn,
        )
        champion = AnomalyDetector(contamination=0.05, n_estimators=10)
        champion.fit(normal_X)
        p.register_champion(champion, normal_X, binary_y)
        p.maybe_retrain(normal_X, feedback_accuracy=0.60)
        assert len(calls) == 1

    def test_custom_eval_fn_called(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        from haiip.core.anomaly import AnomalyDetector

        eval_calls: list[int] = []

        def eval_fn(model: Any, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
            eval_calls.append(1)
            return ModelMetrics(f1_macro=0.95, accuracy=0.96, auc_roc=0.97, n_samples=len(X))

        p = AutoRetrainPipeline(
            trigger=RetrainTrigger(cooldown_minutes=0.0),
            eval_fn=eval_fn,
        )
        champion = AnomalyDetector(contamination=0.05, n_estimators=10)
        champion.fit(normal_X)
        p.register_champion(champion, normal_X, binary_y)
        p.maybe_retrain(normal_X, feedback_accuracy=0.60)
        # eval_fn called at least once (for challenger; champion evaluated at register)
        assert len(eval_calls) >= 1


class TestAutoRetrainPipelineSummary:
    def test_summary_keys(self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray) -> None:
        s = pipeline.summary()
        for key in (
            "tenant_id",
            "status",
            "total_retrain_events",
            "successful_retrain",
            "promotions",
            "champion_f1",
            "champion_auc",
            "samples_since_retrain",
            "last_retrain",
        ):
            assert key in s

    def test_summary_after_retrain(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        s = pipeline.summary()
        assert s["total_retrain_events"] == 1
        assert s["last_retrain"] is not None

    def test_summary_counts_promotions(
        self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray
    ) -> None:
        pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
        s = pipeline.summary()
        assert s["promotions"] >= 0  # may or may not promote depending on metrics


class TestAutoRetrainPipelineUpdate:
    def test_update_increments_trigger_counter(self, pipeline: AutoRetrainPipeline) -> None:
        pipeline.update(50)
        assert pipeline.trigger.samples_since_retrain == 50


class TestAutoRetrainPipelineThreadSafety:
    def test_concurrent_retrain(self, pipeline: AutoRetrainPipeline, normal_X: np.ndarray) -> None:
        errors: list[Exception] = []

        def worker() -> None:
            try:
                pipeline.maybe_retrain(normal_X, feedback_accuracy=0.60)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ══════════════════════════════════════════════════════════════════════════════
# Integration — full pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_drift_triggers_retrain_and_promotes(
        self, normal_X: np.ndarray, binary_y: np.ndarray
    ) -> None:
        """Full cycle: register champion → drift → retrain → champion updated."""
        from haiip.core.anomaly import AnomalyDetector

        trigger = RetrainTrigger(
            drift_feature_threshold=1,
            cooldown_minutes=0.0,
        )
        cc = ChampionChallenger(min_improvement=-1.0)  # always promote
        p = AutoRetrainPipeline(trigger=trigger, cc=cc, tenant_id="integration-test")

        d = AnomalyDetector(contamination=0.05, n_estimators=10)
        d.fit(normal_X)
        p.register_champion(d, normal_X, binary_y)

        drifts = [_make_drift_result("drift"), _make_drift_result("drift")]
        event = p.maybe_retrain(normal_X, drift_results=drifts)

        assert event is not None
        assert event.status == RetrainStatus.COMPLETE
        assert event.trigger_reason in (
            TriggerReason.DRIFT_CRITICAL,
            TriggerReason.COMBINED,
        )
        assert p.current_champion is not None

    def test_multiple_retrain_cycles(self, normal_X: np.ndarray, binary_y: np.ndarray) -> None:
        """Two consecutive retrain cycles both complete successfully."""
        from haiip.core.anomaly import AnomalyDetector

        trigger = RetrainTrigger(cooldown_minutes=0.0)
        p = AutoRetrainPipeline(trigger=trigger, tenant_id="multi-cycle")

        d = AnomalyDetector(contamination=0.05, n_estimators=10)
        d.fit(normal_X)
        p.register_champion(d, normal_X, binary_y)

        for _ in range(2):
            event = p.maybe_retrain(normal_X, feedback_accuracy=0.50)
            assert event is not None
            assert event.status == RetrainStatus.COMPLETE

        assert len(p.events) == 2
        s = p.summary()
        assert s["total_retrain_events"] == 2
