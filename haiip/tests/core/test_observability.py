"""Brutal tests for observability: PredictionCostModel + HAIIPTracer.

Coverage (cost_model):
- compute() formula correctness for true positive, false negative
- fleet_roi() aggregation
- storage cost included in fleet_roi
- Extreme values (latency=0, p=0, p=1)
- to_dict serialisation
- report_id is UUID

Coverage (telemetry):
- HAIIPTracer.span() context manager (no exception)
- SLA breach logged (mocked logger)
- instrument() decorator works
- Noop path (tracer=None) works
- get_tracer() returns singleton
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from haiip.observability.cost_model import CostReport, PredictionCostModel
from haiip.observability.telemetry import HAIIPTracer, get_tracer


# ── PredictionCostModel ────────────────────────────────────────────────────────

class TestPredictionCostModel:
    @pytest.fixture
    def model(self) -> PredictionCostModel:
        return PredictionCostModel(
            gpu_hourly_rate_eur  = 0.50,
            downtime_cost_eur    = 4000.0,
            maintenance_cost_eur = 590.0,
            safety_factor        = 1.5,
        )

    def test_compute_returns_cost_report(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.8)
        assert isinstance(r, CostReport)

    def test_report_id_is_uuid(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.5)
        uuid.UUID(r.report_id)

    def test_compute_cost_formula(self, model: PredictionCostModel) -> None:
        # gpu_rate_per_ms = 0.50 / 3_600_000
        expected = 100.0 * (0.50 / 3_600_000)
        r = model.compute(inference_time_ms=100.0, failure_probability=0.3)
        assert r.compute_cost_eur == pytest.approx(expected, rel=1e-4)

    def test_high_failure_prob_avoided_downtime(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.9, was_correct=True)
        assert r.avoided_downtime_eur > 0

    def test_low_failure_prob_no_avoided_downtime(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.2, was_correct=True)
        assert r.avoided_downtime_eur == pytest.approx(0.0)

    def test_false_negative_cost_nonzero(self, model: PredictionCostModel) -> None:
        # p < 0.5 and was_correct=False → false negative
        r = model.compute(inference_time_ms=50.0, failure_probability=0.3, was_correct=False)
        assert r.false_negative_cost_eur > 0

    def test_true_positive_no_fn_cost(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.9, was_correct=True)
        assert r.false_negative_cost_eur == pytest.approx(0.0)

    def test_zero_latency(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=0.0, failure_probability=0.5)
        assert r.compute_cost_eur == pytest.approx(0.0)

    def test_p_zero(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.0)
        assert r.avoided_downtime_eur == pytest.approx(0.0)

    def test_p_one(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=1.0, was_correct=True)
        assert r.avoided_downtime_eur > 0

    def test_to_dict_keys(self, model: PredictionCostModel) -> None:
        r = model.compute(inference_time_ms=50.0, failure_probability=0.7)
        dct = r.to_dict()
        for key in ("report_id", "compute_cost_eur", "avoided_downtime_eur",
                    "false_negative_cost_eur", "net_value_eur",
                    "inference_time_ms", "failure_probability", "was_correct"):
            assert key in dct

    def test_fleet_roi_empty(self, model: PredictionCostModel) -> None:
        roi = model.fleet_roi([])
        assert roi["net_roi_eur"] == 0.0

    def test_fleet_roi_aggregation(self, model: PredictionCostModel) -> None:
        reports = [
            model.compute(inference_time_ms=50.0, failure_probability=0.9)
            for _ in range(10)
        ]
        roi = model.fleet_roi(reports)
        assert roi["predictions_total"] == 10
        assert "total_compute_cost_eur" in roi
        assert "total_avoided_downtime_eur" in roi
        assert "net_roi_eur" in roi
        assert "avg_inference_ms" in roi

    def test_fleet_roi_storage_cost_included(self, model: PredictionCostModel) -> None:
        reports = [model.compute(50.0, 0.5)]
        roi = model.fleet_roi(reports)
        assert "storage_cost_eur" in roi
        assert roi["storage_cost_eur"] >= 0

    def test_fleet_roi_predictions_per_day(self, model: PredictionCostModel) -> None:
        reports = [model.compute(50.0, 0.5)] * 30
        roi = model.fleet_roi(reports, period_days=30)
        assert roi["predictions_per_day"] == pytest.approx(1.0)


# ── HAIIPTracer ────────────────────────────────────────────────────────────────

class TestHAIIPTracer:
    def test_span_noop_no_exception(self) -> None:
        tracer = HAIIPTracer.__new__(HAIIPTracer)
        tracer.service_name = "test"
        tracer._tracer = None

        with tracer.span("predict", {"machine_id": "M001"}):
            x = 1 + 1
        assert x == 2

    def test_span_catches_and_reraises(self) -> None:
        tracer = HAIIPTracer.__new__(HAIIPTracer)
        tracer.service_name = "test"
        tracer._tracer = None

        with pytest.raises(ValueError, match="boom"):
            with tracer.span("predict"):
                raise ValueError("boom")

    def test_sla_breach_logged(self) -> None:
        tracer = HAIIPTracer.__new__(HAIIPTracer)
        tracer.service_name = "test"
        tracer._tracer = None
        # Set a very tight threshold
        tracer.SLA_THRESHOLDS = {"predict": 0.0001}  # 0.0001 ms

        with patch("haiip.observability.telemetry.logger") as mock_log:
            with tracer.span("predict", check_sla=True):
                pass
            mock_log.warning.assert_called_once()
            call_args = mock_log.warning.call_args
            assert "sla_breach" in call_args[0]

    def test_instrument_decorator(self) -> None:
        tracer = HAIIPTracer.__new__(HAIIPTracer)
        tracer.service_name = "test"
        tracer._tracer = None
        tracer.SLA_THRESHOLDS = {"test_fn": 999999}

        @tracer.instrument("test_fn")
        def my_func(x: int) -> int:
            return x * 2

        assert my_func(5) == 10

    def test_instrument_preserves_function_name(self) -> None:
        tracer = HAIIPTracer.__new__(HAIIPTracer)
        tracer.service_name = "test"
        tracer._tracer = None
        tracer.SLA_THRESHOLDS = {}

        @tracer.instrument("span_name")
        def original_name() -> None: ...

        assert original_name.__name__ == "original_name"

    def test_get_tracer_returns_singleton(self) -> None:
        import haiip.observability.telemetry as tel
        tel._default_tracer = None  # reset
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
        tel._default_tracer = None  # cleanup

    def test_no_sla_breach_within_threshold(self) -> None:
        tracer = HAIIPTracer.__new__(HAIIPTracer)
        tracer.service_name = "test"
        tracer._tracer = None
        tracer.SLA_THRESHOLDS = {"fast_op": 999999}  # huge threshold

        with patch("haiip.observability.telemetry.logger") as mock_log:
            with tracer.span("fast_op", check_sla=True):
                pass
            mock_log.warning.assert_not_called()
