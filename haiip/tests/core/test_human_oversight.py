"""Brutal tests for HumanOversightEngine — HIR, HOG, TCS metrics.

Coverage:
- HIR formula correctness (0, 0.5, 1.0)
- HOG sign (positive, zero, negative)
- TCS / ECE calibration bins
- HIR by action category
- Risk reduction (€ saved)
- Rolling HIR trend
- Batch recording
- Edge cases (perfect AI, all overrides, empty error)
- Report string contains key fields
- to_dict serialisation
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from haiip.core.human_oversight import (
    HumanOversightEngine,
    OversightEvent,
    OversightMetrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_event(
    ai_label: str = "normal",
    true_label: str = "normal",
    ai_confidence: float = 0.8,
    human_reviewed: bool = False,
    human_overrode: bool = False,
    human_label: str | None = None,
    action_category: str = "monitor",
    expected_cost_ai: float = 100.0,
    expected_cost_human: float | None = None,
) -> OversightEvent:
    return OversightEvent.create(
        decision_id        = str(uuid.uuid4()),
        ai_label           = ai_label,
        ai_confidence      = ai_confidence,
        true_label         = true_label,
        human_reviewed     = human_reviewed,
        human_overrode     = human_overrode,
        human_label        = human_label,
        action_category    = action_category,
        expected_cost_ai   = expected_cost_ai,
        expected_cost_human= expected_cost_human,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> HumanOversightEngine:
    return HumanOversightEngine(target_hir=0.10)


# ── OversightEvent ─────────────────────────────────────────────────────────────

class TestOversightEvent:
    def test_ai_correct_when_labels_match(self) -> None:
        e = make_event(ai_label="failure", true_label="failure")
        assert e.ai_correct is True

    def test_ai_incorrect_when_labels_differ(self) -> None:
        e = make_event(ai_label="failure", true_label="normal")
        assert e.ai_correct is False

    def test_human_correct_uses_human_label(self) -> None:
        e = make_event(
            ai_label="failure", true_label="normal",
            human_overrode=True, human_label="normal",
        )
        assert e.human_correct is True

    def test_human_correct_defaults_to_ai_when_not_overridden(self) -> None:
        e = make_event(ai_label="normal", true_label="normal", human_reviewed=True)
        assert e.human_correct is True

    def test_event_id_is_uuid(self) -> None:
        e = make_event()
        uuid.UUID(e.event_id)  # should not raise

    def test_confidence_clamped_above(self) -> None:
        e = make_event(ai_confidence=1.5)
        assert e.ai_confidence == pytest.approx(1.0)

    def test_confidence_clamped_below(self) -> None:
        e = make_event(ai_confidence=-0.3)
        assert e.ai_confidence == pytest.approx(0.0)


# ── HIR ────────────────────────────────────────────────────────────────────────

class TestHIR:
    def test_hir_zero_no_reviews(self, engine: HumanOversightEngine) -> None:
        for _ in range(5):
            engine.record(make_event(human_reviewed=False))
        m = engine.compute_metrics()
        assert m.hir == pytest.approx(0.0)
        engine.clear()

    def test_hir_one_all_reviewed(self, engine: HumanOversightEngine) -> None:
        for _ in range(10):
            engine.record(make_event(human_reviewed=True))
        m = engine.compute_metrics()
        assert m.hir == pytest.approx(1.0)
        engine.clear()

    def test_hir_half(self, engine: HumanOversightEngine) -> None:
        for _ in range(5):
            engine.record(make_event(human_reviewed=True))
        for _ in range(5):
            engine.record(make_event(human_reviewed=False))
        m = engine.compute_metrics()
        assert m.hir == pytest.approx(0.5)
        engine.clear()

    def test_n_reviewed_matches_hir(self, engine: HumanOversightEngine) -> None:
        for _ in range(3):
            engine.record(make_event(human_reviewed=True))
        for _ in range(7):
            engine.record(make_event(human_reviewed=False))
        m = engine.compute_metrics()
        assert m.n_reviewed == 3
        assert m.n_events == 10
        engine.clear()


# ── HOG ────────────────────────────────────────────────────────────────────────

class TestHOG:
    def test_hog_positive_when_human_fixes_mistakes(
        self, engine: HumanOversightEngine
    ) -> None:
        # AI wrong, human corrects → positive HOG
        for _ in range(5):
            engine.record(make_event(
                ai_label="failure", true_label="normal",
                human_reviewed=True, human_overrode=True, human_label="normal",
            ))
        for _ in range(5):
            engine.record(make_event(
                ai_label="normal", true_label="normal",
                human_reviewed=False,
            ))
        m = engine.compute_metrics()
        assert m.hog > 0
        engine.clear()

    def test_hog_zero_when_ai_perfect(self, engine: HumanOversightEngine) -> None:
        for _ in range(10):
            engine.record(make_event(
                ai_label="normal", true_label="normal",
                human_reviewed=False,
            ))
        m = engine.compute_metrics()
        assert m.hog == pytest.approx(0.0)
        engine.clear()

    def test_hog_negative_when_human_makes_worse(
        self, engine: HumanOversightEngine
    ) -> None:
        # AI correct, human overrides to wrong answer
        for _ in range(5):
            engine.record(make_event(
                ai_label="failure", true_label="failure",
                human_reviewed=True, human_overrode=True, human_label="normal",
            ))
        m = engine.compute_metrics()
        assert m.hog < 0
        engine.clear()


# ── TCS / ECE ──────────────────────────────────────────────────────────────────

class TestCalibration:
    def test_perfect_calibration_tcs_near_one(
        self, engine: HumanOversightEngine
    ) -> None:
        # High confidence on correct predictions → low ECE → high TCS
        for _ in range(50):
            engine.record(make_event(
                ai_label="normal", true_label="normal", ai_confidence=0.95,
            ))
        m = engine.compute_metrics()
        assert m.tcs >= 0.7  # well calibrated
        engine.clear()

    def test_poor_calibration_high_ece(
        self, engine: HumanOversightEngine
    ) -> None:
        # High confidence on wrong predictions → high ECE → low TCS
        for _ in range(50):
            engine.record(make_event(
                ai_label="failure", true_label="normal", ai_confidence=0.95,
            ))
        m = engine.compute_metrics()
        assert m.ece > 0.1  # miscalibrated
        engine.clear()

    def test_tcs_bounded(self, engine: HumanOversightEngine) -> None:
        engine.record(make_event())
        m = engine.compute_metrics()
        assert 0.0 <= m.tcs <= 1.0
        assert m.ece >= 0.0
        engine.clear()


# ── HIR by action ──────────────────────────────────────────────────────────────

class TestHIRByAction:
    def test_hir_decomposed_by_category(
        self, engine: HumanOversightEngine
    ) -> None:
        for _ in range(5):
            engine.record(make_event(
                human_reviewed=True, action_category="repair_now"
            ))
        for _ in range(5):
            engine.record(make_event(
                human_reviewed=False, action_category="monitor"
            ))
        m = engine.compute_metrics()
        assert "repair_now" in m.hir_by_action
        assert "monitor" in m.hir_by_action
        assert m.hir_by_action["repair_now"] == pytest.approx(1.0)
        assert m.hir_by_action["monitor"] == pytest.approx(0.0)
        engine.clear()


# ── Risk reduction ─────────────────────────────────────────────────────────────

class TestRiskReduction:
    def test_positive_risk_reduction_when_human_saves_cost(
        self, engine: HumanOversightEngine
    ) -> None:
        for _ in range(5):
            engine.record(make_event(
                human_overrode=True,
                expected_cost_ai=1000.0,
                expected_cost_human=200.0,
            ))
        m = engine.compute_metrics()
        assert m.risk_reduction_pct > 0
        engine.clear()

    def test_zero_risk_reduction_no_overrides(
        self, engine: HumanOversightEngine
    ) -> None:
        for _ in range(5):
            engine.record(make_event(
                human_reviewed=False,
                expected_cost_ai=500.0,
            ))
        m = engine.compute_metrics()
        assert m.risk_reduction_pct == pytest.approx(0.0)
        engine.clear()


# ── Rolling HIR ────────────────────────────────────────────────────────────────

class TestRollingHIR:
    def test_rolling_hir_length(self, engine: HumanOversightEngine) -> None:
        for i in range(60):
            engine.record(make_event(human_reviewed=(i % 3 == 0)))
        hir_series = engine.rolling_hir(window=20)
        assert len(hir_series) == 60 - 20 + 1
        engine.clear()

    def test_rolling_hir_values_bounded(
        self, engine: HumanOversightEngine
    ) -> None:
        for _ in range(30):
            engine.record(make_event(human_reviewed=True))
        hir_series = engine.rolling_hir(window=10)
        for v in hir_series:
            assert 0.0 <= v <= 1.0
        engine.clear()

    def test_rolling_hir_empty_returns_empty(
        self, engine: HumanOversightEngine
    ) -> None:
        assert engine.rolling_hir() == []


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_raises(self, engine: HumanOversightEngine) -> None:
        with pytest.raises(ValueError, match="No oversight events"):
            engine.compute_metrics()

    def test_single_event(self, engine: HumanOversightEngine) -> None:
        engine.record(make_event(human_reviewed=True))
        m = engine.compute_metrics()
        assert m.n_events == 1
        engine.clear()

    def test_batch_record(self, engine: HumanOversightEngine) -> None:
        events = [make_event() for _ in range(10)]
        engine.record_batch(events)
        assert engine.event_count == 10
        engine.clear()

    def test_clear_resets(self, engine: HumanOversightEngine) -> None:
        engine.record(make_event())
        engine.clear()
        assert engine.event_count == 0

    def test_to_dict_has_all_keys(self, engine: HumanOversightEngine) -> None:
        engine.record(make_event(human_reviewed=True))
        m = engine.compute_metrics()
        dct = m.to_dict()
        for key in ("HIR", "HOG", "TCS", "ECE", "HIR_by_action",
                    "ai_accuracy", "human_accuracy", "risk_reduction_pct"):
            assert key in dct, f"Missing key: {key}"
        engine.clear()

    def test_report_contains_key_fields(self, engine: HumanOversightEngine) -> None:
        engine.record(make_event(human_reviewed=True))
        m = engine.compute_metrics()
        assert "HIR" in m.report
        assert "HOG" in m.report
        assert "TCS" in m.report
        assert "EU AI Act" in m.report
        engine.clear()
