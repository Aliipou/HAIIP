"""Tests for core/feedback.py — FeedbackEngine."""

import pytest

from haiip.core.feedback import FeedbackEngine


@pytest.fixture
def engine():
    return FeedbackEngine(window_size=100, retrain_threshold=0.80, min_samples=10)


def test_init():
    e = FeedbackEngine()
    state = e.get_state()
    assert state.window_size == 0
    assert state.needs_retraining is False


def test_record_correct():
    e = FeedbackEngine(min_samples=1, retrain_threshold=0.8)
    state = e.record("pred-1", was_correct=True)
    assert state.window_size == 1
    assert state.window_accuracy == 1.0


def test_record_incorrect():
    e = FeedbackEngine()
    e.record("pred-1", was_correct=False, corrected_label="anomaly")
    state = e.get_state()
    assert state.window_accuracy < 1.0


def test_needs_retraining_triggered():
    e = FeedbackEngine(window_size=100, retrain_threshold=0.80, min_samples=5)
    for i in range(6):
        e.record(f"pred-{i}", was_correct=False)
    state = e.get_state()
    assert state.needs_retraining is True


def test_no_retrain_below_min_samples():
    e = FeedbackEngine(min_samples=50, retrain_threshold=0.80)
    for i in range(10):
        e.record(f"pred-{i}", was_correct=False)
    state = e.get_state()
    assert state.needs_retraining is False  # only 10 samples, need 50


def test_error_distribution_tracked():
    e = FeedbackEngine()
    e.record("p1", was_correct=False, corrected_label="TWF")
    e.record("p2", was_correct=False, corrected_label="TWF")
    e.record("p3", was_correct=False, corrected_label="HDF")
    state = e.get_state()
    assert state.error_distribution.get("TWF") == 2
    assert state.error_distribution.get("HDF") == 1


def test_adjust_confidence_no_feedback():
    e = FeedbackEngine()
    adjusted = e.adjust_confidence(0.8)
    assert adjusted == 0.8  # unchanged — not enough feedback


def test_adjust_confidence_high_accuracy():
    e = FeedbackEngine(window_size=100, min_samples=5)
    for i in range(20):
        e.record(f"p{i}", was_correct=True)
    adjusted = e.adjust_confidence(0.8)
    assert adjusted >= 0.8  # accuracy is high — confidence boosted


def test_adjust_confidence_low_accuracy():
    e = FeedbackEngine(window_size=100, min_samples=5)
    for i in range(20):
        e.record(f"p{i}", was_correct=False)
    adjusted = e.adjust_confidence(0.8)
    assert adjusted <= 0.8  # accuracy is low — confidence reduced


def test_adjust_confidence_clamped():
    e = FeedbackEngine(window_size=100, min_samples=5)
    for i in range(20):
        e.record(f"p{i}", was_correct=True)
    assert e.adjust_confidence(1.0) <= 1.0
    assert e.adjust_confidence(0.0) >= 0.0


def test_record_batch():
    e = FeedbackEngine()
    records = [{"prediction_id": f"p{i}", "was_correct": i % 2 == 0} for i in range(10)]
    state = e.record_batch(records)
    assert state.window_size == 10
    assert state.cumulative_total == 10


def test_reset_window():
    e = FeedbackEngine(min_samples=3, retrain_threshold=0.8)
    for i in range(5):
        e.record(f"p{i}", was_correct=False)
    e.reset_window()
    state = e.get_state()
    assert state.window_size == 0
    assert state.needs_retraining is False


def test_cumulative_accuracy_tracked():
    e = FeedbackEngine()
    e.record("p1", was_correct=True)
    e.record("p2", was_correct=True)
    e.record("p3", was_correct=False)
    state = e.get_state()
    assert abs(state.cumulative_accuracy - 2 / 3) < 0.001


def test_state_to_dict():
    e = FeedbackEngine()
    e.record("p1", was_correct=True)
    state = e.get_state()
    d = state.to_dict()
    assert "window_accuracy" in d
    assert "needs_retraining" in d
    assert "error_distribution" in d
