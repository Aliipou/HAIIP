"""Tests for ActiveLearningSampler and LabelingQueue — 100% branch coverage."""

from __future__ import annotations

import numpy as np
import pytest

from haiip.core.active_learning import (
    ActiveLearningSampler,
    LabelingQueue,
    QueryBatch,
    STRATEGIES,
)


def _preds(n: int, conf_values: list[float] | None = None) -> list[dict]:
    """Build n prediction dicts."""
    if conf_values is None:
        conf_values = [0.5 + i * 0.05 for i in range(n)]
    return [
        {"label": "anomaly" if i % 2 == 0 else "normal", "confidence": conf_values[i]}
        for i in range(n)
    ]


class TestActiveLearningStrategies:
    def test_all_strategies_valid(self):
        assert "uncertainty" in STRATEGIES
        assert "margin" in STRATEGIES
        assert "entropy" in STRATEGIES
        assert "coreset" in STRATEGIES
        assert "random" in STRATEGIES

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ActiveLearningSampler(strategy="bogus")

    def test_uncertainty_selects_lowest_confidence(self):
        sampler = ActiveLearningSampler(strategy="uncertainty", budget=3)
        preds = _preds(10, [0.51, 0.52, 0.53, 0.90, 0.91, 0.92, 0.93, 0.94, 0.55, 0.56])
        batch = sampler.select(preds)
        # Should select lowest confidence (closest to 0.5)
        assert 0 in batch.indices  # conf=0.51 → uncertainty=0.49
        assert batch.size == 3

    def test_margin_selects_closest_to_decision_boundary(self):
        sampler = ActiveLearningSampler(strategy="margin", budget=2)
        # conf=0.5 → margin=0 (most uncertain), conf=0.99 → margin≈0.98
        preds = [
            {"label": "normal", "confidence": 0.50},  # most uncertain
            {"label": "anomaly", "confidence": 0.99},  # most certain
            {"label": "normal", "confidence": 0.52},  # second most uncertain
        ]
        batch = sampler.select(preds)
        assert 0 in batch.indices  # conf=0.5 → smallest margin
        assert batch.size == 2

    def test_entropy_selects_highest_entropy(self):
        sampler = ActiveLearningSampler(strategy="entropy", budget=2)
        preds = [
            {"label": "anomaly", "confidence": 0.5},   # max entropy
            {"label": "normal", "confidence": 0.5},    # max entropy
            {"label": "anomaly", "confidence": 0.99},  # low entropy
        ]
        batch = sampler.select(preds)
        assert 0 in batch.indices or 1 in batch.indices
        assert batch.size == 2

    def test_random_sampling(self):
        sampler = ActiveLearningSampler(strategy="random", budget=5, random_state=0)
        preds = _preds(20)
        batch = sampler.select(preds)
        assert batch.size == 5
        assert len(set(batch.indices)) == 5  # unique indices

    def test_coreset_sampling_with_X(self):
        sampler = ActiveLearningSampler(strategy="coreset", budget=3, random_state=42)
        preds = _preds(10)
        X = np.random.default_rng(42).random((10, 4))
        batch = sampler.select(preds, X=X)
        assert batch.size == 3
        assert len(set(batch.indices)) == 3  # unique

    def test_coreset_fallback_without_X(self):
        sampler = ActiveLearningSampler(strategy="coreset", budget=3)
        preds = _preds(5)
        # No X → falls back to uncertainty
        batch = sampler.select(preds)
        assert batch.size <= 3

    def test_empty_predictions_returns_empty_batch(self):
        sampler = ActiveLearningSampler(strategy="uncertainty", budget=5)
        batch = sampler.select([])
        assert batch.indices == []
        assert batch.scores == []
        assert batch.size == 0

    def test_budget_larger_than_pool_clamped(self):
        sampler = ActiveLearningSampler(strategy="uncertainty", budget=100)
        preds = _preds(5)
        batch = sampler.select(preds)
        assert batch.size == 5  # clamped to pool size

    def test_confidence_floor_filters_low_confidence(self):
        sampler = ActiveLearningSampler(
            strategy="uncertainty", budget=10, confidence_floor=0.6
        )
        preds = _preds(5, [0.3, 0.4, 0.7, 0.8, 0.9])
        batch = sampler.select(preds)
        # Only indices with conf >= 0.6 considered
        for idx in batch.indices:
            assert preds[idx]["confidence"] >= 0.6

    def test_margin_confidence_floor(self):
        sampler = ActiveLearningSampler(
            strategy="margin", budget=5, confidence_floor=0.7
        )
        preds = _preds(5, [0.3, 0.4, 0.75, 0.80, 0.85])
        batch = sampler.select(preds)
        for idx in batch.indices:
            assert preds[idx]["confidence"] >= 0.7

    def test_entropy_confidence_floor(self):
        sampler = ActiveLearningSampler(
            strategy="entropy", budget=5, confidence_floor=0.7
        )
        preds = _preds(5, [0.2, 0.3, 0.75, 0.80, 0.85])
        batch = sampler.select(preds)
        for idx in batch.indices:
            assert preds[idx]["confidence"] >= 0.7

    def test_batch_metadata_populated(self):
        sampler = ActiveLearningSampler(strategy="random", budget=3)
        batch = sampler.select(_preds(10))
        assert batch.metadata["n_pool"] == 10
        assert batch.metadata["n_selected"] == 3

    def test_entropy_normal_label_maps_p_correctly(self):
        # For label="normal", p = 1 - confidence → entropy computed correctly
        sampler = ActiveLearningSampler(strategy="entropy", budget=2)
        preds = [
            {"label": "normal", "confidence": 0.5},
            {"label": "anomaly", "confidence": 0.5},
        ]
        batch = sampler.select(preds)
        # Both have same entropy (max) — both should be selected
        assert batch.size == 2


class TestCoreset:
    def test_coreset_empty_X_returns_empty(self):
        sampler = ActiveLearningSampler(strategy="coreset", budget=3)
        X = np.empty((0, 4))
        indices, scores = sampler._coreset_sampling(X, 3)
        assert indices == []
        assert scores == []

    def test_coreset_selects_diverse_points(self):
        sampler = ActiveLearningSampler(strategy="coreset", budget=3, random_state=0)
        # Clustered at 0 and 100
        X = np.array([[0.0], [0.1], [0.2], [100.0], [100.1]])
        indices, scores = sampler._coreset_sampling(X, 3)
        # Should select one from each cluster + one more
        assert len(indices) <= 3
        # At least one far point (near 100)
        selected_vals = [X[i, 0] for i in indices]
        assert any(v > 50 for v in selected_vals)

    def test_coreset_budget_exceeds_pool(self):
        sampler = ActiveLearningSampler(strategy="coreset", budget=10, random_state=0)
        X = np.random.default_rng(0).random((3, 2))
        indices, scores = sampler._coreset_sampling(X, 10)
        assert len(indices) <= 3


class TestQueryBatch:
    def test_size_property(self):
        batch = QueryBatch(indices=[0, 1, 2], scores=[0.9, 0.8, 0.7], strategy="uncertainty", budget=3)
        assert batch.size == 3

    def test_size_empty(self):
        batch = QueryBatch(indices=[], scores=[], strategy="random", budget=5)
        assert batch.size == 0

    def test_metadata_default_empty(self):
        batch = QueryBatch(indices=[0], scores=[1.0], strategy="margin", budget=1)
        assert batch.metadata == {}


class TestLabelingQueue:
    def test_add_and_queue_size(self):
        q = LabelingQueue()
        q.add({"id": 1})
        q.add({"id": 2})
        assert q.queue_size == 2

    def test_add_batch(self):
        q = LabelingQueue()
        added = q.add_batch([{"id": i} for i in range(5)])
        assert added == 5
        assert q.queue_size == 5

    def test_label_moves_to_labeled(self):
        q = LabelingQueue()
        q.add({"id": 1, "prediction": "anomaly"})
        labeled = q.label(0, label="normal", labeler_id="operator1")
        assert labeled["human_label"] == "normal"
        assert labeled["labeler_id"] == "operator1"
        assert q.queue_size == 0
        assert q.labeled_count == 1

    def test_label_out_of_range_raises(self):
        q = LabelingQueue()
        q.add({"id": 1})
        with pytest.raises(IndexError, match="out of range"):
            q.label(5, label="normal")

    def test_label_negative_index_raises(self):
        q = LabelingQueue()
        q.add({"id": 1})
        with pytest.raises(IndexError):
            q.label(-1, label="normal")

    def test_drain_labeled_clears_buffer(self):
        q = LabelingQueue()
        q.add({"id": 1})
        q.add({"id": 2})
        q.label(0, label="anomaly")
        q.label(0, label="normal")
        result = q.drain_labeled()
        assert len(result) == 2
        assert q.labeled_count == 0

    def test_drain_returns_copy(self):
        q = LabelingQueue()
        q.add({"id": 1})
        q.label(0, label="X")
        result1 = q.drain_labeled()
        result2 = q.drain_labeled()
        assert len(result1) == 1
        assert len(result2) == 0  # already drained

    def test_max_size_drops_oldest(self):
        q = LabelingQueue(max_size=3)
        for i in range(4):
            q.add({"id": i})
        # After adding 4 to max_size=3, first item dropped
        assert q.queue_size == 3
        first = q.peek(1)[0]
        assert first["id"] == 1  # id=0 was dropped

    def test_peek_returns_n_items(self):
        q = LabelingQueue()
        for i in range(5):
            q.add({"id": i})
        peeked = q.peek(3)
        assert len(peeked) == 3
        assert peeked[0]["id"] == 0

    def test_peek_more_than_available(self):
        q = LabelingQueue()
        q.add({"id": 1})
        peeked = q.peek(10)
        assert len(peeked) == 1

    def test_peek_does_not_remove(self):
        q = LabelingQueue()
        q.add({"id": 1})
        q.peek(1)
        assert q.queue_size == 1

    def test_labeled_count_accurate(self):
        q = LabelingQueue()
        q.add({"id": 1})
        q.add({"id": 2})
        q.label(0, "normal")
        assert q.labeled_count == 1
        q.label(0, "anomaly")
        assert q.labeled_count == 2


class TestIntegrationActiveLearning:
    def test_uncertainty_then_label_full_cycle(self):
        """End-to-end: predict → select → label → drain."""
        sampler = ActiveLearningSampler(strategy="uncertainty", budget=3)
        q = LabelingQueue()

        preds = _preds(10)
        batch = sampler.select(preds)
        assert batch.size == 3

        # Add selected samples to labeling queue
        for idx in batch.indices:
            q.add({**preds[idx], "pool_idx": idx})
        assert q.queue_size == 3

        # Label all
        while q.queue_size > 0:
            q.label(0, label="normal")

        labeled = q.drain_labeled()
        assert len(labeled) == 3
        assert all("human_label" in s for s in labeled)

    def test_entropy_strategy_informativeness_order(self):
        """Higher entropy samples scored first."""
        sampler = ActiveLearningSampler(strategy="entropy", budget=5)
        # conf=0.5 → max entropy, conf=0.99 → min entropy
        preds = [
            {"label": "anomaly", "confidence": 0.5},   # most uncertain
            {"label": "anomaly", "confidence": 0.6},
            {"label": "anomaly", "confidence": 0.7},
            {"label": "anomaly", "confidence": 0.8},
            {"label": "anomaly", "confidence": 0.99},  # most certain
        ]
        batch = sampler.select(preds)
        # Index 0 (conf=0.5) should appear first
        assert batch.indices[0] == 0
