"""Active learning loop for HAIIP — uncertainty sampling + query strategy.

Reduces labeling cost for SME operators by selecting the most informative
samples for human review. Integrates with FeedbackEngine to close the loop.

Strategies implemented:
- UncertaintySampling: select samples with confidence closest to decision boundary
- MarginSampling: minimize margin between top-2 class probabilities
- EntropySampling: maximize prediction entropy (most uncertain)
- CoreSetSampling: maximise coverage of feature space (greedy k-center)

Usage:
    sampler = ActiveLearningSampler(strategy="uncertainty", budget=10)
    query_indices = sampler.select(predictions, X)
    # Present X[query_indices] to operator for labeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STRATEGIES = ("uncertainty", "margin", "entropy", "coreset", "random")


@dataclass
class QueryBatch:
    """A batch of samples selected for human labeling."""

    indices: list[int]
    scores: list[float]  # informativeness score (higher = more informative)
    strategy: str
    budget: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.indices)


class ActiveLearningSampler:
    """Selects the most informative unlabeled samples for operator review.

    Args:
        strategy: sampling strategy name (see STRATEGIES)
        budget: number of samples to select per query round
        confidence_floor: min confidence to consider for uncertainty sampling
                          (skip very low-confidence — likely noise)
    """

    def __init__(
        self,
        strategy: str = "uncertainty",
        budget: int = 10,
        confidence_floor: float = 0.0,
        random_state: int = 42,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")
        self.strategy = strategy
        self.budget = budget
        self.confidence_floor = confidence_floor
        self._rng = np.random.default_rng(random_state)

    def select(
        self,
        predictions: list[dict[str, Any]],
        X: np.ndarray | None = None,
    ) -> QueryBatch:
        """Select indices of most informative samples.

        Args:
            predictions: list of dicts from AnomalyDetector.predict()
            X: optional feature matrix (required for coreset strategy)

        Returns:
            QueryBatch with selected indices and informativeness scores.
        """
        if not predictions:
            return QueryBatch(indices=[], scores=[], strategy=self.strategy, budget=self.budget)

        n = len(predictions)
        budget = min(self.budget, n)

        if self.strategy == "uncertainty":
            indices, scores = self._uncertainty_sampling(predictions, budget)
        elif self.strategy == "margin":
            indices, scores = self._margin_sampling(predictions, budget)
        elif self.strategy == "entropy":
            indices, scores = self._entropy_sampling(predictions, budget)
        elif self.strategy == "coreset" and X is not None:
            indices, scores = self._coreset_sampling(X, budget)
        elif self.strategy == "random":
            indices, scores = self._random_sampling(n, budget)
        else:
            # Fallback to uncertainty if coreset called without X
            logger.warning("Falling back to uncertainty sampling (X not provided for coreset)")
            indices, scores = self._uncertainty_sampling(predictions, budget)

        return QueryBatch(
            indices=indices,
            scores=scores,
            strategy=self.strategy,
            budget=budget,
            metadata={"n_pool": n, "n_selected": len(indices)},
        )

    # ── Sampling strategies ───────────────────────────────────────────────────

    def _uncertainty_sampling(
        self, predictions: list[dict[str, Any]], budget: int
    ) -> tuple[list[int], list[float]]:
        """Select samples with confidence closest to 0.5 (decision boundary)."""
        scores: list[tuple[int, float]] = []
        for i, pred in enumerate(predictions):
            conf = float(pred.get("confidence", 0.5))
            if conf < self.confidence_floor:
                continue
            # Uncertainty = distance from certainty (1.0)
            uncertainty = 1.0 - conf
            scores.append((i, uncertainty))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:budget]
        return [s[0] for s in selected], [s[1] for s in selected]

    def _margin_sampling(
        self, predictions: list[dict[str, Any]], budget: int
    ) -> tuple[list[int], list[float]]:
        """Minimize margin between anomaly and normal probabilities.

        For binary classifier: margin = |2×conf - 1| → 0 means most uncertain.
        Smallest margin = most informative.
        """
        scores: list[tuple[int, float]] = []
        for i, pred in enumerate(predictions):
            conf = float(pred.get("confidence", 0.5))
            if conf < self.confidence_floor:
                continue
            margin = abs(2 * conf - 1)  # 0 at 0.5, 1 at 0 or 1
            scores.append((i, margin))

        # Sort by smallest margin first (most uncertain)
        scores.sort(key=lambda x: x[1])
        selected = scores[:budget]
        # Return 1-margin as informativeness score (higher = more informative)
        return [s[0] for s in selected], [1.0 - s[1] for s in selected]

    def _entropy_sampling(
        self, predictions: list[dict[str, Any]], budget: int
    ) -> tuple[list[int], list[float]]:
        """Maximise binary entropy H(p) = -p log p - (1-p) log(1-p)."""
        scores: list[tuple[int, float]] = []
        for i, pred in enumerate(predictions):
            conf = float(pred.get("confidence", 0.5))
            if conf < self.confidence_floor:
                continue
            # Treat confidence as P(predicted class)
            # Map to probability of "anomaly" regardless of label
            if pred.get("label") == "anomaly":
                p = conf
            else:
                p = 1.0 - conf
            p = np.clip(p, 1e-10, 1 - 1e-10)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            scores.append((i, float(entropy)))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:budget]
        return [s[0] for s in selected], [s[1] for s in selected]

    def _coreset_sampling(self, X: np.ndarray, budget: int) -> tuple[list[int], list[float]]:
        """Greedy k-center algorithm — maximise coverage of feature space.

        Select points that minimize the maximum distance to the nearest
        already-selected center (maximizes diversity).
        """
        n = len(X)
        if n == 0:
            return [], []

        selected: list[int] = []
        # Start with a random point
        first = int(self._rng.integers(0, n))
        selected.append(first)

        # Distance of each point to nearest selected center
        min_dists = np.full(n, np.inf)
        min_dists = self._update_dists(X, min_dists, first)

        scores: list[float] = []
        for _ in range(budget - 1):
            if len(selected) >= n:
                break
            # Select the point farthest from any center
            next_idx = int(np.argmax(min_dists))
            scores.append(float(min_dists[next_idx]))
            selected.append(next_idx)
            min_dists[next_idx] = 0.0
            min_dists = self._update_dists(X, min_dists, next_idx)

        # Score for first point = max of all remaining distances
        if selected:
            scores.insert(0, float(np.max(min_dists)) if len(min_dists) > 0 else 0.0)

        return selected[:budget], scores[:budget]

    @staticmethod
    def _update_dists(X: np.ndarray, min_dists: np.ndarray, center_idx: int) -> np.ndarray:
        """Update min distances after adding a new center."""
        dists = np.linalg.norm(X - X[center_idx], axis=1)
        return np.minimum(min_dists, dists)

    def _random_sampling(self, n: int, budget: int) -> tuple[list[int], list[float]]:
        """Random baseline sampling."""
        indices = list(self._rng.choice(n, size=budget, replace=False).astype(int))
        scores = [1.0 / budget] * len(indices)
        return indices, scores


class LabelingQueue:
    """Manages a queue of samples awaiting operator labeling.

    Integrates with FeedbackEngine: labeled samples are automatically
    forwarded for model improvement.

    Thread-safe: uses a list protected by logic (not a lock) since
    all operations are from a single operator session.
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._queue: list[dict[str, Any]] = []
        self._labeled: list[dict[str, Any]] = []

    def add(self, sample: dict[str, Any]) -> bool:
        """Add a sample to the labeling queue.

        Returns False if queue is full.
        """
        if len(self._queue) >= self.max_size:
            logger.warning("LabelingQueue full (%d) — dropping oldest sample", self.max_size)
            self._queue.pop(0)
        self._queue.append(sample)
        return True

    def add_batch(self, samples: list[dict[str, Any]]) -> int:
        """Add multiple samples. Returns number added."""
        added = 0
        for s in samples:
            if self.add(s):
                added += 1
        return added

    def label(self, index: int, label: str, labeler_id: str = "operator") -> dict[str, Any]:
        """Label the sample at position `index` in the queue.

        Removes it from queue and moves to labeled buffer.
        Returns the labeled sample.
        """
        if index < 0 or index >= len(self._queue):
            raise IndexError(f"Queue index {index} out of range (size={len(self._queue)})")

        sample = self._queue.pop(index)
        labeled = {**sample, "human_label": label, "labeler_id": labeler_id}
        self._labeled.append(labeled)
        return labeled

    def drain_labeled(self) -> list[dict[str, Any]]:
        """Return and clear all labeled samples (for batch feedback submission)."""
        result = list(self._labeled)
        self._labeled.clear()
        return result

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def labeled_count(self) -> int:
        return len(self._labeled)

    def peek(self, n: int = 5) -> list[dict[str, Any]]:
        """Return first n items from queue without removing."""
        return list(self._queue[:n])
