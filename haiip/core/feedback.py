"""Human feedback loop — confidence adjustment and retraining triggers.

Architecture:
- Feedback signals from operators adjust model confidence thresholds
- Accumulated feedback triggers automated retraining (via Celery workers)
- All feedback is logged for EU AI Act human oversight compliance

Confidence adjustment formula:
    adjusted_confidence = base_confidence * (1 + feedback_factor)
    feedback_factor derived from recent accuracy on human-verified predictions
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Retraining is triggered when accuracy drops below this threshold
RETRAIN_ACCURACY_THRESHOLD = 0.80
# Minimum feedback samples before triggering retraining
MIN_FEEDBACK_FOR_RETRAIN = 50


@dataclass
class FeedbackRecord:
    prediction_id: str
    was_correct: bool
    corrected_label: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    machine_id: str | None = None


class FeedbackEngine:
    """Accumulates operator feedback and adjusts model confidence scores.

    Thread-safe for reads; not thread-safe for concurrent writes.
    Use Celery tasks for write operations in production.
    """

    def __init__(
        self,
        window_size: int = 200,
        retrain_threshold: float = RETRAIN_ACCURACY_THRESHOLD,
        min_samples: int = MIN_FEEDBACK_FOR_RETRAIN,
    ) -> None:
        self.window_size = window_size
        self.retrain_threshold = retrain_threshold
        self.min_samples = min_samples
        self._records: deque[FeedbackRecord] = deque(maxlen=window_size)
        self._cumulative_correct = 0
        self._cumulative_total = 0

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def record(
        self,
        prediction_id: str,
        was_correct: bool,
        corrected_label: str | None = None,
        machine_id: str | None = None,
    ) -> "FeedbackEngineState":
        """Record a single feedback signal from a human operator.

        Returns current engine state including whether retraining is needed.
        """
        record = FeedbackRecord(
            prediction_id=prediction_id,
            was_correct=was_correct,
            corrected_label=corrected_label,
            machine_id=machine_id,
        )
        self._records.append(record)
        self._cumulative_total += 1
        if was_correct:
            self._cumulative_correct += 1

        state = self._compute_state()
        if state.needs_retraining:
            logger.warning(
                "Retraining triggered: accuracy=%.3f threshold=%.3f samples=%d",
                state.window_accuracy,
                self.retrain_threshold,
                state.window_size,
            )
        return state

    def record_batch(self, records: list[dict[str, Any]]) -> "FeedbackEngineState":
        """Record multiple feedback signals at once."""
        for rec in records:
            self._records.append(
                FeedbackRecord(
                    prediction_id=rec["prediction_id"],
                    was_correct=rec["was_correct"],
                    corrected_label=rec.get("corrected_label"),
                    machine_id=rec.get("machine_id"),
                )
            )
            self._cumulative_total += 1
            if rec["was_correct"]:
                self._cumulative_correct += 1

        return self._compute_state()

    # ── Confidence adjustment ─────────────────────────────────────────────────

    def adjust_confidence(self, base_confidence: float) -> float:
        """Adjust a model's raw confidence using recent feedback accuracy.

        If recent accuracy is high, confidence is scaled up slightly.
        If accuracy is low (drift), confidence is scaled down — more conservative.

        Returns adjusted confidence clamped to [0.0, 1.0].
        """
        state = self._compute_state()
        if state.window_size < 10:
            return base_confidence  # not enough feedback yet

        # Linear scaling: accuracy=1.0 → factor=+10%, accuracy=0.5 → factor=-10%
        factor = (state.window_accuracy - 0.75) * 0.4
        adjusted = base_confidence * (1.0 + factor)
        return float(max(0.0, min(1.0, adjusted)))

    # ── State ─────────────────────────────────────────────────────────────────

    def _compute_state(self) -> "FeedbackEngineState":
        window = list(self._records)
        window_size = len(window)

        if window_size == 0:
            return FeedbackEngineState(
                window_size=0,
                window_accuracy=1.0,
                cumulative_accuracy=1.0,
                cumulative_total=self._cumulative_total,
                needs_retraining=False,
                error_distribution={},
            )

        window_correct = sum(1 for r in window if r.was_correct)
        window_accuracy = window_correct / window_size

        cumulative_accuracy = (
            self._cumulative_correct / self._cumulative_total
            if self._cumulative_total > 0
            else 1.0
        )

        # Count corrected labels (error distribution)
        error_dist: dict[str, int] = {}
        for r in window:
            if not r.was_correct and r.corrected_label:
                error_dist[r.corrected_label] = error_dist.get(r.corrected_label, 0) + 1

        needs_retraining = (
            window_size >= self.min_samples
            and window_accuracy < self.retrain_threshold
        )

        return FeedbackEngineState(
            window_size=window_size,
            window_accuracy=round(window_accuracy, 4),
            cumulative_accuracy=round(cumulative_accuracy, 4),
            cumulative_total=self._cumulative_total,
            needs_retraining=needs_retraining,
            error_distribution=error_dist,
        )

    def get_state(self) -> "FeedbackEngineState":
        return self._compute_state()

    def reset_window(self) -> None:
        """Clear the sliding window — call after retraining is complete."""
        self._records.clear()
        logger.info("FeedbackEngine window reset after retraining")


@dataclass
class FeedbackEngineState:
    window_size: int
    window_accuracy: float
    cumulative_accuracy: float
    cumulative_total: int
    needs_retraining: bool
    error_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "window_accuracy": self.window_accuracy,
            "cumulative_accuracy": self.cumulative_accuracy,
            "cumulative_total": self.cumulative_total,
            "needs_retraining": self.needs_retraining,
            "error_distribution": self.error_distribution,
        }
