"""Active Learning API routes.

Endpoints:
- POST /active-learning/select    — select most informative samples for labeling
- POST /active-learning/label     — submit operator label for a sample
- GET  /active-learning/queue     — view current labeling queue stats
- POST /active-learning/drain     — drain labeled samples (for feedback submission)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from haiip.api.deps import CurrentUser
from haiip.core.active_learning import STRATEGIES, ActiveLearningSampler, LabelingQueue

router = APIRouter(prefix="/active-learning", tags=["Active Learning"])

# In-memory queue per tenant (production: use Redis or DB)
_queues: dict[str, LabelingQueue] = {}


def _get_queue(tenant_id: str) -> LabelingQueue:
    if tenant_id not in _queues:
        _queues[tenant_id] = LabelingQueue(max_size=500)
    return _queues[tenant_id]


# ── Models ────────────────────────────────────────────────────────────────────


class PredictionInput(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    machine_id: str = ""
    prediction_id: str = ""


class SelectRequest(BaseModel):
    predictions: list[PredictionInput]
    strategy: str = Field("uncertainty", description=f"One of: {STRATEGIES}")
    budget: int = Field(10, ge=1, le=100)
    confidence_floor: float = Field(0.0, ge=0.0, le=1.0)


class SelectResponse(BaseModel):
    selected_indices: list[int]
    scores: list[float]
    strategy: str
    budget: int
    n_pool: int


class LabelRequest(BaseModel):
    queue_index: int = Field(..., ge=0)
    human_label: str
    labeler_id: str = "operator"


class LabelResponse(BaseModel):
    labeled_sample: dict[str, Any]
    queue_size_remaining: int


class QueueStats(BaseModel):
    queue_size: int
    labeled_count: int
    max_size: int


class DrainResponse(BaseModel):
    labeled_samples: list[dict[str, Any]]
    count: int


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/select",
    response_model=SelectResponse,
    summary="Select most informative samples for labeling",
)
async def select_samples(
    body: SelectRequest,
    current_user: CurrentUser,
) -> SelectResponse:
    """Run active learning query strategy to select samples for human review.

    Returns indices of the most informative predictions along with informativeness scores.
    """
    if body.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown strategy '{body.strategy}'. Valid: {STRATEGIES}",
        )

    sampler = ActiveLearningSampler(
        strategy=body.strategy,
        budget=body.budget,
        confidence_floor=body.confidence_floor,
    )

    preds = [p.model_dump() for p in body.predictions]
    batch = sampler.select(preds)

    return SelectResponse(
        selected_indices=batch.indices,
        scores=batch.scores,
        strategy=batch.strategy,
        budget=batch.budget,
        n_pool=len(preds),
    )


@router.post(
    "/queue",
    status_code=status.HTTP_201_CREATED,
    summary="Add selected samples to labeling queue",
)
async def add_to_queue(
    samples: list[dict[str, Any]],
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Add samples to the tenant's labeling queue."""
    q = _get_queue(str(current_user.tenant_id))
    added = q.add_batch(samples)
    return {"added": added, "queue_size": q.queue_size}


@router.post(
    "/label",
    response_model=LabelResponse,
    summary="Submit operator label for a queued sample",
)
async def label_sample(
    body: LabelRequest,
    current_user: CurrentUser,
) -> LabelResponse:
    """Label a sample at the given queue index."""
    q = _get_queue(str(current_user.tenant_id))
    if body.queue_index >= q.queue_size:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Queue index {body.queue_index} out of range (size={q.queue_size})",
        )
    labeled = q.label(body.queue_index, body.human_label, body.labeler_id)
    return LabelResponse(labeled_sample=labeled, queue_size_remaining=q.queue_size)


@router.get(
    "/queue/stats",
    response_model=QueueStats,
    summary="Get labeling queue statistics",
)
async def queue_stats(
    current_user: CurrentUser,
) -> QueueStats:
    """Return current queue size and labeled sample count."""
    q = _get_queue(str(current_user.tenant_id))
    return QueueStats(
        queue_size=q.queue_size,
        labeled_count=q.labeled_count,
        max_size=q.max_size,
    )


@router.post(
    "/drain",
    response_model=DrainResponse,
    summary="Drain labeled samples for feedback submission",
)
async def drain_labeled(
    current_user: CurrentUser,
) -> DrainResponse:
    """Return and clear all labeled samples. Submit to FeedbackEngine for retraining."""
    q = _get_queue(str(current_user.tenant_id))
    samples = q.drain_labeled()
    return DrainResponse(labeled_samples=samples, count=len(samples))
