"""Feedback routes — human-in-the-loop correction of predictions."""

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from haiip.api.deps import DB, CurrentUser
from haiip.api.models import FeedbackLog, Prediction
from haiip.api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    body: FeedbackRequest,
    current_user: CurrentUser,
    db: DB,
) -> FeedbackLog:
    """Submit human feedback on a prediction result.

    Used for:
    - Confidence adjustment (core/feedback.py)
    - Automated retraining triggers (workers/tasks.py)
    - EU AI Act human oversight audit trail
    """
    # Verify prediction belongs to this tenant
    pred_result = await db.execute(
        select(Prediction).where(
            Prediction.id == body.prediction_id,
            Prediction.tenant_id == current_user.tenant_id,
        )
    )
    prediction = pred_result.scalar_one_or_none()
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    # Mark prediction as human-verified
    prediction.human_verified = True
    await db.flush()

    # Record feedback
    feedback = FeedbackLog(
        tenant_id=current_user.tenant_id,
        prediction_id=body.prediction_id,
        user_id=current_user.id,
        was_correct=body.was_correct,
        corrected_label=body.corrected_label,
        notes=body.notes,
    )
    db.add(feedback)
    await db.flush()
    await db.refresh(feedback)

    logger.info(
        "feedback.submitted",
        prediction_id=body.prediction_id,
        was_correct=body.was_correct,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )

    return feedback


@router.get("/feedback", response_model=list[FeedbackResponse])
async def list_feedback(
    current_user: CurrentUser,
    db: DB,
    prediction_id: str | None = None,
    limit: int = 50,
) -> list[FeedbackLog]:
    """List feedback for the current tenant."""
    query = select(FeedbackLog).where(FeedbackLog.tenant_id == current_user.tenant_id)
    if prediction_id:
        query = query.where(FeedbackLog.prediction_id == prediction_id)

    result = await db.execute(query.order_by(FeedbackLog.created_at.desc()).limit(min(limit, 200)))
    return list(result.scalars().all())
