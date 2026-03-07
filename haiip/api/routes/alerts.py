"""Alert routes — list, acknowledge, and create alerts."""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from haiip.api.deps import DB, CurrentUser, EngineerUser
from haiip.api.models import Alert
from haiip.api.schemas import AcknowledgeAlertRequest, AlertResponse, Page

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/alerts", response_model=Page[AlertResponse])
async def list_alerts(
    current_user: CurrentUser,
    db: DB,
    severity: str | None = None,
    unacknowledged_only: bool = False,
    page: int = 1,
    size: int = 20,
) -> Page[AlertResponse]:
    """List alerts for the current tenant."""
    query = select(Alert).where(Alert.tenant_id == current_user.tenant_id)

    if severity:
        query = query.where(Alert.severity == severity)
    if unacknowledged_only:
        query = query.where(Alert.is_acknowledged.is_(False))

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    result = await db.execute(
        query.order_by(Alert.created_at.desc()).offset((page - 1) * size).limit(size)
    )
    items = list(result.scalars().all())

    return Page(items=items, total=total, page=page, size=size)


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    current_user: CurrentUser,
    db: DB,
) -> Alert:
    """Get a single alert by ID."""
    result = await db.execute(
        select(Alert).where(
            Alert.id == alert_id,
            Alert.tenant_id == current_user.tenant_id,
        )
    )
    alert = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    return alert


@router.patch("/alerts/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: str,
    body: AcknowledgeAlertRequest,
    current_user: CurrentUser,
    db: DB,
) -> Alert:
    """Acknowledge an alert — marks it as reviewed by a human operator."""
    result = await db.execute(
        select(Alert).where(
            Alert.id == alert_id,
            Alert.tenant_id == current_user.tenant_id,
        )
    )
    alert = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if alert.is_acknowledged:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Alert already acknowledged",
        )

    alert.is_acknowledged = True
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.now(UTC)
    await db.flush()
    await db.refresh(alert)

    logger.info(
        "alert.acknowledged",
        alert_id=alert_id,
        by=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    return alert


@router.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    machine_id: str,
    severity: str,
    title: str,
    message: str,
    current_user: EngineerUser,
    db: DB,
) -> Alert:
    """Manually create an alert. Engineers and admins only."""
    valid_severities = {"critical", "high", "medium", "low"}
    if severity not in valid_severities:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"severity must be one of {valid_severities}",
        )

    alert = Alert(
        tenant_id=current_user.tenant_id,
        machine_id=machine_id,
        severity=severity,
        title=title,
        message=message,
    )
    db.add(alert)
    await db.flush()
    await db.refresh(alert)

    logger.info(
        "alert.created",
        alert_id=alert.id,
        severity=severity,
        machine_id=machine_id,
        tenant_id=current_user.tenant_id,
    )
    return alert
