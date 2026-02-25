"""Metrics routes — machine KPIs, system health, anomaly rates."""

import time

import structlog
from fastapi import APIRouter
from sqlalchemy import func, select

from haiip.api.database import check_database_connection
from haiip.api.deps import CurrentUser, DB
from haiip.api.models import Alert, Prediction
from haiip.api.schemas import MachineMetrics, SystemHealthResponse

router = APIRouter()
logger = structlog.get_logger(__name__)

_app_start_time = time.monotonic()


@router.get("/metrics/health", response_model=SystemHealthResponse)
async def system_health(current_user: CurrentUser) -> SystemHealthResponse:
    """Return system health status for the dashboard."""
    db_ok = await check_database_connection()

    return SystemHealthResponse(
        status="healthy" if db_ok else "degraded",
        database=db_ok,
        redis=False,  # Phase 6: connect to real Redis health check
        model_loaded=True,
        uptime_seconds=round(time.monotonic() - _app_start_time, 2),
    )


@router.get("/metrics/machines", response_model=list[MachineMetrics])
async def machine_metrics(
    current_user: CurrentUser,
    db: DB,
) -> list[MachineMetrics]:
    """Return per-machine KPI summary for the current tenant."""
    result = await db.execute(
        select(
            Prediction.machine_id,
            func.count(Prediction.id).label("total"),
            func.sum(
                func.cast(Prediction.prediction_label == "anomaly", func.Integer)
            ).label("anomalies"),
            func.avg(Prediction.confidence).label("avg_confidence"),
            func.max(Prediction.created_at).label("last_prediction_at"),
        )
        .where(Prediction.tenant_id == current_user.tenant_id)
        .group_by(Prediction.machine_id)
        .order_by(Prediction.machine_id)
    )

    rows = result.all()
    metrics: list[MachineMetrics] = []
    for row in rows:
        total = row.total or 0
        anomalies = row.anomalies or 0
        metrics.append(
            MachineMetrics(
                machine_id=row.machine_id,
                total_predictions=total,
                anomaly_count=anomalies,
                anomaly_rate=round(anomalies / total, 4) if total > 0 else 0.0,
                avg_confidence=round(float(row.avg_confidence or 0.0), 4),
                last_prediction_at=row.last_prediction_at,
            )
        )
    return metrics


@router.get("/metrics/alerts/summary")
async def alert_summary(
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Return alert count by severity for the current tenant."""
    result = await db.execute(
        select(Alert.severity, func.count(Alert.id).label("count"))
        .where(Alert.tenant_id == current_user.tenant_id)
        .group_by(Alert.severity)
    )
    summary = {row.severity: row.count for row in result.all()}
    return {
        "critical": summary.get("critical", 0),
        "high": summary.get("high", 0),
        "medium": summary.get("medium", 0),
        "low": summary.get("low", 0),
        "total": sum(summary.values()),
    }
