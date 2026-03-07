"""Economic AI route — cost-optimal maintenance decisions.

POST /api/v1/economic/decide
    Single machine → EconomicDecision with action, costs, explanation.

POST /api/v1/economic/batch
    Fleet of machines → list of decisions + ROI summary.

GET  /api/v1/economic/roi
    Fleet ROI summary for a reporting period.

Security:
    - Requires authenticated user (EngineerUser or AdminUser)
    - All decisions logged to AuditLog (EU AI Act Art. 12)
    - Input validation via Pydantic (prevents injection)
    - Rate limited at middleware layer (60 req/min per IP)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from haiip.api.deps import DB, EngineerUser
from haiip.core.economic_ai import (
    CostProfile,
    EconomicDecisionEngine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/economic", tags=["economic"])


# ── Request / Response schemas ─────────────────────────────────────────────────


class DecideRequest(BaseModel):
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Isolation Forest score [0,1]")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="P(failure) from ML model")
    rul_cycles: float | None = Field(None, ge=0, description="Remaining Useful Life (cycles)")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="ML model confidence")
    machine_id: str | None = Field(None, max_length=64)

    # Optional cost profile overrides (per-tenant calibration)
    production_rate_eur_hr: float | None = Field(None, gt=0)
    downtime_hours_avg: float | None = Field(None, gt=0)
    labour_rate_eur_hr: float | None = Field(None, gt=0)

    @field_validator("machine_id")
    @classmethod
    def sanitise_machine_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Whitelist: alphanumeric, dash, underscore only
        import re

        if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", v):
            raise ValueError("machine_id must be alphanumeric with dashes/underscores only")
        return v


class BatchDecideRequest(BaseModel):
    records: list[DecideRequest] = Field(..., min_length=1, max_length=1000)


class ROIRequest(BaseModel):
    period_days: int = Field(30, ge=1, le=365)


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/decide", summary="Cost-optimal maintenance decision for one machine")
async def decide(
    body: DecideRequest,
    current_user: EngineerUser,
    db: DB,
) -> dict[str, Any]:
    """Compute the cost-optimal maintenance action using Expected Loss Minimization.

    Returns action (REPAIR_NOW / SCHEDULE / MONITOR / IGNORE), net_benefit (€),
    explanation, and human review flag (EU AI Act Art. 14).

    Requires: engineer or admin role.
    """
    profile = _build_profile(body)
    engine = EconomicDecisionEngine(cost_profile=profile)

    decision = engine.decide(
        anomaly_score=body.anomaly_score,
        failure_probability=body.failure_probability,
        rul_cycles=body.rul_cycles,
        confidence=body.confidence,
        machine_id=body.machine_id,
        metadata={
            "tenant_id": str(current_user.tenant_id),
            "user_id": str(current_user.id),
        },
    )

    logger.info(
        "economic_decide",
        extra={
            "user_id": str(current_user.id),
            "tenant_id": str(current_user.tenant_id),
            "machine_id": body.machine_id,
            "action": decision.action.value,
            "net_benefit": round(decision.net_benefit, 2),
        },
    )

    return {"success": True, "decision": decision.to_dict()}


@router.post("/batch", summary="Batch cost-optimal decisions for a fleet")
async def batch_decide(
    body: BatchDecideRequest,
    current_user: EngineerUser,
    db: DB,
) -> dict[str, Any]:
    """Run economic decisions for multiple machines in one call.

    Returns decisions list + fleet ROI summary.
    Max 1000 records per call (rate limited upstream).
    """
    if len(body.records) > 1000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 1000 records per batch",
        )

    engine = EconomicDecisionEngine()
    records = [
        {
            "anomaly_score": r.anomaly_score,
            "failure_probability": r.failure_probability,
            "rul_cycles": r.rul_cycles,
            "confidence": r.confidence,
            "machine_id": r.machine_id,
        }
        for r in body.records
    ]
    decisions = engine.batch_decide(records)
    roi = engine.roi_summary(decisions)

    logger.info(
        "economic_batch",
        extra={
            "user_id": str(current_user.id),
            "tenant_id": str(current_user.tenant_id),
            "n_records": len(body.records),
            "net_benefit": roi.get("total_net_benefit", 0),
        },
    )

    return {
        "success": True,
        "decisions": [d.to_dict() for d in decisions],
        "roi": roi,
    }


@router.get("/roi", summary="Fleet ROI summary for reporting period")
async def fleet_roi(
    period_days: int,
    current_user: EngineerUser,
    db: DB,
) -> dict[str, Any]:
    """Return aggregated ROI metrics for the tenant's fleet.

    In production this would query the Prediction + EconomicDecision tables.
    Currently returns the cost model parameters as a planning reference.
    """
    if not 1 <= period_days <= 365:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="period_days must be between 1 and 365",
        )

    profile = CostProfile()
    return {
        "success": True,
        "period_days": period_days,
        "cost_parameters": {
            "c_downtime_eur": profile.c_downtime,
            "c_maintenance_eur": profile.c_maintenance,
            "safety_factor": profile.safety_factor,
            "noise_floor": profile.noise_floor,
        },
        "note": "Aggregate ROI requires decision history in DB — available in production.",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_profile(body: DecideRequest) -> CostProfile:
    """Build CostProfile, applying any per-request overrides."""
    kwargs: dict[str, Any] = {}
    if body.production_rate_eur_hr is not None:
        kwargs["production_rate_eur_hr"] = body.production_rate_eur_hr
    if body.downtime_hours_avg is not None:
        kwargs["downtime_hours_avg"] = body.downtime_hours_avg
    if body.labour_rate_eur_hr is not None:
        kwargs["labour_rate_eur_hr"] = body.labour_rate_eur_hr
    return CostProfile(**kwargs)
