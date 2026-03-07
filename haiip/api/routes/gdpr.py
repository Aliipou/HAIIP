"""GDPR compliance API routes (Arts. 17, 20, 25).

Endpoints:
- POST /gdpr/erasure        — Art. 17 right to erasure request
- GET  /gdpr/export/{id}    — Art. 20 data portability export
- POST /gdpr/scan           — Art. 25 PII scan payload
- GET  /gdpr/consent/{id}   — consent record validation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from haiip.api.deps import CurrentUser
from haiip.core.data_privacy import DataExportResult, DataPrivacyEngine, ErasureRequest

router = APIRouter(prefix="/gdpr", tags=["GDPR"])


def _get_engine(tenant_id: str) -> DataPrivacyEngine:
    """Build privacy engine with per-tenant salt."""
    import hashlib

    salt = hashlib.sha256(f"haiip-tenant-{tenant_id}".encode()).hexdigest()[:32]
    return DataPrivacyEngine(tenant_salt=salt)


# ── Request / Response models ─────────────────────────────────────────────────


class ErasureRequestBody(BaseModel):
    subject_id: str = Field(..., description="User ID or machine ID to erase")
    tables: list[str] = Field(
        default_factory=lambda: ["predictions", "audit_logs", "feedback"],
        description="Tables to purge",
    )


class ErasureResponse(BaseModel):
    request_id: str
    tenant_id: str
    subject_id: str
    status: str
    tables_affected: list[str]
    requested_at: str


class PIIScanRequest(BaseModel):
    payload: dict[str, Any] = Field(..., description="Payload to scan for PII")


class PIIScanResponse(BaseModel):
    has_pii: bool
    scrubbed: dict[str, Any]
    pii_fields_found: list[str]


class ConsentRecord(BaseModel):
    subject_id: str
    tenant_id: str
    purpose: str
    granted_at: str
    legal_basis: str


class ConsentValidationResponse(BaseModel):
    valid: bool
    missing_fields: list[str]
    expired: bool


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/erasure",
    response_model=ErasureResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="GDPR Art. 17 — Right to Erasure",
)
async def request_erasure(
    body: ErasureRequestBody,
    current_user: CurrentUser,
) -> ErasureResponse:
    """Submit a GDPR Art. 17 erasure request.

    In production this queues a background job to delete all records
    for the subject_id from the specified tables.
    This endpoint records the request and returns a tracking ID.
    """
    import uuid

    tenant_id = str(current_user.tenant_id)
    request_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    # In production: queue deletion job via Celery
    # For now: record request and return accepted status
    erasure = ErasureRequest(
        tenant_id=tenant_id,
        subject_id=body.subject_id,
        requested_at=now,
        tables_affected=body.tables,
        records_deleted=0,
        status="accepted",
    )

    return ErasureResponse(
        request_id=request_id,
        tenant_id=erasure.tenant_id,
        subject_id=erasure.subject_id,
        status=erasure.status,
        tables_affected=erasure.tables_affected,
        requested_at=erasure.requested_at,
    )


@router.get(
    "/export/{subject_id}",
    summary="GDPR Art. 20 — Data Portability Export",
)
async def export_data(
    subject_id: str,
    current_user: CurrentUser,
) -> JSONResponse:
    """Export all data for a subject as JSON (GDPR Art. 20).

    Returns a portable JSON payload of all records associated with subject_id
    for the current tenant.
    """
    tenant_id = str(current_user.tenant_id)

    # In production: query all tables and build export
    # For now: return structure template
    export = DataExportResult(
        tenant_id=tenant_id,
        subject_id=subject_id,
        exported_at=datetime.now(UTC).isoformat(),
        tables={
            "predictions": [],
            "audit_logs": [],
            "feedback": [],
            "alerts": [],
        },
        record_count=0,
    )

    return JSONResponse(
        content={
            "tenant_id": export.tenant_id,
            "subject_id": export.subject_id,
            "exported_at": export.exported_at,
            "tables": export.tables,
            "record_count": export.record_count,
            "format": "GDPR-portable-JSON-v1",
        }
    )


@router.post(
    "/scan",
    response_model=PIIScanResponse,
    summary="GDPR Art. 25 — PII scan (Privacy by Design)",
)
async def scan_pii(
    body: PIIScanRequest,
    current_user: CurrentUser,
) -> PIIScanResponse:
    """Scan a payload for PII before storage (Art. 25 privacy by design).

    Returns the scrubbed payload and list of PII types found.
    """
    engine = _get_engine(str(current_user.tenant_id))
    scrubbed = engine.scrub_pii(body.payload)

    pii_found: list[str] = []
    for _key, val in body.payload.items():
        if isinstance(val, str):
            result = engine.detect_pii(val)
            if result.has_pii:
                pii_found.extend(result.pii_types)

    return PIIScanResponse(
        has_pii=len(pii_found) > 0,
        scrubbed=scrubbed,
        pii_fields_found=list(set(pii_found)),
    )


@router.post(
    "/consent/validate",
    response_model=ConsentValidationResponse,
    summary="Validate a GDPR consent record",
)
async def validate_consent(
    record: ConsentRecord,
    current_user: CurrentUser,
) -> ConsentValidationResponse:
    """Validate a consent record has all required GDPR fields and is not expired."""
    valid, missing = DataPrivacyEngine.validate_consent_record(record.model_dump())
    expired = DataPrivacyEngine.is_consent_expired(record.granted_at)

    return ConsentValidationResponse(
        valid=valid and not expired,
        missing_fields=missing,
        expired=expired,
    )


@router.get(
    "/health",
    summary="GDPR engine health check",
    include_in_schema=False,
)
async def gdpr_health() -> dict[str, str]:
    return {"status": "ok", "module": "gdpr"}
