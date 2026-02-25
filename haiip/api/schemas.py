"""Pydantic v2 request/response schemas — single source of truth for API contracts.

Rules:
- Inputs: strict validation, reject extra fields
- Outputs: use model_config with from_attributes=True for ORM compat
- Never expose hashed_password in any response schema
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

T = TypeVar("T")


# ── Base schemas ──────────────────────────────────────────────────────────────

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── Generic pagination / response wrappers ────────────────────────────────────

class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int = Field(ge=1)
    size: int = Field(ge=1, le=100)

    @property
    def pages(self) -> int:
        return max(1, -(-self.total // self.size))  # ceiling division


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T
    message: str = "OK"


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None


# ── Auth schemas ──────────────────────────────────────────────────────────────

class LoginRequest(StrictModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    tenant_slug: str = Field(min_length=1, max_length=64)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(StrictModel):
    refresh_token: str


# ── User schemas ──────────────────────────────────────────────────────────────

class UserCreate(StrictModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=256)
    role: Literal["admin", "engineer", "operator", "viewer"] = "operator"

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(ORMModel):
    id: str
    tenant_id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: datetime | None = None


# ── Prediction schemas ────────────────────────────────────────────────────────

class SensorReading(StrictModel):
    """Single sensor reading sent to the predict endpoint."""
    machine_id: str = Field(min_length=1, max_length=128)
    air_temperature: float = Field(ge=-50.0, le=200.0)       # Celsius
    process_temperature: float = Field(ge=-50.0, le=500.0)   # Celsius
    rotational_speed: float = Field(ge=0.0, le=50000.0)      # RPM
    torque: float = Field(ge=0.0, le=1000.0)                  # Nm
    tool_wear: float = Field(ge=0.0, le=500.0)                # minutes
    extra_features: dict[str, float] | None = None


class PredictionResponse(ORMModel):
    id: str
    machine_id: str
    model_type: str
    prediction_label: str
    confidence: float
    anomaly_score: float | None = None
    rul_cycles: int | None = None
    explanation: Any | None = None
    human_verified: bool
    created_at: datetime


class BatchPredictRequest(StrictModel):
    readings: list[SensorReading] = Field(min_length=1, max_length=100)
    model_type: Literal["anomaly_detection", "predictive_maintenance", "rul_prediction"] = (
        "anomaly_detection"
    )


# ── Alert schemas ─────────────────────────────────────────────────────────────

class AlertResponse(ORMModel):
    id: str
    machine_id: str
    severity: str
    title: str
    message: str
    is_acknowledged: bool
    acknowledged_at: datetime | None = None
    created_at: datetime


class AcknowledgeAlertRequest(StrictModel):
    notes: str | None = Field(default=None, max_length=1024)


# ── Metrics schemas ───────────────────────────────────────────────────────────

class MachineMetrics(BaseModel):
    machine_id: str
    total_predictions: int
    anomaly_count: int
    anomaly_rate: float
    avg_confidence: float
    last_prediction_at: datetime | None = None


class SystemHealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    database: bool
    redis: bool
    model_loaded: bool
    uptime_seconds: float
    version: str = "0.1.0"


# ── Feedback schemas ──────────────────────────────────────────────────────────

class FeedbackRequest(StrictModel):
    prediction_id: str
    was_correct: bool
    corrected_label: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=2048)


class FeedbackResponse(ORMModel):
    id: str
    prediction_id: str
    was_correct: bool
    corrected_label: str | None = None
    created_at: datetime


# ── RAG / Query schemas ────────────────────────────────────────────────────────

class QueryRequest(StrictModel):
    question: str = Field(min_length=3, max_length=2000)
    machine_id: str | None = None
    context_window: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    machine_context: dict[str, Any] | None = None


# ── Tenant schemas ────────────────────────────────────────────────────────────

class TenantCreate(StrictModel):
    name: str = Field(min_length=2, max_length=128)
    slug: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")


class TenantResponse(ORMModel):
    id: str
    name: str
    slug: str
    is_active: bool
    created_at: datetime
