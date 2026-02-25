"""SQLAlchemy ORM models — one file, all tables visible at a glance."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from haiip.api.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── Tenant ────────────────────────────────────────────────────────────────────
class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    users: Mapped[list["User"]] = relationship("User", back_populates="tenant")
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="tenant"
    )
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="tenant")

    __table_args__ = (Index("ix_tenants_slug", "slug"),)


# ── User ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(256), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    full_name: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(
        Enum("admin", "engineer", "operator", "viewer", name="user_role"),
        default="operator",
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    last_login: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="users")

    __table_args__ = (
        Index("ix_users_email_tenant", "email", "tenant_id", unique=True),
    )


# ── Prediction ────────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    machine_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(
        Enum(
            "anomaly_detection",
            "predictive_maintenance",
            "rul_prediction",
            name="model_type",
        ),
        nullable=False,
    )
    prediction_label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rul_cycles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_features: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    human_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_tenant_machine", "tenant_id", "machine_id"),
        Index("ix_predictions_created_at", "created_at"),
    )


# ── Alert ─────────────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    machine_id: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(
        Enum("critical", "high", "medium", "low", name="alert_severity"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_tenant_severity", "tenant_id", "severity"),
        Index("ix_alerts_created_at", "created_at"),
    )


# ── AuditLog (EU AI Act) ──────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_audit_logs_tenant_action", "tenant_id", "action"),
        Index("ix_audit_logs_created_at", "created_at"),
    )


# ── ModelRegistry ─────────────────────────────────────────────────────────────
class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    dataset_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_model_registry_tenant_name", "tenant_id", "model_name"),
    )


# ── FeedbackLog ───────────────────────────────────────────────────────────────
class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False)
    prediction_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    was_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    corrected_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_feedback_prediction", "prediction_id"),
    )
