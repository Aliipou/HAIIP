"""Admin routes — tenant management, user administration, model registry, audit log.

All endpoints require admin role (enforced via AdminUser dependency).
Tenant isolation: admins can only act within their own tenant.
Platform-level super-admin (is_superadmin flag) not yet implemented — planned for v0.2.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from haiip.api.auth import hash_password
from haiip.api.deps import AdminUser, CurrentUser, DB
from haiip.api.models import AuditLog, ModelRegistry, Tenant, User

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Schemas (admin-only, not exposed in public schemas.py) ────────────────────

class UserListItem(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class UserCreateAdmin(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=256)
    role: str = Field(default="operator", pattern="^(admin|engineer|operator|viewer)$")
    password: str = Field(..., min_length=8)


class UserUpdateAdmin(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=256)
    role: str | None = Field(default=None, pattern="^(admin|engineer|operator|viewer)$")
    is_active: bool | None = None


class TenantInfo(BaseModel):
    id: str
    name: str
    slug: str
    is_active: bool
    created_at: str
    user_count: int
    prediction_count: int

    class Config:
        from_attributes = True


class ModelRegistryItem(BaseModel):
    id: str
    model_name: str
    model_version: str
    artifact_path: str
    metrics: str | None
    is_active: bool
    trained_at: str
    dataset_hash: str | None

    class Config:
        from_attributes = True


class ModelActivate(BaseModel):
    model_id: str


# ── Tenant info (admin sees own tenant) ───────────────────────────────────────

@router.get("/admin/tenant", response_model=TenantInfo)
async def get_own_tenant(
    current_user: AdminUser,
    db: DB,
) -> TenantInfo:
    """Get details about the current admin's tenant."""
    result = await db.execute(select(Tenant).where(Tenant.id == current_user.tenant_id))
    tenant = result.scalar_one_or_none()
    if tenant is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tenant not found")

    user_count_r = await db.execute(
        select(func.count(User.id)).where(User.tenant_id == tenant.id)
    )
    user_count = user_count_r.scalar_one()

    from haiip.api.models import Prediction
    pred_count_r = await db.execute(
        select(func.count(Prediction.id)).where(Prediction.tenant_id == tenant.id)
    )
    pred_count = pred_count_r.scalar_one()

    return TenantInfo(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        created_at=tenant.created_at.isoformat(),
        user_count=user_count,
        prediction_count=pred_count,
    )


# ── User management ───────────────────────────────────────────────────────────

@router.get("/admin/users", response_model=list[UserListItem])
async def list_users(
    current_user: AdminUser,
    db: DB,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    role: str | None = Query(None, pattern="^(admin|engineer|operator|viewer)$"),
    is_active: bool | None = Query(None),
) -> list[UserListItem]:
    """List all users in the admin's tenant."""
    q = select(User).where(User.tenant_id == current_user.tenant_id)
    if role:
        q = q.where(User.role == role)
    if is_active is not None:
        q = q.where(User.is_active == is_active)
    q = q.offset(skip).limit(limit)
    result = await db.execute(q)
    users = result.scalars().all()
    return [
        UserListItem(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at.isoformat(),
        )
        for u in users
    ]


@router.post("/admin/users", response_model=UserListItem, status_code=status.HTTP_201_CREATED)
async def create_user_admin(
    body: UserCreateAdmin,
    current_user: AdminUser,
    db: DB,
) -> UserListItem:
    """Create a new user in the admin's tenant."""
    # Check email uniqueness within tenant
    existing = await db.execute(
        select(User).where(
            User.email == body.email,
            User.tenant_id == current_user.tenant_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered in this tenant")

    user = User(
        tenant_id=current_user.tenant_id,
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    logger.info("admin.user_created", admin=current_user.id, new_user=user.id, role=body.role)

    return UserListItem(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )


@router.patch("/admin/users/{user_id}", response_model=UserListItem)
async def update_user_admin(
    user_id: str,
    body: UserUpdateAdmin,
    current_user: AdminUser,
    db: DB,
) -> UserListItem:
    """Update a user's role, name, or active status (within same tenant)."""
    result = await db.execute(
        select(User).where(User.id == user_id, User.tenant_id == current_user.tenant_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    # Prevent admin from deactivating themselves
    if user.id == current_user.id and body.is_active is False:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot deactivate your own account")

    if body.full_name is not None:
        user.full_name = body.full_name
    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active

    await db.flush()
    await db.refresh(user)
    logger.info("admin.user_updated", admin=current_user.id, target=user_id)

    return UserListItem(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )


@router.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None)
async def deactivate_user(
    user_id: str,
    current_user: AdminUser,
    db: DB,
):
    """Soft-delete (deactivate) a user. Does not delete DB record — preserves audit trail."""
    if user_id == current_user.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot deactivate your own account")

    result = await db.execute(
        select(User).where(User.id == user_id, User.tenant_id == current_user.tenant_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    user.is_active = False
    await db.flush()
    logger.info("admin.user_deactivated", admin=current_user.id, target=user_id)


# ── Audit log ─────────────────────────────────────────────────────────────────

class AuditLogItem(BaseModel):
    id: str
    user_id: str | None
    action: str
    resource_type: str
    resource_id: str | None
    details: str | None
    ip_address: str | None
    created_at: str


@router.get("/audit", response_model=list[AuditLogItem])
async def get_audit_log(
    current_user: AdminUser,
    db: DB,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    action: str | None = Query(None),
) -> list[AuditLogItem]:
    """Retrieve the EU AI Act audit log for the current tenant."""
    q = select(AuditLog).where(AuditLog.tenant_id == current_user.tenant_id)
    if action:
        q = q.where(AuditLog.action == action)
    q = q.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(q)
    logs = result.scalars().all()
    return [
        AuditLogItem(
            id=log.id,
            user_id=log.user_id,
            action=log.action,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            details=log.details,
            ip_address=log.ip_address,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]


# ── Model Registry ────────────────────────────────────────────────────────────

@router.get("/admin/models", response_model=list[ModelRegistryItem])
async def list_models(
    current_user: AdminUser,
    db: DB,
) -> list[ModelRegistryItem]:
    """List all models in the registry for this tenant."""
    result = await db.execute(
        select(ModelRegistry)
        .where(ModelRegistry.tenant_id == current_user.tenant_id)
        .order_by(ModelRegistry.trained_at.desc())
    )
    models = result.scalars().all()
    return [
        ModelRegistryItem(
            id=m.id,
            model_name=m.model_name,
            model_version=m.model_version,
            artifact_path=m.artifact_path,
            metrics=m.metrics,
            is_active=m.is_active,
            trained_at=m.trained_at.isoformat(),
            dataset_hash=m.dataset_hash,
        )
        for m in models
    ]


@router.post("/admin/models/{model_id}/activate", response_model=ModelRegistryItem)
async def activate_model(
    model_id: str,
    current_user: AdminUser,
    db: DB,
) -> ModelRegistryItem:
    """Activate a model version (deactivates all others of same name)."""
    result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.id == model_id,
            ModelRegistry.tenant_id == current_user.tenant_id,
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")

    # Deactivate all other versions of this model
    await db.execute(
        update(ModelRegistry)
        .where(
            ModelRegistry.tenant_id == current_user.tenant_id,
            ModelRegistry.model_name == model.model_name,
            ModelRegistry.id != model_id,
        )
        .values(is_active=False)
    )
    model.is_active = True
    await db.flush()
    await db.refresh(model)

    logger.info(
        "admin.model_activated",
        admin=current_user.id,
        model_id=model_id,
        model_name=model.model_name,
        version=model.model_version,
    )

    return ModelRegistryItem(
        id=model.id,
        model_name=model.model_name,
        model_version=model.model_version,
        artifact_path=model.artifact_path,
        metrics=model.metrics,
        is_active=model.is_active,
        trained_at=model.trained_at.isoformat(),
        dataset_hash=model.dataset_hash,
    )


# ── System stats (admin dashboard) ───────────────────────────────────────────

class SystemStats(BaseModel):
    tenant_id: str
    total_users: int
    active_users: int
    total_predictions: int
    total_alerts: int
    unacknowledged_alerts: int
    total_feedback: int
    active_models: int
    audit_events_today: int


@router.get("/admin/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: AdminUser,
    db: DB,
) -> SystemStats:
    """Aggregate stats for the admin dashboard."""
    from datetime import datetime, timezone

    from haiip.api.models import Alert, FeedbackLog, Prediction

    tid = current_user.tenant_id
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    async def count(model, *filters):  # type: ignore[no-untyped-def]
        r = await db.execute(select(func.count(model.id)).where(*filters))
        return r.scalar_one()

    total_users = await count(User, User.tenant_id == tid)
    active_users = await count(User, User.tenant_id == tid, User.is_active == True)  # noqa: E712
    total_predictions = await count(Prediction, Prediction.tenant_id == tid)
    total_alerts = await count(Alert, Alert.tenant_id == tid)
    unacked_alerts = await count(Alert, Alert.tenant_id == tid, Alert.is_acknowledged == False)  # noqa: E712
    total_feedback = await count(FeedbackLog, FeedbackLog.tenant_id == tid)
    active_models = await count(ModelRegistry, ModelRegistry.tenant_id == tid, ModelRegistry.is_active == True)  # noqa: E712
    audit_today = await count(
        AuditLog,
        AuditLog.tenant_id == tid,
        AuditLog.created_at >= today,
    )

    return SystemStats(
        tenant_id=tid,
        total_users=total_users,
        active_users=active_users,
        total_predictions=total_predictions,
        total_alerts=total_alerts,
        unacknowledged_alerts=unacked_alerts,
        total_feedback=total_feedback,
        active_models=active_models,
        audit_events_today=audit_today,
    )
