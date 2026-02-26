"""Tests for admin API routes.

Covers: tenant info, user CRUD, audit log, model registry, system stats.
Enforces RBAC: operator cannot access admin endpoints.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from haiip.api.auth import create_access_token, hash_password
from haiip.api.models import AuditLog, ModelRegistry, Tenant, User


# ── Extra fixtures ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_engineer(db_session: AsyncSession, test_tenant: Tenant) -> User:
    user = User(
        tenant_id=test_tenant.id,
        email="engineer@test-sme.com",
        hashed_password=hash_password("Engineer123!"),
        full_name="Test Engineer",
        role="engineer",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
def engineer_headers(test_engineer: User, test_tenant: Tenant) -> dict:
    token = create_access_token(test_engineer.id, test_tenant.id, test_engineer.role)
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def model_entry(db_session: AsyncSession, test_tenant: Tenant) -> ModelRegistry:
    model = ModelRegistry(
        tenant_id=test_tenant.id,
        model_name="anomaly_detector",
        model_version="1.0.0",
        artifact_path="/models/anomaly_v1.joblib",
        metrics='{"f1": 0.91, "precision": 0.89}',
        is_active=False,
        dataset_hash="abc123",
    )
    db_session.add(model)
    await db_session.flush()
    await db_session.refresh(model)
    return model


@pytest_asyncio.fixture
async def audit_entry(db_session: AsyncSession, test_tenant: Tenant, test_admin: User) -> AuditLog:
    log = AuditLog(
        tenant_id=test_tenant.id,
        user_id=test_admin.id,
        action="prediction.created",
        resource_type="prediction",
        resource_id="pred-001",
        details='{"confidence": 0.92}',
    )
    db_session.add(log)
    await db_session.flush()
    await db_session.refresh(log)
    return log


# ── Tenant info ───────────────────────────────────────────────────────────────

class TestTenantInfo:
    @pytest.mark.asyncio
    async def test_admin_can_get_own_tenant(
        self, client: AsyncClient, admin_headers: dict, test_tenant: Tenant
    ):
        resp = await client.get("/api/v1/admin/tenant", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == test_tenant.id
        assert data["slug"] == test_tenant.slug

    @pytest.mark.asyncio
    async def test_operator_cannot_get_tenant(
        self, client: AsyncClient, operator_headers: dict
    ):
        resp = await client.get("/api/v1/admin/tenant", headers=operator_headers)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_unauthenticated_cannot_get_tenant(self, client: AsyncClient):
        resp = await client.get("/api/v1/admin/tenant")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_tenant_response_has_user_count(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.get("/api/v1/admin/tenant", headers=admin_headers)
        assert "user_count" in resp.json()
        assert isinstance(resp.json()["user_count"], int)


# ── User management ───────────────────────────────────────────────────────────

class TestUserManagement:
    @pytest.mark.asyncio
    async def test_admin_lists_users(
        self, client: AsyncClient, admin_headers: dict, test_admin: User, test_operator: User
    ):
        resp = await client.get("/api/v1/admin/users", headers=admin_headers)
        assert resp.status_code == 200
        emails = [u["email"] for u in resp.json()]
        assert test_admin.email in emails

    @pytest.mark.asyncio
    async def test_operator_cannot_list_users(
        self, client: AsyncClient, operator_headers: dict
    ):
        resp = await client.get("/api/v1/admin/users", headers=operator_headers)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_creates_user(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/admin/users",
            json={
                "email": "newuser@test-sme.com",
                "full_name": "New User",
                "role": "viewer",
                "password": "Secure123!",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "newuser@test-sme.com"
        assert data["role"] == "viewer"

    @pytest.mark.asyncio
    async def test_admin_cannot_create_duplicate_email(
        self, client: AsyncClient, admin_headers: dict, test_admin: User
    ):
        resp = await client.post(
            "/api/v1/admin/users",
            json={
                "email": test_admin.email,
                "full_name": "Dup",
                "role": "viewer",
                "password": "Secure123!",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_create_user_invalid_role_rejected(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/admin/users",
            json={
                "email": "badrole@test-sme.com",
                "full_name": "Bad Role",
                "role": "superuser",
                "password": "Secure123!",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_admin_updates_user_role(
        self, client: AsyncClient, admin_headers: dict, test_operator: User
    ):
        resp = await client.patch(
            f"/api/v1/admin/users/{test_operator.id}",
            json={"role": "engineer"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "engineer"

    @pytest.mark.asyncio
    async def test_admin_cannot_deactivate_self(
        self, client: AsyncClient, admin_headers: dict, test_admin: User
    ):
        resp = await client.patch(
            f"/api/v1/admin/users/{test_admin.id}",
            json={"is_active": False},
            headers=admin_headers,
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_admin_deactivates_other_user(
        self, client: AsyncClient, admin_headers: dict, test_operator: User
    ):
        resp = await client.delete(
            f"/api/v1/admin/users/{test_operator.id}",
            headers=admin_headers,
        )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_update_nonexistent_user_returns_404(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.patch(
            "/api/v1/admin/users/nonexistent-id",
            json={"role": "operator"},
            headers=admin_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_filter_users_by_role(
        self, client: AsyncClient, admin_headers: dict, test_admin: User
    ):
        resp = await client.get(
            "/api/v1/admin/users?role=admin",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        for user in resp.json():
            assert user["role"] == "admin"


# ── Audit log ─────────────────────────────────────────────────────────────────

class TestAuditLog:
    @pytest.mark.asyncio
    async def test_admin_retrieves_audit_log(
        self, client: AsyncClient, admin_headers: dict, audit_entry: AuditLog
    ):
        resp = await client.get("/api/v1/audit", headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_audit_log_contains_entry(
        self, client: AsyncClient, admin_headers: dict, audit_entry: AuditLog
    ):
        resp = await client.get("/api/v1/audit", headers=admin_headers)
        ids = [e["id"] for e in resp.json()]
        assert audit_entry.id in ids

    @pytest.mark.asyncio
    async def test_operator_cannot_read_audit_log(
        self, client: AsyncClient, operator_headers: dict
    ):
        resp = await client.get("/api/v1/audit", headers=operator_headers)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_audit_log_filter_by_action(
        self, client: AsyncClient, admin_headers: dict, audit_entry: AuditLog
    ):
        resp = await client.get(
            "/api/v1/audit?action=prediction.created",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        for entry in resp.json():
            assert entry["action"] == "prediction.created"

    @pytest.mark.asyncio
    async def test_audit_log_entry_has_required_fields(
        self, client: AsyncClient, admin_headers: dict, audit_entry: AuditLog
    ):
        resp = await client.get("/api/v1/audit", headers=admin_headers)
        entries = resp.json()
        if entries:
            entry = entries[0]
            for field in ["id", "action", "resource_type", "created_at"]:
                assert field in entry


# ── Model registry ────────────────────────────────────────────────────────────

class TestModelRegistry:
    @pytest.mark.asyncio
    async def test_admin_lists_models(
        self, client: AsyncClient, admin_headers: dict, model_entry: ModelRegistry
    ):
        resp = await client.get("/api/v1/admin/models", headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_model_entry_in_list(
        self, client: AsyncClient, admin_headers: dict, model_entry: ModelRegistry
    ):
        resp = await client.get("/api/v1/admin/models", headers=admin_headers)
        ids = [m["id"] for m in resp.json()]
        assert model_entry.id in ids

    @pytest.mark.asyncio
    async def test_admin_activates_model(
        self, client: AsyncClient, admin_headers: dict, model_entry: ModelRegistry
    ):
        resp = await client.post(
            f"/api/v1/admin/models/{model_entry.id}/activate",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

    @pytest.mark.asyncio
    async def test_activate_nonexistent_model_returns_404(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/admin/models/nonexistent-id/activate",
            headers=admin_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_operator_cannot_list_models(
        self, client: AsyncClient, operator_headers: dict
    ):
        resp = await client.get("/api/v1/admin/models", headers=operator_headers)
        assert resp.status_code == 403


# ── System stats ──────────────────────────────────────────────────────────────

class TestSystemStats:
    @pytest.mark.asyncio
    async def test_admin_gets_stats(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.get("/api/v1/admin/stats", headers=admin_headers)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_has_required_fields(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.get("/api/v1/admin/stats", headers=admin_headers)
        data = resp.json()
        for field in [
            "total_users", "active_users", "total_predictions",
            "total_alerts", "unacknowledged_alerts", "active_models",
        ]:
            assert field in data

    @pytest.mark.asyncio
    async def test_stats_non_negative_values(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.get("/api/v1/admin/stats", headers=admin_headers)
        data = resp.json()
        for key, val in data.items():
            if isinstance(val, (int, float)):
                assert val >= 0, f"{key} should be non-negative"

    @pytest.mark.asyncio
    async def test_operator_cannot_get_stats(
        self, client: AsyncClient, operator_headers: dict
    ):
        resp = await client.get("/api/v1/admin/stats", headers=operator_headers)
        assert resp.status_code == 403
