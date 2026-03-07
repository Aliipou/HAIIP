"""Tests for GDPR API routes (/api/v1/gdpr/...)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestGDPRErasure:
    @pytest.mark.asyncio
    async def test_erasure_returns_202(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/gdpr/erasure",
            json={"subject_id": "user-123"},
            headers=admin_headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["subject_id"] == "user-123"
        assert "request_id" in data
        assert "requested_at" in data

    @pytest.mark.asyncio
    async def test_erasure_custom_tables(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/gdpr/erasure",
            json={"subject_id": "machine-001", "tables": ["predictions"]},
            headers=admin_headers,
        )
        assert resp.status_code == 202
        assert resp.json()["tables_affected"] == ["predictions"]

    @pytest.mark.asyncio
    async def test_erasure_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/gdpr/erasure",
            json={"subject_id": "user-123"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_erasure_missing_subject_id(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/gdpr/erasure",
            json={},
            headers=admin_headers,
        )
        assert resp.status_code == 422


class TestGDPRExport:
    @pytest.mark.asyncio
    async def test_export_returns_200(self, client: AsyncClient, admin_headers: dict):
        resp = await client.get(
            "/api/v1/gdpr/export/user-123",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["subject_id"] == "user-123"
        assert "tables" in data
        assert "format" in data
        assert data["format"] == "GDPR-portable-JSON-v1"

    @pytest.mark.asyncio
    async def test_export_requires_auth(self, client: AsyncClient):
        resp = await client.get("/api/v1/gdpr/export/user-123")
        assert resp.status_code == 401


class TestGDPRPIIScan:
    @pytest.mark.asyncio
    async def test_scan_detects_email(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/gdpr/scan",
            json={"payload": {"message": "contact admin@test.com", "value": 300.0}},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_pii"] is True
        assert "email" in data["pii_fields_found"]

    @pytest.mark.asyncio
    async def test_scan_no_pii(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/gdpr/scan",
            json={"payload": {"vibration": 300.5, "temp": 72.1}},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_pii"] is False

    @pytest.mark.asyncio
    async def test_scan_returns_scrubbed_payload(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/gdpr/scan",
            json={"payload": {"email": "admin@haiip.ai", "reading": 100}},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        scrubbed = resp.json()["scrubbed"]
        assert scrubbed["email"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_scan_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/gdpr/scan",
            json={"payload": {"x": 1}},
        )
        assert resp.status_code == 401


class TestGDPRConsent:
    @pytest.mark.asyncio
    async def test_valid_consent(self, client: AsyncClient, admin_headers: dict):
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        resp = await client.post(
            "/api/v1/gdpr/consent/validate",
            json={
                "subject_id": "u-1",
                "tenant_id": "t-1",
                "purpose": "predictive_maintenance",
                "granted_at": recent,
                "legal_basis": "legitimate_interest",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["missing_fields"] == []
        assert data["expired"] is False

    @pytest.mark.asyncio
    async def test_expired_consent(self, client: AsyncClient, admin_headers: dict):
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        resp = await client.post(
            "/api/v1/gdpr/consent/validate",
            json={
                "subject_id": "u-1",
                "tenant_id": "t-1",
                "purpose": "maintenance",
                "granted_at": old,
                "legal_basis": "consent",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["expired"] is True
        assert data["valid"] is False

    @pytest.mark.asyncio
    async def test_consent_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/gdpr/consent/validate",
            json={},
        )
        assert resp.status_code in (401, 422)


class TestGDPRHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        resp = await client.get("/api/v1/gdpr/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
