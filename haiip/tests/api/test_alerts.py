"""Tests for alert routes."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_alerts_empty(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/alerts", headers=operator_headers)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_list_alerts_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/alerts")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_alert_engineer_only(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/alerts",
        params={
            "machine_id": "CNC-001",
            "severity": "high",
            "title": "Test Alert",
            "message": "Test message",
        },
        headers=operator_headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_alert_success(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/alerts",
        params={
            "machine_id": "CNC-001",
            "severity": "high",
            "title": "Anomaly Detected",
            "message": "Torque exceeded normal range",
        },
        headers=admin_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["machine_id"] == "CNC-001"
    assert data["severity"] == "high"
    assert data["is_acknowledged"] is False


@pytest.mark.asyncio
async def test_create_alert_invalid_severity(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/alerts",
        params={
            "machine_id": "CNC-001",
            "severity": "extreme",  # invalid
            "title": "Test",
            "message": "Test",
        },
        headers=admin_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_acknowledge_alert(client: AsyncClient, admin_headers):
    # Create alert first
    create_resp = await client.post(
        "/api/v1/alerts",
        params={"machine_id": "M-001", "severity": "medium", "title": "Test", "message": "Msg"},
        headers=admin_headers,
    )
    alert_id = create_resp.json()["id"]

    # Acknowledge it
    ack_resp = await client.patch(
        f"/api/v1/alerts/{alert_id}/acknowledge",
        json={},
        headers=admin_headers,
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["is_acknowledged"] is True


@pytest.mark.asyncio
async def test_acknowledge_already_acknowledged(client: AsyncClient, admin_headers):
    create_resp = await client.post(
        "/api/v1/alerts",
        params={"machine_id": "M-002", "severity": "low", "title": "Test", "message": "Msg"},
        headers=admin_headers,
    )
    alert_id = create_resp.json()["id"]

    await client.patch(f"/api/v1/alerts/{alert_id}/acknowledge", json={}, headers=admin_headers)
    second = await client.patch(
        f"/api/v1/alerts/{alert_id}/acknowledge", json={}, headers=admin_headers
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_get_alert_not_found(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/alerts/nonexistent-id", headers=operator_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_filter_unacknowledged(client: AsyncClient, admin_headers):
    response = await client.get(
        "/api/v1/alerts?unacknowledged_only=true",
        headers=admin_headers,
    )
    assert response.status_code == 200
