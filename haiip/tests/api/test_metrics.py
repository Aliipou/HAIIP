"""Tests for metrics routes."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_system_health(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/metrics/health", headers=operator_headers)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "database" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_system_health_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/metrics/health")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_machine_metrics_empty(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/metrics/machines", headers=operator_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_machine_metrics_after_predictions(client: AsyncClient, operator_headers):
    # Create a prediction first
    reading = {
        "machine_id": "METRICS-TEST-001",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1538.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }
    await client.post("/api/v1/predict", json=reading, headers=operator_headers)

    response = await client.get("/api/v1/metrics/machines", headers=operator_headers)
    assert response.status_code == 200
    machines = response.json()
    machine_ids = [m["machine_id"] for m in machines]
    assert "METRICS-TEST-001" in machine_ids


@pytest.mark.asyncio
async def test_alert_summary(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/metrics/alerts/summary", headers=operator_headers)
    assert response.status_code == 200
    data = response.json()
    assert "critical" in data
    assert "high" in data
    assert "medium" in data
    assert "low" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_health_endpoint_no_auth(client: AsyncClient):
    """Public /health endpoint should not require auth."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
