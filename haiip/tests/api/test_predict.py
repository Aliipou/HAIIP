"""Tests for prediction routes."""

import pytest
from httpx import AsyncClient

VALID_READING = {
    "machine_id": "MACHINE-001",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1538.0,
    "torque": 40.0,
    "tool_wear": 100.0,
}


@pytest.mark.asyncio
async def test_predict_single_success(client: AsyncClient, operator_headers):
    response = await client.post("/api/v1/predict", json=VALID_READING, headers=operator_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    result = data["data"]
    assert result["machine_id"] == "MACHINE-001"
    assert result["prediction_label"] in ("normal", "anomaly")
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_predict_requires_auth(client: AsyncClient):
    response = await client.post("/api/v1/predict", json=VALID_READING)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_predict_invalid_temperature(client: AsyncClient, operator_headers):
    bad = {**VALID_READING, "air_temperature": 9999.0}  # out of range
    response = await client.post("/api/v1/predict", json=bad, headers=operator_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_negative_torque(client: AsyncClient, operator_headers):
    bad = {**VALID_READING, "torque": -10.0}
    response = await client.post("/api/v1/predict", json=bad, headers=operator_headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_batch_success(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/predict/batch",
        json={
            "readings": [VALID_READING, {**VALID_READING, "machine_id": "MACHINE-002"}],
            "model_type": "anomaly_detection",
        },
        headers=operator_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_predict_batch_empty_rejected(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/predict/batch",
        json={"readings": [], "model_type": "anomaly_detection"},
        headers=operator_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_predictions(client: AsyncClient, operator_headers):
    # First create some predictions
    await client.post("/api/v1/predict", json=VALID_READING, headers=operator_headers)

    response = await client.get("/api/v1/predictions", headers=operator_headers)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_list_predictions_filter_by_machine(client: AsyncClient, operator_headers):
    response = await client.get(
        "/api/v1/predictions?machine_id=NONEXISTENT",
        headers=operator_headers,
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


@pytest.mark.asyncio
async def test_get_prediction_not_found(client: AsyncClient, operator_headers):
    response = await client.get(
        "/api/v1/predictions/nonexistent-id",
        headers=operator_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_predict_stores_to_db(client: AsyncClient, operator_headers):
    response = await client.post("/api/v1/predict", json=VALID_READING, headers=operator_headers)
    pred_id = response.json()["data"]["id"]

    get_response = await client.get(f"/api/v1/predictions/{pred_id}", headers=operator_headers)
    assert get_response.status_code == 200
    assert get_response.json()["data"]["id"] == pred_id
