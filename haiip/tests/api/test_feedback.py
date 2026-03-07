"""Tests for feedback routes."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_submit_feedback_requires_auth(client: AsyncClient):
    response = await client.post(
        "/api/v1/feedback",
        json={
            "prediction_id": "some-id",
            "was_correct": True,
        },
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_submit_feedback_prediction_not_found(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/feedback",
        json={"prediction_id": "nonexistent-pred-id", "was_correct": True},
        headers=operator_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_submit_feedback_success(client: AsyncClient, operator_headers):
    # Create prediction first
    reading = {
        "machine_id": "FEEDBACK-MACHINE",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1538.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }
    pred_resp = await client.post("/api/v1/predict", json=reading, headers=operator_headers)
    pred_id = pred_resp.json()["data"]["id"]

    # Submit feedback
    fb_resp = await client.post(
        "/api/v1/feedback",
        json={
            "prediction_id": pred_id,
            "was_correct": False,
            "corrected_label": "anomaly",
            "notes": "Operator confirmed this was a genuine anomaly",
        },
        headers=operator_headers,
    )
    assert fb_resp.status_code == 201
    data = fb_resp.json()
    assert data["prediction_id"] == pred_id
    assert data["was_correct"] is False
    assert data["corrected_label"] == "anomaly"


@pytest.mark.asyncio
async def test_submit_feedback_marks_prediction_verified(client: AsyncClient, operator_headers):
    reading = {
        "machine_id": "VERIFY-TEST",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1538.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }
    pred_resp = await client.post("/api/v1/predict", json=reading, headers=operator_headers)
    pred_id = pred_resp.json()["data"]["id"]

    await client.post(
        "/api/v1/feedback",
        json={"prediction_id": pred_id, "was_correct": True},
        headers=operator_headers,
    )

    # Verify prediction is now marked as human_verified
    pred_resp2 = await client.get(f"/api/v1/predictions/{pred_id}", headers=operator_headers)
    assert pred_resp2.json()["data"]["human_verified"] is True


@pytest.mark.asyncio
async def test_list_feedback(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/feedback", headers=operator_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_list_feedback_filter_by_prediction(client: AsyncClient, operator_headers):
    response = await client.get(
        "/api/v1/feedback?prediction_id=nonexistent",
        headers=operator_headers,
    )
    assert response.status_code == 200
    assert response.json() == []
