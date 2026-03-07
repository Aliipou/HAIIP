"""Tests for Active Learning API routes (/api/v1/active-learning/...)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestSelectSamples:
    @pytest.mark.asyncio
    async def test_select_uncertainty(self, client: AsyncClient, operator_headers: dict):
        preds = [
            {"label": "anomaly", "confidence": 0.51},
            {"label": "normal", "confidence": 0.90},
            {"label": "anomaly", "confidence": 0.52},
        ]
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={"predictions": preds, "strategy": "uncertainty", "budget": 2},
            headers=operator_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "uncertainty"
        assert len(data["selected_indices"]) == 2
        assert data["n_pool"] == 3

    @pytest.mark.asyncio
    async def test_select_margin_strategy(self, client: AsyncClient, operator_headers: dict):
        preds = [{"label": "anomaly", "confidence": 0.5 + i * 0.05} for i in range(10)]
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={"predictions": preds, "strategy": "margin", "budget": 3},
            headers=operator_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_select_entropy_strategy(self, client: AsyncClient, operator_headers: dict):
        preds = [{"label": "anomaly", "confidence": 0.5 + i * 0.05} for i in range(10)]
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={"predictions": preds, "strategy": "entropy", "budget": 3},
            headers=operator_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_select_random_strategy(self, client: AsyncClient, operator_headers: dict):
        preds = [{"label": "anomaly", "confidence": 0.5 + i * 0.05} for i in range(10)]
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={"predictions": preds, "strategy": "random", "budget": 3},
            headers=operator_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_select_invalid_strategy(self, client: AsyncClient, operator_headers: dict):
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={
                "predictions": [{"label": "anomaly", "confidence": 0.5}],
                "strategy": "bogus",
                "budget": 1,
            },
            headers=operator_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_select_empty_predictions(self, client: AsyncClient, operator_headers: dict):
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={"predictions": [], "strategy": "uncertainty", "budget": 5},
            headers=operator_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_indices"] == []

    @pytest.mark.asyncio
    async def test_select_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={"predictions": [], "strategy": "random", "budget": 1},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_select_confidence_floor(self, client: AsyncClient, operator_headers: dict):
        preds = [
            {"label": "anomaly", "confidence": 0.3},
            {"label": "anomaly", "confidence": 0.8},
        ]
        resp = await client.post(
            "/api/v1/active-learning/select",
            json={
                "predictions": preds,
                "strategy": "uncertainty",
                "budget": 2,
                "confidence_floor": 0.5,
            },
            headers=operator_headers,
        )
        assert resp.status_code == 200
        assert 0 not in resp.json()["selected_indices"]


class TestQueue:
    @pytest.mark.asyncio
    async def test_add_to_queue(self, client: AsyncClient, operator_headers: dict):
        samples = [{"id": i, "label": "anomaly", "confidence": 0.5} for i in range(3)]
        resp = await client.post(
            "/api/v1/active-learning/queue",
            json=samples,
            headers=operator_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["added"] == 3

    @pytest.mark.asyncio
    async def test_queue_stats(self, client: AsyncClient, operator_headers: dict):
        resp = await client.get(
            "/api/v1/active-learning/queue/stats",
            headers=operator_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "queue_size" in data
        assert "labeled_count" in data
        assert data["max_size"] == 500

    @pytest.mark.asyncio
    async def test_queue_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/active-learning/queue",
            json=[],
        )
        assert resp.status_code == 401


class TestLabel:
    @pytest.mark.asyncio
    async def test_label_out_of_range_empty_queue(
        self, client: AsyncClient, operator_headers: dict
    ):
        """Labeling when queue is at a non-existent index returns 404."""
        # Use a very large index that's certainly out of range
        resp = await client.post(
            "/api/v1/active-learning/label",
            json={"queue_index": 9999, "human_label": "normal"},
            headers=operator_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_label_requires_auth(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/active-learning/label",
            json={"queue_index": 0, "human_label": "normal"},
        )
        assert resp.status_code == 401


class TestDrain:
    @pytest.mark.asyncio
    async def test_drain_returns_response(self, client: AsyncClient, operator_headers: dict):
        resp = await client.post(
            "/api/v1/active-learning/drain",
            headers=operator_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "labeled_samples" in data

    @pytest.mark.asyncio
    async def test_drain_requires_auth(self, client: AsyncClient):
        resp = await client.post("/api/v1/active-learning/drain")
        assert resp.status_code == 401
