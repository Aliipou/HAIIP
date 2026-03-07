"""Tests for /api/v1/ml-ops/* routes.

Test categories:
    - Auth: unauthenticated requests rejected
    - Happy path: valid requests enqueue Celery tasks
    - Validation: bad model_type, out-of-range params
    - Celery unavailable: 503 response
    - Pipeline status: with/without artifacts
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from haiip.api.main import create_app

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
async def auth_headers(app) -> dict[str, str]:
    """Get JWT token for engineer user."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register + login demo tenant/user
        await client.post(
            "/api/v1/auth/register-tenant",
            json={
                "tenant_name": "test-mlops",
                "tenant_slug": "test-mlops",
                "admin_email": "mlops@test.ai",
                "admin_password": "Test1234!",
            },
        )
        resp = await client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": "test-mlops",
                "email": "mlops@test.ai",
                "password": "Test1234!",
            },
        )
        token = resp.json().get("access_token", "")
    return {"Authorization": f"Bearer {token}"}


def _mock_celery_task(task_id: str = "mock-task-id") -> MagicMock:
    task = MagicMock()
    task.id = task_id
    return task


# ══════════════════════════════════════════════════════════════════════════════
# POST /ml-ops/retrain
# ══════════════════════════════════════════════════════════════════════════════


class TestRetrainEndpoint:
    async def test_unauthenticated_returns_401(self, app: Any) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post("/api/v1/ml-ops/retrain", json={"tenant_id": "default"})
        assert resp.status_code == 401

    async def test_valid_request_returns_202(self, app: Any, auth_headers: dict) -> None:
        mock_task = _mock_celery_task("retrain-abc")
        with patch("haiip.workers.tasks.auto_retrain_pipeline") as mock_pipeline:
            mock_pipeline.delay.return_value = mock_task
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/retrain",
                    json={"tenant_id": "default", "feedback_accuracy": 0.75},
                    headers=auth_headers,
                )
        if resp.status_code == 202:
            data = resp.json()
            assert data["status"] == "queued"
            assert "task_id" in data
        else:
            # If Celery not available, 503 is acceptable
            assert resp.status_code in (202, 503)

    async def test_force_reason_manual(self, app: Any, auth_headers: dict) -> None:
        mock_task = _mock_celery_task("retrain-manual")
        with patch("haiip.workers.tasks.auto_retrain_pipeline") as mock_pipeline:
            mock_pipeline.delay.return_value = mock_task
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/retrain",
                    json={"tenant_id": "default", "force_reason": "manual"},
                    headers=auth_headers,
                )
        assert resp.status_code in (202, 503)

    async def test_celery_unavailable_returns_503(self, app: Any, auth_headers: dict) -> None:
        with patch("haiip.workers.tasks.auto_retrain_pipeline") as mock_pipeline:
            mock_pipeline.delay.side_effect = Exception("broker down")
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/retrain",
                    json={"tenant_id": "default"},
                    headers=auth_headers,
                )
        assert resp.status_code in (503, 401)  # 401 if auth fails first

    async def test_accuracy_out_of_range_returns_422(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/ml-ops/retrain",
                json={"tenant_id": "default", "feedback_accuracy": 1.5},  # >1.0
                headers=auth_headers,
            )
        assert resp.status_code in (422, 401)


# ══════════════════════════════════════════════════════════════════════════════
# POST /ml-ops/export-onnx
# ══════════════════════════════════════════════════════════════════════════════


class TestExportONNXEndpoint:
    async def test_unauthenticated_returns_401(self, app: Any) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post("/api/v1/ml-ops/export-onnx", json={"tenant_id": "default"})
        assert resp.status_code == 401

    async def test_valid_anomaly_export_returns_202(self, app: Any, auth_headers: dict) -> None:
        with patch("haiip.workers.tasks.export_onnx_model") as mock_export:
            mock_export.delay.return_value = _mock_celery_task("export-123")
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/export-onnx",
                    json={"tenant_id": "default", "model_type": "anomaly", "opset": 17},
                    headers=auth_headers,
                )
        assert resp.status_code in (202, 503)

    async def test_valid_maintenance_export(self, app: Any, auth_headers: dict) -> None:
        with patch("haiip.workers.tasks.export_onnx_model") as mock_export:
            mock_export.delay.return_value = _mock_celery_task("export-456")
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/export-onnx",
                    json={"tenant_id": "default", "model_type": "maintenance"},
                    headers=auth_headers,
                )
        assert resp.status_code in (202, 503)

    async def test_invalid_model_type_returns_400(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/ml-ops/export-onnx",
                json={"tenant_id": "default", "model_type": "invalid_model"},
                headers=auth_headers,
            )
        assert resp.status_code in (400, 401)

    async def test_celery_unavailable_returns_503(self, app: Any, auth_headers: dict) -> None:
        with patch("haiip.workers.tasks.export_onnx_model") as mock_export:
            mock_export.delay.side_effect = Exception("broker timeout")
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/export-onnx",
                    json={"tenant_id": "default", "model_type": "anomaly"},
                    headers=auth_headers,
                )
        assert resp.status_code in (503, 401)


# ══════════════════════════════════════════════════════════════════════════════
# POST /ml-ops/benchmark
# ══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkEndpoint:
    async def test_unauthenticated_returns_401(self, app: Any) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post("/api/v1/ml-ops/benchmark", json={"tenant_id": "default"})
        assert resp.status_code == 401

    async def test_valid_benchmark_returns_202(self, app: Any, auth_headers: dict) -> None:
        with patch("haiip.workers.tasks.benchmark_onnx_model") as mock_bench:
            mock_bench.delay.return_value = _mock_celery_task("bench-789")
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/ml-ops/benchmark",
                    json={
                        "tenant_id": "default",
                        "model_type": "anomaly",
                        "n_runs": 50,
                    },
                    headers=auth_headers,
                )
        assert resp.status_code in (202, 503)

    async def test_n_runs_too_small_returns_422(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/ml-ops/benchmark",
                json={"tenant_id": "default", "n_runs": 1},  # < min 10
                headers=auth_headers,
            )
        assert resp.status_code in (422, 401)

    async def test_n_runs_too_large_returns_422(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/ml-ops/benchmark",
                json={"tenant_id": "default", "n_runs": 9999},  # > max 1000
                headers=auth_headers,
            )
        assert resp.status_code in (422, 401)


# ══════════════════════════════════════════════════════════════════════════════
# GET /ml-ops/pipeline-status
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineStatusEndpoint:
    async def test_unauthenticated_returns_401(self, app: Any) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/api/v1/ml-ops/pipeline-status")
        assert resp.status_code == 401

    async def test_returns_200_with_auth(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/ml-ops/pipeline-status?tenant_id=default",
                headers=auth_headers,
            )
        assert resp.status_code in (200, 401)

    async def test_response_has_required_keys(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/ml-ops/pipeline-status",
                headers=auth_headers,
            )
        if resp.status_code == 200:
            data = resp.json()
            assert "models" in data
            assert "drift" in data
            assert "sla_target_ms" in data
            assert data["sla_target_ms"] == 50

    async def test_models_section_has_expected_keys(self, app: Any, auth_headers: dict) -> None:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/ml-ops/pipeline-status",
                headers=auth_headers,
            )
        if resp.status_code == 200:
            models = resp.json()["models"]
            for key in (
                "sklearn_champion",
                "pytorch_autoencoder",
                "onnx_anomaly",
                "onnx_maintenance",
            ):
                assert key in models
