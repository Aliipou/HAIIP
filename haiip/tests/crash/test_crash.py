"""Crash / robustness tests — edge cases, malformed inputs, resource exhaustion.

Tests verify the system does NOT crash under adversarial or unexpected conditions.
Every test should pass — meaning the system handles the bad input gracefully,
returning an appropriate error code rather than throwing a 500.

Categories:
1. Malformed API inputs (validation, type errors)
2. Missing required fields
3. Extreme numeric values (overflow, underflow, NaN, Inf)
4. Very large payloads
5. ML model edge cases (untrained, empty features)
6. Concurrent access simulation
7. Invalid auth tokens

Note: These tests intentionally send bad data. A passing test means the API
returned 4xx, NOT 5xx (unless a 500 is specifically expected and handled).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from haiip.api.models import Tenant

# ── API Input Validation Crash Tests ─────────────────────────────────────────


class TestMalformedInputs:
    @pytest.mark.asyncio
    async def test_predict_with_string_features_rejected(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": "CNC-001",
                "features": "not_a_dict",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_with_null_machine_id_rejected(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": None,
                "features": {"air_temperature": 298.0},
            },
            headers=admin_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_with_empty_body_rejected(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/predict",
            json={},
            headers=admin_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_with_no_body_rejected(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/predict",
            content=b"",
            headers={**admin_headers, "Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_with_invalid_json_rejected(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/predict",
            content=b"{{invalid json}}",
            headers={**admin_headers, "Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_alert_with_invalid_severity(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/alerts",
            json={
                "machine_id": "CNC-001",
                "severity": "catastrophic",  # not in enum
                "title": "Bad severity",
                "message": "Test",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_with_invalid_email(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "notanemail",
                "password": "Pass123!",
                "full_name": "Bad Email",
                "tenant_slug": "test-sme",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_with_short_password(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "valid@test.com",
                "password": "123",  # too short
                "full_name": "Short Pass",
                "tenant_slug": "test-sme",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_feedback_nonexistent_prediction(self, client: AsyncClient, admin_headers: dict):
        """Feedback for a non-existent prediction should not crash the server."""
        resp = await client.post(
            "/api/v1/feedback",
            json={
                "prediction_id": "00000000-0000-0000-0000-000000000000",
                "was_correct": True,
            },
            headers=admin_headers,
        )
        # Should be 4xx, not 500
        assert resp.status_code < 500


# ── Extreme Numeric Values ────────────────────────────────────────────────────


class TestExtremeValues:
    @pytest.mark.asyncio
    async def test_predict_with_very_large_temperature(
        self, client: AsyncClient, admin_headers: dict
    ):
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": "EXTREME-001",
                "features": {
                    "air_temperature": 1e18,  # absurdly large
                    "process_temperature": 1e18,
                    "rotational_speed": 1500,
                    "torque": 40.0,
                    "tool_wear": 0,
                },
            },
            headers=admin_headers,
        )
        # Must not be 500 — either validates and returns prediction, or returns 422
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_predict_with_negative_values(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": "NEG-001",
                "features": {
                    "air_temperature": -273.15,
                    "process_temperature": -100.0,
                    "rotational_speed": -999,
                    "torque": -50.0,
                    "tool_wear": -1,
                },
            },
            headers=admin_headers,
        )
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_predict_with_zero_features(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": "ZERO-001",
                "features": {
                    "air_temperature": 0.0,
                    "process_temperature": 0.0,
                    "rotational_speed": 0,
                    "torque": 0.0,
                    "tool_wear": 0,
                },
            },
            headers=admin_headers,
        )
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_predict_with_empty_features_dict(self, client: AsyncClient, admin_headers: dict):
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": "EMPTY-FEAT",
                "features": {},
            },
            headers=admin_headers,
        )
        # Either handled gracefully with a prediction or 422 validation error
        assert resp.status_code != 500


# ── Large Payload Tests ───────────────────────────────────────────────────────


class TestLargePayloads:
    @pytest.mark.asyncio
    async def test_predict_with_many_extra_features(self, client: AsyncClient, admin_headers: dict):
        """Very large feature dict should not crash."""
        features = {f"sensor_{i}": float(i) for i in range(500)}
        resp = await client.post(
            "/api/v1/predict",
            json={"machine_id": "BIG-001", "features": features},
            headers=admin_headers,
        )
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_alert_message_very_long(self, client: AsyncClient, admin_headers: dict):
        long_msg = "A" * 10_000
        resp = await client.post(
            "/api/v1/alerts",
            json={
                "machine_id": "LONG-001",
                "severity": "low",
                "title": "Long message test",
                "message": long_msg,
            },
            headers=admin_headers,
        )
        assert resp.status_code != 500

    @pytest.mark.asyncio
    async def test_machine_id_max_length(self, client: AsyncClient, admin_headers: dict):
        long_id = "M" * 200
        resp = await client.post(
            "/api/v1/predict",
            json={
                "machine_id": long_id,
                "features": {"air_temperature": 298.0},
            },
            headers=admin_headers,
        )
        assert resp.status_code != 500


# ── Authentication Edge Cases ─────────────────────────────────────────────────


class TestAuthEdgeCases:
    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, client: AsyncClient):
        """A crafted expired token must return 401, not 500."""
        # JWT with past expiry (manually crafted)
        fake_expired = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiJ1c3ItMDAxIiwidGlkIjoidGVzdCIsInJvbGUiOiJhZG1pbiIsImV4cCI6MX0."
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        )
        resp = await client.get(
            "/api/v1/predictions",
            headers={"Authorization": f"Bearer {fake_expired}"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_token_rejected(self, client: AsyncClient):
        resp = await client.get(
            "/api/v1/predictions",
            headers={"Authorization": "Bearer notajwtatall"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_bearer_prefix_rejected(self, client: AsyncClient):
        resp = await client.get(
            "/api/v1/predictions",
            headers={"Authorization": "sometoken"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_authorization_header_rejected(self, client: AsyncClient):
        resp = await client.get(
            "/api/v1/predictions",
            headers={"Authorization": ""},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_credentials_rejected(self, client: AsyncClient, test_tenant: Tenant):
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": "nobody@test-sme.com", "password": "WrongPass!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code in (401, 422)

    @pytest.mark.asyncio
    async def test_nonexistent_tenant_login_rejected(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@nonexistent-tenant.com", "password": "Pass123!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code in (401, 422)


# ── ML Engine Edge Cases ──────────────────────────────────────────────────────


class TestMLEngineEdgeCases:
    def test_anomaly_detector_handles_single_sample(self):
        from haiip.core.anomaly import AnomalyDetector

        detector = AnomalyDetector()
        detector.fit([[1.0, 2.0, 3.0, 4.0, 5.0]] * 50)
        result = detector.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result is not None
        assert 0.0 <= result["confidence"] <= 1.0

    def test_anomaly_detector_handles_untrained(self):
        from haiip.core.anomaly import AnomalyDetector

        detector = AnomalyDetector()
        # Should not crash — returns safe default
        result = detector.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result is not None
        assert "label" in result

    def test_anomaly_detector_all_same_features(self):
        """Constant features should not cause division by zero."""
        from haiip.core.anomaly import AnomalyDetector

        detector = AnomalyDetector()
        data = [[5.0, 5.0, 5.0, 5.0, 5.0]] * 100
        detector.fit(data)
        result = detector.predict([5.0, 5.0, 5.0, 5.0, 5.0])
        assert result is not None
        assert "label" in result

    def test_maintenance_predictor_handles_untrained(self):
        from haiip.core.maintenance import MaintenancePredictor

        predictor = MaintenancePredictor()
        result = predictor.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result is not None
        assert "label" in result

    def test_drift_detector_requires_fit_before_check(self):
        import pytest

        from haiip.core.drift import DriftDetector

        detector = DriftDetector(feature_names=["temp"])
        # Without fit_reference, check() should raise RuntimeError
        import numpy as np

        with pytest.raises(RuntimeError):
            detector.check(np.array([[298.0]]))

    def test_feedback_engine_zero_samples(self):
        from haiip.core.feedback import FeedbackEngine

        engine = FeedbackEngine()
        # With no feedback, no retraining needed
        state = engine.get_state()
        assert not state.needs_retraining

    def test_compliance_engine_empty_detect_incidents(self):
        from haiip.core.compliance import ComplianceEngine

        engine = ComplianceEngine()
        # Empty events → no incidents
        assert engine.detect_incidents() == []

    def test_rag_engine_query_empty_index(self):
        """Query with empty RAG index should not crash."""
        from haiip.core.rag import RAGEngine

        rag = RAGEngine()
        result = rag.query("What is the maintenance schedule?")
        assert result is not None


# ── Concurrent Request Simulation ────────────────────────────────────────────


class TestConcurrentAccess:
    @pytest.mark.asyncio
    async def test_multiple_simultaneous_predictions(
        self, client: AsyncClient, admin_headers: dict
    ):
        """10 concurrent predictions should all succeed."""
        import asyncio

        async def make_prediction(i: int):
            return await client.post(
                "/api/v1/predict",
                json={
                    "machine_id": f"CONCURRENT-{i:03d}",
                    "features": {
                        "air_temperature": 298.0 + i,
                        "process_temperature": 308.0,
                        "rotational_speed": 1500,
                        "torque": 40.0,
                        "tool_wear": i,
                    },
                },
                headers=admin_headers,
            )

        responses = await asyncio.gather(*[make_prediction(i) for i in range(10)])
        for resp in responses:
            assert resp.status_code in (
                200,
                201,
                429,
            )  # 429 = rate limited (acceptable)

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_alert_creates(
        self, client: AsyncClient, admin_headers: dict
    ):
        """Concurrent alert creation should not cause DB constraint violations."""
        import asyncio

        async def create_alert(i: int):
            return await client.post(
                "/api/v1/alerts",
                json={
                    "machine_id": f"CONC-ALERT-{i}",
                    "severity": "low",
                    "title": f"Concurrent test {i}",
                    "message": "Concurrency test alert",
                },
                headers=admin_headers,
            )

        responses = await asyncio.gather(*[create_alert(i) for i in range(5)])
        success_count = sum(1 for r in responses if r.status_code in (200, 201))
        assert success_count >= 3  # at least 3 of 5 succeed
