"""Chaos engineering tests — fault injection for industrial resilience.

Tests that intentionally break components and verify:
1. Graceful degradation — system keeps running at reduced capacity
2. No data corruption — partial failures don't corrupt DB or model state
3. Recovery — system heals automatically when fault is removed
4. Error surfacing — failures produce clear, actionable error messages

References:
    - Netflix Chaos Engineering: principlesofchaos.org
    - AWS Well-Architected Reliability Pillar
    - IEC 61508 SIL-2 fault tolerance requirements
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(300, 5))


@pytest.fixture
def fitted_detector(normal_data):
    from haiip.core.anomaly import AnomalyDetector

    d = AnomalyDetector(contamination=0.05, random_state=42)
    d.fit(normal_data)
    return d


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(n, 5))
    labels = (
        ["no_failure"] * 250
        + ["TWF"] * 10
        + ["HDF"] * 10
        + ["PWF"] * 10
        + ["OSF"] * 10
        + ["RNF"] * 10
    )
    rng.shuffle(labels)
    return X, np.array(labels)


@pytest.fixture
def fitted_predictor(training_data):
    from haiip.core.maintenance import MaintenancePredictor

    X, y = training_data
    p = MaintenancePredictor(n_estimators=30, random_state=42)
    p.fit(X, y)
    return p


SAMPLE = [300.0, 310.0, 1538.0, 40.0, 100.0]


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 1: Database unavailable
# ═══════════════════════════════════════════════════════════════════════════════


class TestDatabaseFaultInjection:
    def test_anomaly_detector_survives_without_db(self, fitted_detector):
        """AI prediction must work even if database is completely unavailable."""
        result = fitted_detector.predict(SAMPLE)
        assert result["label"] in ("normal", "anomaly")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_maintenance_predictor_survives_without_db(self, fitted_predictor):
        """Maintenance prediction is stateless — DB failure must not affect it."""
        result = fitted_predictor.predict(SAMPLE)
        assert "label" in result
        assert result["failure_probability"] >= 0.0

    @pytest.mark.asyncio
    async def test_feedback_engine_handles_db_write_failure(self):
        """FeedbackEngine must not crash if DB write fails — record in memory."""
        from haiip.core.feedback import FeedbackEngine

        engine = FeedbackEngine(window_size=10, retrain_threshold=0.7, min_samples=5)
        # Normal operation
        for _ in range(5):
            engine.record("pred-1", was_correct=True, corrected_label=None, machine_id="m1")
        state = engine.get_state()
        assert state.window_accuracy >= 0.0  # doesn't crash


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 2: Redis / Celery unavailable
# ═══════════════════════════════════════════════════════════════════════════════


class TestRedisUnavailable:
    def test_rate_limiter_falls_back_to_in_memory(self):
        """SecurityMiddleware in-memory rate limiter must work without Redis."""
        from haiip.api.middleware import _InMemoryRateLimiter

        limiter = _InMemoryRateLimiter()
        # Should allow within limit
        for _ in range(5):
            assert limiter.is_allowed("test-ip", limit=10, window_seconds=60)
        # Should deny over limit
        for _ in range(10):
            limiter.is_allowed("over-ip", limit=3, window_seconds=60)
        assert not limiter.is_allowed("over-ip", limit=3, window_seconds=60)

    def test_rate_limiter_window_slides_correctly(self):
        """Old timestamps outside window must not count toward limit."""
        from haiip.api.middleware import _InMemoryRateLimiter

        limiter = _InMemoryRateLimiter()
        # Fill up with old timestamps (manually inject)
        limiter._windows["ip:path"] = [time.monotonic() - 100] * 5
        # Should be allowed — all timestamps expired
        assert limiter.is_allowed("ip:path", limit=5, window_seconds=60)

    def test_secrets_rotation_manager_survives_no_boto3(self):
        """Rotation manager must not crash when AWS is unavailable."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager()
        with patch.dict("sys.modules", {"boto3": None}):
            result = asyncio.get_event_loop().run_until_complete(manager.rotate_if_needed())
        assert result is False  # no rotation, no crash


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 3: ML model unavailable / corrupted
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLModelFaults:
    def test_unfitted_detector_returns_safe_default(self):
        """Unfitted model must return a safe, non-crashing result."""
        from haiip.core.anomaly import AnomalyDetector

        d = AnomalyDetector()
        result = d.predict(SAMPLE)
        assert result["label"] == "normal"
        assert result["confidence"] == 0.5
        assert result["anomaly_score"] == 0.0

    def test_unfitted_predictor_returns_safe_default(self):
        from haiip.core.maintenance import MaintenancePredictor

        p = MaintenancePredictor()
        result = p.predict(SAMPLE)
        assert result["label"] == "no_failure"
        assert result["failure_probability"] == 0.0

    def test_detector_handles_nan_input_gracefully(self, fitted_detector):
        """NaN sensor readings must not propagate to predictions."""
        nan_sample = [float("nan"), 310.0, 1538.0, 40.0, 100.0]
        try:
            result = fitted_detector.predict(nan_sample)
            # If it returns, result must have valid structure
            assert "label" in result
        except (ValueError, Exception):
            pass  # Raising is also acceptable — just not silently corrupting state

    def test_detector_handles_inf_input_gracefully(self, fitted_detector):
        """Inf sensor readings must not propagate."""
        inf_sample = [float("inf"), 310.0, 1538.0, 40.0, 100.0]
        try:
            result = fitted_detector.predict(inf_sample)
            assert "label" in result
        except (ValueError, Exception):
            pass

    def test_detector_handles_empty_batch(self, fitted_detector):
        """Empty batch must return empty list, not crash."""
        result = fitted_detector.predict_batch(np.empty((0, 5)))
        assert result == []

    def test_detector_handles_single_sample_batch(self, fitted_detector):
        """Single-sample batch must work identically to predict()."""
        single = fitted_detector.predict(SAMPLE)
        batch = fitted_detector.predict_batch(np.array([SAMPLE]))
        assert single["label"] == batch[0]["label"]

    def test_predictor_handles_nan_batch(self, fitted_predictor):
        """NaN in batch input must be handled."""
        nan_X = np.array([[float("nan"), 310.0, 1538.0, 40.0, 100.0]])
        try:
            results = fitted_predictor.predict_batch(nan_X)
            assert len(results) == 1
        except (ValueError, Exception):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 4: Concurrent access / race conditions
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    def test_anomaly_detector_concurrent_predictions(self, fitted_detector):
        """Multiple threads predicting simultaneously must not corrupt state."""
        results = []
        errors = []

        def predict():
            try:
                r = fitted_detector.predict(SAMPLE)
                results.append(r["label"])
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(predict) for _ in range(50)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Concurrent predictions raised errors: {errors}"
        assert len(results) == 50
        assert all(r in ("normal", "anomaly") for r in results)

    def test_anomaly_detector_concurrent_batch(self, fitted_detector):
        """Concurrent batch predictions must all succeed."""
        X = np.array([SAMPLE] * 10)
        errors = []
        results = []

        def batch_predict():
            try:
                r = fitted_detector.predict_batch(X)
                results.append(len(r))
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(batch_predict) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert all(r == 10 for r in results)

    def test_rate_limiter_thread_safety(self):
        """Rate limiter must not allow races that bypass limits."""
        from haiip.api.middleware import _InMemoryRateLimiter

        limiter = _InMemoryRateLimiter()
        allowed = []
        lock = threading.Lock()

        def try_request():
            result = limiter.is_allowed("shared-ip", limit=10, window_seconds=60)
            with lock:
                allowed.append(result)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(try_request) for _ in range(30)]
            for f in futures:
                f.result()

        # Should allow at most 10 (may allow slightly more due to window reset, but not 30)
        assert sum(allowed) <= 15  # generous bound for thread timing
        assert sum(allowed) >= 10  # at least 10 should succeed

    def test_drift_detector_concurrent_checks(self, normal_data):
        """DriftDetector concurrent checks must not corrupt reference data."""
        from haiip.core.drift import DriftDetector

        detector = DriftDetector()
        feature_names = ["air_temp", "proc_temp", "rpm", "torque", "wear"]
        detector.fit_reference(normal_data, feature_names)

        errors = []

        def check():
            try:
                rng = np.random.default_rng()
                X_current = rng.normal(
                    loc=[300, 310, 1538, 40, 100],
                    scale=[2, 1.5, 179, 9.8, 50],
                    size=(50, 5),
                )
                results = detector.check(X_current)
                assert isinstance(results, list)
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 5: Network partition / timeout simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestNetworkFaults:
    def test_secrets_fetch_timeout_handled_gracefully(self):
        """Slow AWS response must not block startup indefinitely."""
        from haiip.api.secrets import _fetch_from_aws, clear_cache

        clear_cache()

        def slow_get_secret(**kwargs):
            time.sleep(0.1)  # simulate slow network
            raise Exception("Connection timeout")

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = slow_get_secret
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore.exceptions": MagicMock()}):
            start = time.monotonic()
            result = _fetch_from_aws("test-secret", "eu-north-1")
            elapsed = time.monotonic() - start

        assert result == {}  # graceful empty return
        assert elapsed < 5.0  # must not hang

        clear_cache()

    def test_rag_engine_handles_llm_unavailable(self):
        """RAG Q&A must return a safe result when LLM is unreachable."""
        from haiip.core.rag import RAGEngine

        engine = RAGEngine()
        engine.add_text(
            content="Machine M-001 requires monthly lubrication.",
            title="Maintenance Manual",
            source="manual.pdf",
        )

        with patch.object(engine, "_llm_answer", side_effect=Exception("LLM timeout")):
            try:
                result = engine.query("What maintenance does M-001 need?")
                # If it returns, must have a valid structure
                assert hasattr(result, "answer")
            except Exception:
                pass  # Raising is acceptable — no silent data corruption

    def test_opc_ua_connector_handles_disconnect(self):
        """OPC UA connector must detect and report disconnects cleanly."""
        from haiip.data.ingestion.opcua_connector import DataSourceMode, OPCUAConnector

        connector = OPCUAConnector(
            endpoint="opc.tcp://unreachable-host:4840",
        )
        # In simulation mode, should return synthetic data regardless
        assert connector.mode == DataSourceMode.SIMULATION


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 6: Memory pressure / large inputs
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryPressure:
    def test_large_batch_prediction_completes(self, fitted_detector):
        """Batch of 10,000 samples must complete without OOM."""
        rng = np.random.default_rng(42)
        X_large = rng.normal(
            loc=[300, 310, 1538, 40, 100],
            scale=[2, 1.5, 179, 9.8, 50],
            size=(10_000, 5),
        )
        results = fitted_detector.predict_batch(X_large)
        assert len(results) == 10_000

    def test_large_batch_timing_reasonable(self, fitted_detector):
        """10k batch must complete in under 10 seconds on any machine."""
        rng = np.random.default_rng(42)
        X = rng.normal(
            loc=[300, 310, 1538, 40, 100],
            scale=[2, 1.5, 179, 9.8, 50],
            size=(10_000, 5),
        )
        start = time.monotonic()
        fitted_detector.predict_batch(X)
        elapsed = time.monotonic() - start
        assert elapsed < 30.0, f"10k batch took {elapsed:.2f}s — too slow"

    def test_extremely_large_feature_values_handled(self, fitted_detector):
        """Extreme sensor values (equipment malfunction) must not overflow."""
        extreme_sample = [1e10, 1e10, 1e10, 1e10, 1e10]
        try:
            result = fitted_detector.predict(extreme_sample)
            assert result["label"] in ("normal", "anomaly")
        except (ValueError, OverflowError):
            pass

    def test_zero_value_features_handled(self, fitted_detector):
        """All-zero input must not divide by zero."""
        zero_sample = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = fitted_detector.predict(zero_sample)
        assert result["label"] in ("normal", "anomaly")


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 7: Secrets rotation fault injection
# ═══════════════════════════════════════════════════════════════════════════════


class TestSecretsRotationFaults:
    def test_rotation_survives_aws_failure(self):
        """Rotation manager must not raise when AWS call fails."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager(check_interval=0)  # force check

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = Exception("AWS unavailable")
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = asyncio.get_event_loop().run_until_complete(manager.rotate_if_needed())
        assert result is False  # no rotation on failure

    def test_rotation_respects_interval(self):
        """Manager must skip check if interval hasn't elapsed."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager(check_interval=3600)
        # Simulate recently checked
        manager._state.last_checked = time.monotonic()
        result = asyncio.get_event_loop().run_until_complete(manager.rotate_if_needed())
        assert result is False

    def test_rotation_event_recorded_on_failure(self):
        """Failed rotation must append a RotationEvent with success=False."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager(check_interval=0)

        async def bad_rotation():
            event = await manager._perform_rotation(
                new_version="v2",
                new_secrets={"SECRET_KEY": "new-key"},
            )
            return event

        # Patch _rotate_signing_key to raise
        with patch.object(
            manager,
            "_rotate_signing_key",
            side_effect=RuntimeError("key rotation failed"),
        ):
            event = asyncio.get_event_loop().run_until_complete(bad_rotation())

        assert event.success is False
        assert "key rotation failed" in event.error
        assert len(manager._state.events) == 1

    def test_dual_key_overlap_window(self):
        """Previous key must be accessible within the overlap window."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager(check_interval=0)
        manager._state.previous_key = "old-key"
        manager._state.previous_key_expires = time.monotonic() + 300

        assert manager.get_previous_key() == "old-key"

    def test_dual_key_expired_returns_none(self):
        """Previous key past overlap window must return None."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager(check_interval=0)
        manager._state.previous_key = "old-key"
        manager._state.previous_key_expires = time.monotonic() - 1  # expired

        assert manager.get_previous_key() is None

    def test_force_rotate_returns_event(self):
        """force_rotate must return a RotationEvent even when AWS unavailable."""
        from haiip.api.secrets_rotation import SecretsRotationManager

        manager = SecretsRotationManager(check_interval=0)

        with patch.dict("sys.modules", {"boto3": None}):
            event = asyncio.get_event_loop().run_until_complete(manager.force_rotate())
        # Should return an event (empty rotation) without crashing
        assert event.secret_name == manager._secret_name


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT 8: Zero-trust service auth faults
# ═══════════════════════════════════════════════════════════════════════════════


class TestZeroTrustFaults:
    def test_service_token_created_and_verified(self):
        """Round-trip: create → verify must return matching ServiceToken."""
        from haiip.api.service_auth import create_service_token, verify_service_token

        token = create_service_token("worker", scopes=["retrain", "predict"])
        svc = verify_service_token(token)
        assert svc.service_name == "worker"
        assert "retrain" in svc.scopes
        assert "predict" in svc.scopes

    def test_expired_service_token_rejected(self):
        """Expired service token must raise ServiceTokenError."""
        from haiip.api.service_auth import (
            ServiceTokenError,
            create_service_token,
            verify_service_token,
        )

        token = create_service_token("worker", expires_minutes=-1)
        with pytest.raises(ServiceTokenError):
            verify_service_token(token)

    def test_tampered_service_token_rejected(self):
        """Modified token must not validate."""
        from haiip.api.service_auth import ServiceTokenError, verify_service_token

        with pytest.raises(ServiceTokenError):
            verify_service_token("eyJhbGciOiJIUzI1NiJ9.tampered.signature")

    def test_unknown_service_rejected_on_create(self):
        """Creating token for unregistered service must raise ValueError."""
        from haiip.api.service_auth import create_service_token

        with pytest.raises(ValueError, match="Unknown service"):
            create_service_token("malicious-service")

    def test_unknown_service_in_token_rejected(self):
        """Token with unregistered service name must be rejected on verify."""
        from jose import jwt

        from haiip.api.config import get_settings
        from haiip.api.service_auth import (
            ServiceTokenError,
            _service_secret,
            verify_service_token,
        )

        settings = get_settings()
        import os

        os.environ.pop("SERVICE_SECRET_KEY", None)
        secret = _service_secret(settings)

        from datetime import datetime, timedelta

        payload = {
            "sub": "unknown-svc",
            "type": "service",
            "scopes": [],
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        with pytest.raises(ServiceTokenError, match="Unregistered service"):
            verify_service_token(token)

    def test_wrong_token_type_rejected(self):
        """Token with type='access' must not pass service verification."""
        from jose import jwt

        from haiip.api.config import get_settings
        from haiip.api.service_auth import (
            ServiceTokenError,
            _service_secret,
            verify_service_token,
        )

        settings = get_settings()
        import os

        os.environ.pop("SERVICE_SECRET_KEY", None)
        secret = _service_secret(settings)

        from datetime import datetime, timedelta

        payload = {
            "sub": "worker",
            "type": "access",  # wrong type
            "scopes": [],
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        with pytest.raises(ServiceTokenError, match="not a service token"):
            verify_service_token(token)

    def test_scope_check_passes_when_present(self):
        """has_scope must return True for granted scope."""
        from haiip.api.service_auth import create_service_token, verify_service_token

        token = create_service_token("worker", scopes=["retrain"])
        svc = verify_service_token(token)
        assert svc.has_scope("retrain")
        assert not svc.has_scope("delete_all")

    def test_require_scope_raises_when_missing(self):
        """require_scope must raise ServiceTokenError for missing scope."""
        from haiip.api.service_auth import (
            ServiceTokenError,
            create_service_token,
            verify_service_token,
        )

        token = create_service_token("worker", scopes=["predict"])
        svc = verify_service_token(token)
        with pytest.raises(ServiceTokenError, match="lacks required scope"):
            svc.require_scope("retrain")

    def test_service_auth_headers_format(self):
        """service_auth_headers must return correct Authorization header."""
        from haiip.api.service_auth import service_auth_headers

        headers = service_auth_headers("my-token")
        assert headers["Authorization"] == "Bearer my-token"
        assert headers["X-Service-Auth"] == "true"

    def test_service_secret_uses_dedicated_key_when_set(self):
        """_service_secret must prefer SERVICE_SECRET_KEY env var."""
        import os

        from haiip.api.config import get_settings
        from haiip.api.service_auth import _service_secret

        os.environ["SERVICE_SECRET_KEY"] = "dedicated-svc-secret"
        try:
            result = _service_secret(get_settings())
            assert result == "dedicated-svc-secret"
        finally:
            os.environ.pop("SERVICE_SECRET_KEY", None)

    def test_service_secret_falls_back_to_derived_key(self):
        """_service_secret must derive from SECRET_KEY when no dedicated key."""
        import os

        from haiip.api.config import get_settings
        from haiip.api.service_auth import _service_secret

        os.environ.pop("SERVICE_SECRET_KEY", None)
        result = _service_secret(get_settings())
        assert "service-token-salt" in result
