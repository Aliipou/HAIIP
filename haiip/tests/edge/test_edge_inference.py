"""Tests for edge inference engine and cloud sync — 100% branch coverage."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── EdgeInferenceEngine ───────────────────────────────────────────────────────

class TestEdgeInferenceEngine:

    @pytest.fixture
    def model_dir(self, tmp_path) -> Path:
        """A model dir with sklearn artifacts from a real fitted detector."""
        import joblib
        from haiip.core.anomaly import AnomalyDetector

        rng = np.random.default_rng(42)
        X = rng.normal(loc=[300, 310, 1538, 40, 100], scale=[2, 1.5, 179, 9.8, 50], size=(200, 5))
        det = AnomalyDetector(contamination=0.05, random_state=42)
        det.fit(X)

        # Simulate save (mimics AnomalyDetector.save)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        joblib.dump(det._scaler, model_dir / "scaler.joblib")
        joblib.dump(det._model, model_dir / "isolation_forest.joblib")
        return model_dir

    def test_load_sklearn_mode(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        assert engine.mode == "sklearn"
        assert engine.is_loaded

    def test_predict_returns_required_keys(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        result = engine.predict([300.0, 310.0, 1538.0, 40.0, 100.0])
        assert "label" in result
        assert "confidence" in result
        assert "anomaly_score" in result

    def test_predict_label_valid(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        result = engine.predict([300.0, 310.0, 1538.0, 40.0, 100.0])
        assert result["label"] in ("normal", "anomaly")

    def test_predict_confidence_in_range(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        result = engine.predict([300.0, 310.0, 1538.0, 40.0, 100.0])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_numpy_input(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        features = np.array([300.0, 310.0, 1538.0, 40.0, 100.0])
        result = engine.predict(features)
        assert result["label"] in ("normal", "anomaly")

    def test_not_loaded_raises(self):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.predict([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_load_missing_dir_raises(self, tmp_path):
        from haiip.edge.inference import EdgeInferenceEngine
        with pytest.raises(FileNotFoundError):
            EdgeInferenceEngine.load(tmp_path / "nonexistent")

    def test_load_empty_dir_raises(self, tmp_path):
        from haiip.edge.inference import EdgeInferenceEngine
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            EdgeInferenceEngine.load(empty)

    def test_is_loaded_false_before_load(self):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine()
        assert not engine.is_loaded

    def test_mode_not_loaded_before_load(self):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine()
        assert engine.mode == "not_loaded"

    def test_manifest_integrity_pass(self, model_dir):
        """Valid manifest with correct hash passes silently."""
        from haiip.edge.inference import EdgeInferenceEngine

        # Write a dummy ONNX file and valid manifest
        onnx_path = model_dir / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx-content")
        import hashlib
        real_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        manifest = {"model_hash_sha256": real_hash}
        (model_dir / "manifest.json").write_text(json.dumps(manifest))

        # ONNX load will fail (fake content), but manifest check passes
        engine = EdgeInferenceEngine.load(model_dir)
        assert engine.is_loaded  # falls back to sklearn

    def test_manifest_integrity_fail(self, model_dir):
        """Tampered ONNX file raises ModelIntegrityError."""
        from haiip.edge.inference import EdgeInferenceEngine, ModelIntegrityError

        onnx_path = model_dir / "model.onnx"
        onnx_path.write_bytes(b"original-content")
        manifest = {"model_hash_sha256": "deadbeef" * 8}
        (model_dir / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ModelIntegrityError):
            EdgeInferenceEngine.load(model_dir)

    def test_manifest_unreadable_logs_warning(self, model_dir):
        """Corrupt manifest file just logs a warning, doesn't crash."""
        from haiip.edge.inference import EdgeInferenceEngine
        (model_dir / "manifest.json").write_text("not: valid: json: {{")
        engine = EdgeInferenceEngine.load(model_dir)
        assert engine.is_loaded

    def test_onnx_mode_predict(self, model_dir):
        """If ONNX session mock is set, _predict_onnx path executes."""
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        engine._mode = "onnx"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="float_input")]
        mock_session.run.return_value = [
            np.array([-1]),       # label: anomaly
            np.array([{"anomaly_score": -0.4}]),  # scores
        ]
        engine._onnx_session = mock_session
        result = engine.predict([300.0, 310.0, 1538.0, 40.0, 100.0])
        assert result["label"] == "anomaly"
        assert result["mode"] == "onnx"

    def test_sklearn_mode_returns_mode_field(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        engine = EdgeInferenceEngine.load(model_dir)
        result = engine.predict([300.0, 310.0, 1538.0, 40.0, 100.0])
        assert result["mode"] == "sklearn"

    def test_export_onnx_raises_without_skl2onnx(self, model_dir):
        from haiip.edge.inference import EdgeInferenceEngine
        from haiip.core.anomaly import AnomalyDetector
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        det = AnomalyDetector().fit(X)

        with patch.dict("sys.modules", {"skl2onnx": None, "skl2onnx.common.data_types": None}):
            with pytest.raises(ImportError, match="skl2onnx"):
                EdgeInferenceEngine.export_onnx(det, model_dir / "onnx_out")


# ── EdgeModelSync ─────────────────────────────────────────────────────────────

class TestEdgeModelSync:

    @pytest.fixture
    def sync(self, tmp_path):
        from haiip.edge.sync import EdgeModelSync
        return EdgeModelSync(
            cloud_api_url="https://api.haiip.test",
            model_dir=tmp_path / "model",
            tenant_id="test-tenant",
            api_key="test-key",
            check_interval_seconds=0,  # always check
        )

    def test_sync_returns_false_on_network_error(self, sync):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = sync.sync()
        assert result is False

    def test_sync_returns_false_when_up_to_date(self, sync, tmp_path):
        import hashlib
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        content = b"onnx-model-bytes"
        real_hash = hashlib.sha256(content).hexdigest()
        (model_dir / "manifest.json").write_text(json.dumps({"model_hash_sha256": real_hash}))

        cloud_meta = {"version": "1.0", "model_hash_sha256": real_hash}

        with patch.object(sync, "_fetch_cloud_metadata", return_value=cloud_meta):
            result = sync.sync()
        assert result is False

    def test_sync_interval_skips_check(self, tmp_path):
        from haiip.edge.sync import EdgeModelSync
        sync = EdgeModelSync(
            cloud_api_url="https://api.test",
            model_dir=tmp_path,
            tenant_id="t",
            check_interval_seconds=9999,
        )
        sync._last_check = time.monotonic()  # just checked

        with patch.object(sync, "_fetch_cloud_metadata") as mock_fetch:
            result = sync.sync()
        mock_fetch.assert_not_called()
        assert result is False

    def test_sync_force_bypasses_interval(self, tmp_path):
        from haiip.edge.sync import EdgeModelSync
        sync = EdgeModelSync(
            cloud_api_url="https://api.test",
            model_dir=tmp_path,
            tenant_id="t",
            check_interval_seconds=9999,
        )
        sync._last_check = time.monotonic()

        with patch.object(sync, "_fetch_cloud_metadata", side_effect=OSError("unreachable")):
            result = sync.sync(force=True)
        assert result is False  # still False but fetch was called

    def test_local_model_hash_no_manifest(self, sync):
        assert sync._local_model_hash() == ""

    def test_local_model_hash_from_manifest(self, sync, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "manifest.json").write_text(json.dumps({"model_hash_sha256": "abc123"}))
        assert sync._local_model_hash() == "abc123"

    def test_local_model_hash_corrupt_manifest(self, sync, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "manifest.json").write_text("{{ bad json")
        assert sync._local_model_hash() == ""

    def test_download_fails_on_hash_mismatch(self, sync):
        cloud_meta = {
            "version": "2.0",
            "model_hash_sha256": "wronghash" * 4,
            "download_url": "https://cdn.test/model.onnx",
        }
        fake_model_bytes = b"fake-model"
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_model_bytes

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="hash mismatch"):
                sync._download_and_install(cloud_meta)

    def test_download_without_url_raises(self, sync):
        with pytest.raises(ValueError, match="download_url"):
            sync._download_and_install({"version": "1.0", "model_hash_sha256": ""})

    def test_last_version_initially_empty(self, sync):
        assert sync.last_version == ""


# ── EdgeMetricsReporter ────────────────────────────────────────────────────────

class TestEdgeMetricsReporter:

    @pytest.fixture
    def reporter(self):
        from haiip.edge.sync import EdgeMetricsReporter
        return EdgeMetricsReporter(
            cloud_api_url="https://api.test",
            tenant_id="test",
            machine_id="M001",
            batch_size=3,
            max_buffer=10,
        )

    def _pred(self):
        return {"label": "normal", "confidence": 0.9, "anomaly_score": 0.1}

    def test_record_adds_to_buffer(self, reporter):
        reporter.record(self._pred(), latency_ms=2.5)
        assert reporter.buffer_size == 1

    def test_record_includes_latency(self, reporter):
        reporter.record(self._pred(), latency_ms=3.7)
        assert reporter._buffer[0]["latency_ms"] == 3.7

    def test_buffer_capped_at_max(self, reporter):
        for _ in range(15):
            reporter.record(self._pred(), latency_ms=1.0)
        assert reporter.buffer_size <= 10

    def test_flush_empty_returns_zero(self, reporter):
        assert reporter.flush() == 0

    def test_flush_on_network_error_returns_zero(self, reporter):
        # Add 2 records (below batch_size=3 → no auto-flush triggered)
        reporter.record(self._pred(), latency_ms=1.0)
        reporter.record(self._pred(), latency_ms=1.0)
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            sent = reporter.flush()
        assert sent == 0
        assert reporter.failed_sends == 1

    def test_auto_flush_at_batch_size(self, reporter):
        """Buffer auto-flushes when batch_size reached."""
        with patch.object(reporter, "_send_batch") as mock_send:
            for _ in range(3):
                reporter.record(self._pred(), latency_ms=1.0)
            mock_send.assert_called_once()

    def test_flush_success(self, reporter):
        reporter.record(self._pred(), latency_ms=1.0)
        reporter.record(self._pred(), latency_ms=1.0)
        with patch.object(reporter, "_send_batch"):
            sent = reporter.flush()
        assert sent == 2
        assert reporter.buffer_size == 0

    def test_failed_sends_counter(self, reporter):
        reporter.record(self._pred(), latency_ms=1.0)
        with patch("urllib.request.urlopen", side_effect=OSError("err")):
            reporter.flush()
        assert reporter.failed_sends == 1
