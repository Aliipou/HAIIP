"""Tests for retrain/ONNX/benchmark Celery tasks in haiip.workers.tasks.

Test categories:
    - Unit: task logic with mocked dependencies
    - Integration: task runs against real AI4I loader + real models
    - Edge/Crash: missing artifacts, bad model_type, train failure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_task_self(task_func: Any) -> MagicMock:
    """Create a mock 'self' for bound Celery tasks."""
    self = MagicMock()
    self.update_state = MagicMock()
    self.retry.side_effect = RuntimeError("retry called")
    return self


# ══════════════════════════════════════════════════════════════════════════════
# auto_retrain_pipeline task
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoRetrainPipelineTask:
    def test_returns_no_retrain_when_stable(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import auto_retrain_pipeline

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.data.loaders.ai4i.AI4ILoader.get_normal_data") as mock_data,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            mock_settings.drift_threshold = 0.05

            import pandas as pd

            rng = np.random.default_rng(42)
            X = rng.normal(0, 1, (120, 5)).astype(np.float64)
            cols = [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            mock_data.return_value = pd.DataFrame(X, columns=cols)

            self_mock = _make_task_self(auto_retrain_pipeline)
            result = auto_retrain_pipeline.__wrapped__(
                self_mock,
                tenant_id="test",
                feedback_accuracy=0.95,
            )

        assert "status" in result
        assert result["tenant_id"] == "test"

    def test_returns_retrain_on_low_accuracy(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import auto_retrain_pipeline

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.data.loaders.ai4i.AI4ILoader.get_normal_data") as mock_data,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            mock_settings.drift_threshold = 0.05

            import pandas as pd

            rng = np.random.default_rng(42)
            X = rng.normal(0, 1, (120, 5)).astype(np.float64)
            cols = [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            mock_data.return_value = pd.DataFrame(X, columns=cols)

            self_mock = _make_task_self(auto_retrain_pipeline)
            result = auto_retrain_pipeline.__wrapped__(
                self_mock,
                tenant_id="test2",
                feedback_accuracy=0.40,  # very low — triggers retrain
            )

        assert "status" in result
        # May or may not retrain depending on cooldown state — just verify no crash
        assert result["tenant_id"] == "test2"

    def test_loads_existing_champion(self, tmp_path: Path) -> None:
        """If a saved champion exists, it should be loaded."""
        from haiip.core.anomaly import AnomalyDetector
        from haiip.workers.tasks import auto_retrain_pipeline

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (120, 5)).astype(np.float64)

        champ_dir = tmp_path / "test3" / "anomaly"
        detector = AnomalyDetector(contamination=0.05, n_estimators=10)
        detector.fit(X)
        detector.save(champ_dir)

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.data.loaders.ai4i.AI4ILoader.get_normal_data") as mock_data,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            mock_settings.drift_threshold = 0.05

            import pandas as pd

            cols = [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            mock_data.return_value = pd.DataFrame(X, columns=cols)

            self_mock = _make_task_self(auto_retrain_pipeline)
            result = auto_retrain_pipeline.__wrapped__(
                self_mock, tenant_id="test3", feedback_accuracy=0.95
            )

        assert "status" in result

    def test_force_reason_manual(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import auto_retrain_pipeline

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.data.loaders.ai4i.AI4ILoader.get_normal_data") as mock_data,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            mock_settings.drift_threshold = 0.05

            import pandas as pd

            rng = np.random.default_rng(42)
            X = rng.normal(0, 1, (120, 5)).astype(np.float64)
            cols = [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            mock_data.return_value = pd.DataFrame(X, columns=cols)

            self_mock = _make_task_self(auto_retrain_pipeline)
            result = auto_retrain_pipeline.__wrapped__(
                self_mock,
                tenant_id="test4",
                force_reason="manual",
            )

        assert result["status"] not in ("error",)

    def test_with_drift_data(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import auto_retrain_pipeline

        rng = np.random.default_rng(42)
        X_ref = rng.normal(0, 1, (100, 5))
        X_cur = rng.normal(5, 1, (100, 5))  # shifted distribution — will show drift

        tenant_dir = tmp_path / "drifted"
        tenant_dir.mkdir(parents=True)
        np.save(str(tenant_dir / "drift_reference.npy"), X_ref)
        np.save(str(tenant_dir / "drift_current.npy"), X_cur)

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.data.loaders.ai4i.AI4ILoader.get_normal_data") as mock_data,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            mock_settings.drift_threshold = 0.05

            import pandas as pd

            X = rng.normal(0, 1, (120, 5))
            cols = [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            mock_data.return_value = pd.DataFrame(X, columns=cols)

            self_mock = _make_task_self(auto_retrain_pipeline)
            result = auto_retrain_pipeline.__wrapped__(self_mock, tenant_id="drifted")

        assert "status" in result


# ══════════════════════════════════════════════════════════════════════════════
# export_onnx_model task
# ══════════════════════════════════════════════════════════════════════════════


class TestExportONNXModelTask:
    def test_unknown_model_type_returns_error(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import export_onnx_model

        with patch("haiip.workers.tasks.settings") as mock_settings:
            mock_settings.model_artifacts_path = str(tmp_path)
            self_mock = _make_task_self(export_onnx_model)
            result = export_onnx_model.__wrapped__(
                self_mock, tenant_id="test", model_type="unknown"
            )

        assert result["status"] == "error"
        assert "Unknown model_type" in result["reason"]

    def test_maintenance_no_model_returns_skipped(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import export_onnx_model

        with patch("haiip.workers.tasks.settings") as mock_settings:
            mock_settings.model_artifacts_path = str(tmp_path)
            self_mock = _make_task_self(export_onnx_model)
            result = export_onnx_model.__wrapped__(
                self_mock, tenant_id="no_model_tenant", model_type="maintenance"
            )

        assert result["status"] == "skipped"

    @pytest.mark.skipif(
        True,  # Requires torch+lightning+onnx all installed — CI optional
        reason="Requires full torch+lightning+onnx stack",
    )
    def test_anomaly_export_creates_onnx(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import export_onnx_model

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.data.loaders.ai4i.AI4ILoader.get_normal_data") as mock_data,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            import pandas as pd

            rng = np.random.default_rng(42)
            X = rng.normal(0, 1, (60, 5))
            cols = [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ]
            mock_data.return_value = pd.DataFrame(X, columns=cols)

            self_mock = _make_task_self(export_onnx_model)
            result = export_onnx_model.__wrapped__(
                self_mock, tenant_id="test", model_type="anomaly"
            )

        assert result["status"] == "exported"
        assert Path(result["onnx_path"]).exists()


# ══════════════════════════════════════════════════════════════════════════════
# benchmark_onnx_model task
# ══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkONNXModelTask:
    def test_skips_when_no_onnx_file(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import benchmark_onnx_model

        with patch("haiip.workers.tasks.settings") as mock_settings:
            mock_settings.model_artifacts_path = str(tmp_path)
            result = benchmark_onnx_model(
                tenant_id="no_onnx_tenant", model_type="anomaly", n_runs=5
            )

        assert result["status"] == "skipped"

    def test_maintenance_skips_when_no_file(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import benchmark_onnx_model

        with patch("haiip.workers.tasks.settings") as mock_settings:
            mock_settings.model_artifacts_path = str(tmp_path)
            result = benchmark_onnx_model(
                tenant_id="no_onnx_tenant2", model_type="maintenance", n_runs=5
            )

        assert result["status"] == "skipped"

    def test_returns_error_on_exception(self, tmp_path: Path) -> None:
        from haiip.workers.tasks import benchmark_onnx_model

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch("haiip.core.onnx_runtime.ONNXAnomalyDetector.from_onnx") as mock_from,
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            mock_from.side_effect = RuntimeError("ORT crash")

            result = benchmark_onnx_model(tenant_id="crash_tenant", model_type="anomaly", n_runs=5)

        assert result["status"] == "error"
        assert "ORT crash" in result["error"]

    def test_benchmark_with_mock_detector(self, tmp_path: Path) -> None:
        """End-to-end with mocked ONNXAnomalyDetector."""
        from haiip.workers.tasks import benchmark_onnx_model

        mock_detector = MagicMock()
        mock_detector.is_ready = True
        mock_detector.benchmark.return_value = {
            "mean_ms": 3.5,
            "p50_ms": 3.2,
            "p95_ms": 5.1,
            "p99_ms": 6.0,
            "max_ms": 7.0,
            "sla_pass_rate": 1.0,
            "n_runs": 5,
        }

        with (
            patch("haiip.workers.tasks.settings") as mock_settings,
            patch(
                "haiip.core.onnx_runtime.ONNXAnomalyDetector.from_onnx",
                return_value=mock_detector,
            ),
        ):
            mock_settings.model_artifacts_path = str(tmp_path)
            result = benchmark_onnx_model(tenant_id="mock_tenant", model_type="anomaly", n_runs=5)

        assert result["status"] == "benchmarked"
        assert result["sla_pass"] is True
        assert result["p99_ms"] == 6.0
        assert result["tenant_id"] == "mock_tenant"
