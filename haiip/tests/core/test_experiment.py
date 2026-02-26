"""Tests for ExperimentTracker — MLOps experiment tracking.

Covers:
1. Run creation and lifecycle
2. Context manager (start_run)
3. Metric and parameter logging
4. Querying (best run, list runs, summarise)
5. Persistence (JSON round-trip)
6. Edge cases (empty experiments, missing metrics)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from haiip.core.experiment import ExperimentTracker, Run, get_tracker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return ExperimentTracker()


@pytest.fixture
def persistent_tracker(tmp_path):
    return ExperimentTracker(storage_path=tmp_path / "experiments")


# ── 1. Run creation ───────────────────────────────────────────────────────────

class TestRunCreation:
    def test_create_run_returns_run(self, tracker):
        run = tracker.create_run("test_exp")
        assert isinstance(run, Run)

    def test_run_has_unique_id(self, tracker):
        r1 = tracker.create_run("exp")
        r2 = tracker.create_run("exp")
        assert r1.run_id != r2.run_id

    def test_run_initial_status_running(self, tracker):
        run = tracker.create_run("exp")
        assert run.status == "running"

    def test_run_initial_metrics_empty(self, tracker):
        run = tracker.create_run("exp")
        assert run.metrics == {}

    def test_run_initial_params_empty(self, tracker):
        run = tracker.create_run("exp")
        assert run.params == {}

    def test_run_with_tags(self, tracker):
        run = tracker.create_run("exp", tags={"dataset": "ai4i", "version": "1.0"})
        assert run.tags["dataset"] == "ai4i"
        assert run.tags["version"] == "1.0"

    def test_run_stored_in_tracker(self, tracker):
        run = tracker.create_run("exp")
        assert tracker.get_run(run.run_id) is run


# ── 2. Context manager ────────────────────────────────────────────────────────

class TestContextManager:
    def test_start_run_sets_finished_on_exit(self, tracker):
        with tracker.start_run("exp") as run:
            run.log_metric("accuracy", 0.92)
        assert run.status == "finished"

    def test_start_run_sets_failed_on_exception(self, tracker):
        with pytest.raises(ValueError):
            with tracker.start_run("exp") as run:
                raise ValueError("Training diverged")
        assert run.status == "failed"

    def test_start_run_records_duration(self, tracker):
        with tracker.start_run("exp") as run:
            pass
        assert run.duration_seconds is not None
        assert run.duration_seconds >= 0

    def test_start_run_records_end_time(self, tracker):
        with tracker.start_run("exp") as run:
            pass
        assert run.end_time is not None


# ── 3. Logging params and metrics ─────────────────────────────────────────────

class TestLogging:
    def test_log_param_stored(self, tracker):
        run = tracker.create_run("exp")
        run.log_param("n_estimators", 200)
        assert run.params["n_estimators"] == 200

    def test_log_metric_stored(self, tracker):
        run = tracker.create_run("exp")
        run.log_metric("auc_roc", 0.934)
        assert abs(run.metrics["auc_roc"] - 0.934) < 1e-6

    def test_log_metric_rounded(self, tracker):
        run = tracker.create_run("exp")
        run.log_metric("f1", 0.123456789)
        assert len(str(run.metrics["f1"]).split(".")[-1]) <= 6

    def test_log_artifact(self, tracker):
        run = tracker.create_run("exp")
        run.log_artifact("models/anomaly_v2.pkl")
        assert "models/anomaly_v2.pkl" in run.artifacts

    def test_set_tag(self, tracker):
        run = tracker.create_run("exp")
        run.set_tag("environment", "production")
        assert run.tags["environment"] == "production"

    def test_set_status_finished(self, tracker):
        run = tracker.create_run("exp")
        run.set_status("finished")
        assert run.status == "finished"

    def test_set_status_invalid_raises(self, tracker):
        run = tracker.create_run("exp")
        with pytest.raises(AssertionError):
            run.set_status("canceled")

    def test_multiple_metrics(self, tracker):
        run = tracker.create_run("exp")
        run.log_metric("auc_roc", 0.91)
        run.log_metric("f1_macro", 0.84)
        run.log_metric("precision", 0.86)
        assert len(run.metrics) == 3


# ── 4. Querying ───────────────────────────────────────────────────────────────

class TestQuerying:
    def test_list_runs_by_experiment(self, tracker):
        tracker.create_run("exp_a")
        tracker.create_run("exp_a")
        tracker.create_run("exp_b")
        assert len(tracker.list_runs("exp_a")) == 2
        assert len(tracker.list_runs("exp_b")) == 1

    def test_list_runs_empty_experiment(self, tracker):
        assert tracker.list_runs("nonexistent") == []

    def test_get_best_run_higher_is_better(self, tracker):
        with tracker.start_run("anomaly") as r1:
            r1.log_metric("auc_roc", 0.88)
        with tracker.start_run("anomaly") as r2:
            r2.log_metric("auc_roc", 0.94)  # best
        with tracker.start_run("anomaly") as r3:
            r3.log_metric("auc_roc", 0.76)

        best = tracker.get_best_run("anomaly", metric="auc_roc")
        assert best is not None
        assert abs(best.metrics["auc_roc"] - 0.94) < 1e-6

    def test_get_best_run_lower_is_better(self, tracker):
        with tracker.start_run("rul") as r1:
            r1.log_metric("mae", 18.5)
        with tracker.start_run("rul") as r2:
            r2.log_metric("mae", 12.3)  # best (lower)

        best = tracker.get_best_run("rul", metric="mae", higher_is_better=False)
        assert best is not None
        assert abs(best.metrics["mae"] - 12.3) < 1e-6

    def test_get_best_run_only_finished(self, tracker):
        """Failed runs should not be considered for best_run."""
        with pytest.raises(ValueError):
            with tracker.start_run("exp") as run:
                run.log_metric("f1", 0.99)  # great metric
                raise ValueError("crashed")

        # Best run should be None (only run was failed)
        best = tracker.get_best_run("exp", metric="f1")
        assert best is None

    def test_get_best_run_no_metric(self, tracker):
        """If metric is not present in run, exclude from best."""
        with tracker.start_run("exp") as r:
            r.log_metric("accuracy", 0.91)  # no "auc_roc"
        best = tracker.get_best_run("exp", metric="auc_roc")
        assert best is None

    def test_summarise_returns_summary(self, tracker):
        with tracker.start_run("exp") as r:
            r.log_metric("f1", 0.82)
        summary = tracker.summarise("exp", metric="f1")
        assert summary.name == "exp"
        assert summary.run_count == 1
        assert summary.best_run_id is not None

    def test_all_experiments(self, tracker):
        tracker.create_run("exp_1")
        tracker.create_run("exp_2")
        tracker.create_run("exp_1")
        exps = tracker.all_experiments()
        assert "exp_1" in exps
        assert "exp_2" in exps
        assert len(exps) == 2


# ── 5. Persistence ────────────────────────────────────────────────────────────

class TestPersistence:
    def test_run_persisted_to_disk(self, persistent_tracker):
        with persistent_tracker.start_run("exp") as run:
            run.log_metric("auc_roc", 0.91)
            run.log_param("n_estimators", 200)

        storage = persistent_tracker._storage_path
        json_files = list(storage.glob("*.json"))
        assert len(json_files) == 1

    def test_run_loaded_from_disk(self, tmp_path):
        storage = tmp_path / "exp_store"
        tracker1 = ExperimentTracker(storage_path=storage)
        with tracker1.start_run("exp") as run:
            run.log_metric("f1", 0.88)
            run_id = run.run_id

        # New tracker instance loads from disk
        tracker2 = ExperimentTracker(storage_path=storage)
        loaded = tracker2.get_run(run_id)
        assert loaded is not None
        assert abs(loaded.metrics["f1"] - 0.88) < 1e-6

    def test_run_dict_round_trip(self, tracker):
        run = tracker.create_run("exp", tags={"model": "v1"})
        run.log_param("lr", 0.001)
        run.log_metric("accuracy", 0.95)
        run.log_artifact("model.pkl")
        run.set_status("finished")

        d = run.to_dict()
        assert d["run_id"] == run.run_id
        assert d["metrics"]["accuracy"] == run.metrics["accuracy"]
        assert d["params"]["lr"] == 0.001
        assert d["status"] == "finished"
        assert "model.pkl" in d["artifacts"]


# ── 6. Singleton ──────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_tracker_returns_same_instance(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_get_tracker_returns_experiment_tracker(self):
        t = get_tracker()
        assert isinstance(t, ExperimentTracker)
