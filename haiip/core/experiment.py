"""MLOps experiment tracking — lightweight MLflow-compatible tracker.

Records model training runs with hyperparameters, metrics, and artifacts.
Designed for RDI reproducibility: every run is timestamped, tagged,
and queryable.

References:
    - Zaharia et al. (2018) Accelerating the Machine Learning Lifecycle with MLflow
    - Sculley et al. (2015) Hidden Technical Debt in Machine Learning Systems

Usage:
    tracker = ExperimentTracker()
    with tracker.start_run("anomaly_detector_v2", tags={"dataset": "ai4i_2020"}) as run:
        run.log_param("n_estimators", 200)
        run.log_param("contamination", 0.05)
        run.log_metric("auc_roc", 0.934)
        run.log_metric("f1_macro", 0.881)
        run.set_status("finished")
    best = tracker.get_best_run("anomaly_detector_v2", metric="auc_roc")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Run:
    """One experiment run — parameters, metrics, and metadata."""
    run_id: str
    experiment_name: str
    status: str  # "running" | "finished" | "failed"
    params: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, str]
    artifacts: list[str]
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = round(float(value), 6)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = str(value)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def set_status(self, status: str) -> None:
        assert status in ("running", "finished", "failed"), f"Unknown status: {status}"
        self.status = status
        if status in ("finished", "failed"):
            self.end_time = datetime.now(timezone.utc)
            self.duration_seconds = round(
                (self.end_time - self.start_time).total_seconds(), 3
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ExperimentSummary:
    name: str
    run_count: int
    best_run_id: str | None
    best_metric_value: float | None
    metric_name: str | None
    runs: list[Run]


# ── Tracker ───────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """In-memory + optional JSON persistence experiment tracker.

    Compatible with MLflow concept (experiments → runs → metrics/params).
    Does NOT require a running MLflow server — stores to local JSON.
    """

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._runs: dict[str, Run] = {}  # run_id → Run
        self._storage_path = Path(storage_path) if storage_path else None
        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    def create_run(
        self,
        experiment_name: str,
        tags: dict[str, str] | None = None,
    ) -> Run:
        """Create and register a new run."""
        run = Run(
            run_id=str(uuid.uuid4()),
            experiment_name=experiment_name,
            status="running",
            params={},
            metrics={},
            tags=tags or {},
            artifacts=[],
            start_time=datetime.now(timezone.utc),
        )
        self._runs[run.run_id] = run
        logger.info("experiment.run_created", run_id=run.run_id, experiment=experiment_name)
        return run

    @contextmanager
    def start_run(
        self,
        experiment_name: str,
        tags: dict[str, str] | None = None,
    ) -> Generator[Run, None, None]:
        """Context manager for a run — auto-sets status on exit."""
        run = self.create_run(experiment_name, tags=tags)
        try:
            yield run
            if run.status == "running":
                run.set_status("finished")
        except Exception:
            run.set_status("failed")
            raise
        finally:
            if self._storage_path:
                self._persist_run(run)

    def finish_run(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        if run:
            run.set_status("finished")
            if self._storage_path:
                self._persist_run(run)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self, experiment_name: str) -> list[Run]:
        return [r for r in self._runs.values() if r.experiment_name == experiment_name]

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        higher_is_better: bool = True,
    ) -> Run | None:
        """Return the run with the best value for the given metric."""
        runs = [
            r for r in self.list_runs(experiment_name)
            if metric in r.metrics and r.status == "finished"
        ]
        if not runs:
            return None
        return max(runs, key=lambda r: r.metrics[metric]) if higher_is_better else min(
            runs, key=lambda r: r.metrics[metric]
        )

    def summarise(self, experiment_name: str, metric: str | None = None) -> ExperimentSummary:
        runs = self.list_runs(experiment_name)
        best_run = self.get_best_run(experiment_name, metric) if metric else None
        return ExperimentSummary(
            name=experiment_name,
            run_count=len(runs),
            best_run_id=best_run.run_id if best_run else None,
            best_metric_value=best_run.metrics.get(metric) if best_run and metric else None,
            metric_name=metric,
            runs=runs,
        )

    def all_experiments(self) -> list[str]:
        return list({r.experiment_name for r in self._runs.values()})

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_run(self, run: Run) -> None:
        if not self._storage_path:
            return
        path = self._storage_path / f"{run.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2)

    def _load_from_disk(self) -> None:
        if not self._storage_path:
            return
        for p in self._storage_path.glob("*.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                run = Run(
                    run_id=data["run_id"],
                    experiment_name=data["experiment_name"],
                    status=data["status"],
                    params=data["params"],
                    metrics=data["metrics"],
                    tags=data["tags"],
                    artifacts=data.get("artifacts", []),
                    start_time=datetime.fromisoformat(data["start_time"]),
                    end_time=(
                        datetime.fromisoformat(data["end_time"])
                        if data.get("end_time")
                        else None
                    ),
                    duration_seconds=data.get("duration_seconds"),
                )
                self._runs[run.run_id] = run
            except Exception as exc:
                logger.warning("experiment.load_failed", path=str(p), error=str(exc))


# ── Singleton ─────────────────────────────────────────────────────────────────

_tracker: ExperimentTracker | None = None


def get_tracker() -> ExperimentTracker:
    """Return the global experiment tracker (lazy init)."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker
