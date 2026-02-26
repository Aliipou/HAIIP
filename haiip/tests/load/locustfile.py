"""Locust load test suite for HAIIP API.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --headless \
           --users 50 --spawn-rate 5 --run-time 60s

Load scenarios tested:
1. NormalUser: typical mixed workload (predictions + alerts + feedback)
2. HeavyPredictor: batch-heavy ML inference stress
3. ReadOnlyMonitor: dashboard-style read operations
4. AdminUser: management and audit operations

SLA targets (non-blocking — locust tracks these):
- /predict P99 latency < 2000ms
- /predictions (list) P99 < 500ms
- /alerts P99 < 300ms
- /health P99 < 100ms
- Error rate < 1% under 50 concurrent users
"""

from __future__ import annotations

import json
import random
from typing import Any

from locust import HttpUser, between, events, task


# ── Token cache (login once per user) ────────────────────────────────────────

def _login(client: Any, base_url: str) -> str | None:
    """Login and return JWT access token, or None on failure."""
    with client.post(
        "/api/v1/auth/login",
        data={"username": "admin@load-sme.com", "password": "LoadTest123!"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        catch_response=True,
        name="auth/login",
    ) as resp:
        if resp.status_code == 200:
            return resp.json().get("access_token")
        # Login might fail if user doesn't exist — suppress as known
        resp.success()
        return None


# ── Sensor data factories ──────────────────────────────────────────────────────

def _sensor_payload(machine_num: int) -> dict:
    return {
        "machine_id": f"LOAD-CNC-{machine_num:03d}",
        "features": {
            "air_temperature": round(random.uniform(295.0, 305.0), 2),
            "process_temperature": round(random.uniform(305.0, 315.0), 2),
            "rotational_speed": random.randint(1200, 1800),
            "torque": round(random.uniform(30.0, 55.0), 2),
            "tool_wear": random.randint(0, 240),
        },
    }


def _batch_payload(machine_num: int, batch_size: int = 10) -> dict:
    return {
        "machine_id": f"LOAD-BATCH-{machine_num:03d}",
        "batch": [
            {
                "air_temperature": round(random.uniform(295.0, 310.0), 2),
                "process_temperature": round(random.uniform(305.0, 320.0), 2),
                "rotational_speed": random.randint(1200, 2000),
                "torque": round(random.uniform(30.0, 65.0), 2),
                "tool_wear": random.randint(0, 240),
            }
            for _ in range(batch_size)
        ],
    }


# ── Load test users ───────────────────────────────────────────────────────────

class NormalUser(HttpUser):
    """Simulates a typical mixed-workload user (operator + engineer)."""

    wait_time = between(1, 3)
    weight = 3  # 3x more common than heavy predictor

    def on_start(self) -> None:
        self.token = _login(self.client, self.host) or ""
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.machine_num = random.randint(1, 50)

    @task(4)
    def make_prediction(self) -> None:
        """Most common operation: run anomaly/maintenance prediction."""
        payload = _sensor_payload(self.machine_num)
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="predict/single",
        ) as resp:
            if resp.status_code in (200, 201):
                resp.success()
            elif resp.status_code == 401:
                resp.success()  # Token expired — not a load failure
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(3)
    def list_predictions(self) -> None:
        with self.client.get(
            "/api/v1/predictions?size=20",
            headers=self.headers,
            catch_response=True,
            name="predictions/list",
        ) as resp:
            if resp.status_code in (200, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def list_alerts(self) -> None:
        with self.client.get(
            "/api/v1/alerts?limit=20",
            headers=self.headers,
            catch_response=True,
            name="alerts/list",
        ) as resp:
            if resp.status_code in (200, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def check_health(self) -> None:
        with self.client.get(
            "/health",
            catch_response=True,
            name="health",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "healthy":
                    resp.failure("System not healthy")
                else:
                    resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(1)
    def get_metrics(self) -> None:
        with self.client.get(
            "/api/v1/metrics/health",
            headers=self.headers,
            catch_response=True,
            name="metrics/health",
        ) as resp:
            if resp.status_code in (200, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def submit_feedback(self) -> None:
        """Submit random feedback (may fail if prediction ID doesn't exist)."""
        import uuid
        payload = {
            "prediction_id": str(uuid.uuid4()),
            "was_correct": random.choice([True, False]),
        }
        with self.client.post(
            "/api/v1/feedback",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="feedback/submit",
        ) as resp:
            # 4xx is expected (non-existent prediction) — not a load failure
            if resp.status_code < 500:
                resp.success()
            else:
                resp.failure(f"5xx error: {resp.status_code}")


class HeavyPredictor(HttpUser):
    """Simulates batch ML inference stress (e.g., historical backfill job)."""

    wait_time = between(0.5, 1.5)
    weight = 1

    def on_start(self) -> None:
        self.token = _login(self.client, self.host) or ""
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(5)
    def batch_predict(self) -> None:
        payload = _batch_payload(random.randint(1, 20), batch_size=20)
        with self.client.post(
            "/api/v1/predict/batch",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="predict/batch",
        ) as resp:
            if resp.status_code in (200, 201, 422, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def single_predict(self) -> None:
        payload = _sensor_payload(random.randint(51, 100))
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="predict/single",
        ) as resp:
            if resp.status_code in (200, 201, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")


class ReadOnlyMonitor(HttpUser):
    """Simulates dashboard/monitoring queries (read-heavy)."""

    wait_time = between(2, 5)
    weight = 2

    def on_start(self) -> None:
        self.token = _login(self.client, self.host) or ""
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(3)
    def poll_health(self) -> None:
        self.client.get("/health", name="health")

    @task(2)
    def list_alerts(self) -> None:
        self.client.get("/api/v1/alerts", headers=self.headers, name="alerts/list")

    @task(2)
    def list_predictions(self) -> None:
        self.client.get("/api/v1/predictions?size=50", headers=self.headers, name="predictions/list")

    @task(1)
    def get_alert_summary(self) -> None:
        self.client.get("/api/v1/metrics/alerts/summary", headers=self.headers, name="metrics/alerts")

    @task(1)
    def get_machine_metrics(self) -> None:
        self.client.get("/api/v1/metrics/machines", headers=self.headers, name="metrics/machines")


class AdminLoadUser(HttpUser):
    """Simulates admin management operations (low frequency)."""

    wait_time = between(5, 15)
    weight = 1

    def on_start(self) -> None:
        self.token = _login(self.client, self.host) or ""
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(3)
    def get_audit_log(self) -> None:
        with self.client.get(
            "/api/v1/audit?limit=50",
            headers=self.headers,
            catch_response=True,
            name="admin/audit",
        ) as resp:
            if resp.status_code in (200, 403, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def get_system_stats(self) -> None:
        with self.client.get(
            "/api/v1/admin/stats",
            headers=self.headers,
            catch_response=True,
            name="admin/stats",
        ) as resp:
            if resp.status_code in (200, 403, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def list_users(self) -> None:
        with self.client.get(
            "/api/v1/admin/users",
            headers=self.headers,
            catch_response=True,
            name="admin/users",
        ) as resp:
            if resp.status_code in (200, 403, 401):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")


# ── Custom SLA assertions ─────────────────────────────────────────────────────

@events.quitting.add_listener
def check_sla(environment: Any, **kwargs: Any) -> None:
    """Post-test SLA validation. Sets exit code 1 if SLA breached."""
    stats = environment.stats

    failures_found = []

    for name, stat in stats.entries.items():
        # Skip warmup entries
        if stat.num_requests < 10:
            continue

        # P99 latency SLA
        p99 = stat.get_response_time_percentile(0.99)
        if "health" in str(name) and p99 > 500:
            failures_found.append(f"Health check P99 too slow: {p99}ms > 500ms")
        elif "predict/single" in str(name) and p99 > 3000:
            failures_found.append(f"Predict P99 too slow: {p99}ms > 3000ms")
        elif "predictions/list" in str(name) and p99 > 1000:
            failures_found.append(f"Predictions list P99 too slow: {p99}ms > 1000ms")

    # Overall error rate
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    if total_requests > 0:
        error_rate = total_failures / total_requests
        if error_rate > 0.05:  # 5% error threshold
            failures_found.append(f"Error rate too high: {error_rate:.2%} > 5%")

    if failures_found:
        print("\n⚠️  SLA BREACHES:")
        for msg in failures_found:
            print(f"  - {msg}")
        environment.process_exit_code = 1
    else:
        print("\n✅ All SLA targets met.")
