"""Phase 6 load tests — Economic AI + Agent + Federated endpoints.

Run: locust -f haiip/tests/load/locustfile_phase6.py --headless -u 50 -r 5 -t 60s

SLA targets (from Model Card):
    - /api/v1/economic/decide    P95 < 100ms
    - /api/v1/agent/query        P95 < 3000ms
    - /api/v1/agent/diagnose     P95 < 5000ms
    - /health                    P95 < 50ms
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# ── Shared payload generators ──────────────────────────────────────────────────

MACHINES = [f"M-{i:03d}" for i in range(1, 11)]
TENANTS = ["acme_fi", "nordic_se", "offshore_no"]


def sensor_payload() -> dict:
    return {
        "anomaly_score": round(random.uniform(0, 1), 3),
        "failure_probability": round(random.uniform(0, 1), 3),
        "rul_cycles": random.randint(50, 5000),
        "confidence": round(random.uniform(0.5, 1.0), 3),
        "machine_id": random.choice(MACHINES),
    }


def agent_query_payload() -> dict:
    queries = [
        "What is the current health status of the conveyor belt?",
        "When should I schedule maintenance for pump M-003?",
        "Is machine M-007 showing signs of bearing failure?",
        "Explain the anomaly detected on spindle M-001 at 14:32.",
        "What does the EU AI Act require for this risk level?",
    ]
    return {
        "query": random.choice(queries),
        "machine_id": random.choice(MACHINES),
    }


# ── User classes ───────────────────────────────────────────────────────────────


class OperatorUser(HttpUser):
    """Simulates a factory floor operator checking machine health."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        resp = self.client.post(
            "/api/v1/auth/token",
            data={
                "username": "operator@acme.fi",
                "password": "TestPass123!",
            },
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token", "")
        else:
            self.token = ""

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")

    @task(10)
    def economic_decide(self) -> None:
        self.client.post(
            "/api/v1/economic/decide",
            json=sensor_payload(),
            headers=self.headers,
            name="/api/v1/economic/decide",
        )

    @task(3)
    def agent_diagnose(self) -> None:
        payload = agent_query_payload()
        payload["sensor_readings"] = {
            "vibration": round(random.uniform(0, 10), 2),
            "temperature": round(random.uniform(20, 90), 1),
        }
        self.client.post(
            "/api/v1/agent/diagnose",
            json=payload,
            headers=self.headers,
            name="/api/v1/agent/diagnose",
        )


class EngineerUser(HttpUser):
    """Simulates a maintenance engineer running queries and reviewing models."""

    wait_time = between(2, 5)

    def on_start(self) -> None:
        resp = self.client.post(
            "/api/v1/auth/token",
            data={
                "username": "engineer@acme.fi",
                "password": "TestPass123!",
            },
        )
        self.token = resp.json().get("access_token", "") if resp.status_code == 200 else ""

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def agent_query(self) -> None:
        self.client.post(
            "/api/v1/agent/query",
            json=agent_query_payload(),
            headers=self.headers,
            name="/api/v1/agent/query",
        )

    @task(3)
    def batch_economic_decide(self) -> None:
        records = [sensor_payload() for _ in range(5)]
        self.client.post(
            "/api/v1/economic/batch",
            json={"records": records},
            headers=self.headers,
            name="/api/v1/economic/batch",
        )

    @task(2)
    def agent_capabilities(self) -> None:
        self.client.get(
            "/api/v1/agent/capabilities",
            headers=self.headers,
            name="/api/v1/agent/capabilities",
        )


class AdminUser(HttpUser):
    """Simulates an admin reviewing compliance + oversight metrics."""

    wait_time = between(5, 15)

    def on_start(self) -> None:
        resp = self.client.post(
            "/api/v1/auth/token",
            data={
                "username": "admin@acme.fi",
                "password": "AdminPass456!",
            },
        )
        self.token = resp.json().get("access_token", "") if resp.status_code == 200 else ""

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def get_tenant_info(self) -> None:
        self.client.get(
            "/api/v1/admin/tenant",
            headers=self.headers,
            name="/api/v1/admin/tenant",
        )

    @task(2)
    def get_audit_log(self) -> None:
        self.client.get(
            "/api/v1/admin/audit-log?limit=20",
            headers=self.headers,
            name="/api/v1/admin/audit-log",
        )


# ── SLA enforcement ────────────────────────────────────────────────────────────

SLA_MS = {
    "/health": 50,
    "/api/v1/economic/decide": 100,
    "/api/v1/agent/query": 3000,
    "/api/v1/agent/diagnose": 5000,
    "/api/v1/economic/batch": 500,
}


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response,  # noqa: ANN001
    context,  # noqa: ANN001
    exception,  # noqa: ANN001
    **kwargs: object,
) -> None:
    threshold = SLA_MS.get(name)
    if threshold and response_time > threshold:
        print(f"[SLA BREACH] {name}: {response_time:.0f}ms > {threshold}ms threshold")
