"""OpenTelemetry tracing for HAIIP.

Instruments every prediction, agent query, and API request with:
    - Distributed traces (OTel spans)
    - Structured span attributes (machine_id, tenant_id, prediction confidence)
    - Automatic error capture (span status = ERROR)
    - SLA enforcement (latency threshold check per span)

When OTEL_EXPORTER_OTLP_ENDPOINT is not set, falls back to console exporter
so the system still works in dev/offline mode.

References:
    - OpenTelemetry Python SDK 1.x
    - OpenTelemetry Semantic Conventions for ML (draft 2024)
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Lightweight OTel wrapper (no hard dependency) ──────────────────────────────

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False


class _NoopSpan:
    """Null-object span used when OTel is unavailable."""
    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_exception(self, exc: Exception) -> None: ...
    def set_status(self, *args: Any, **kwargs: Any) -> None: ...
    def __enter__(self) -> "_NoopSpan": return self
    def __exit__(self, *args: Any) -> None: ...


class HAIIPTracer:
    """Thin wrapper around OpenTelemetry tracer.

    Falls back gracefully to no-ops when OTel SDK is not installed.
    All instrumentation points go through this class so the business logic
    stays clean.

    Usage::
        tracer = HAIIPTracer(service_name="haiip-api")

        with tracer.span("predict", {"machine_id": "M001", "tenant": "acme"}):
            result = detector.predict(X)

        # Or as decorator:
        @tracer.instrument("anomaly_detect")
        def predict(X):
            ...
    """

    # SLA thresholds (ms) — from HAIIP Model Card
    SLA_THRESHOLDS: dict[str, float] = {
        "predict":          200.0,
        "agent_query":     2000.0,
        "rag_query":       1500.0,
        "federated_round": 5000.0,
        "default":          500.0,
    }

    def __init__(
        self,
        service_name: str = "haiip",
        otlp_endpoint: str | None = None,
    ) -> None:
        self.service_name = service_name
        self._tracer: Any = None
        self._init_tracer(otlp_endpoint)

    def _init_tracer(self, endpoint: str | None) -> None:
        if not _OTEL_AVAILABLE:
            logger.warning("opentelemetry not installed — using noop tracer")
            return

        provider = TracerProvider()
        endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=endpoint)
            except ImportError:
                logger.warning("OTLP exporter not installed — falling back to console")
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self.service_name)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        check_sla: bool = True,
    ) -> Generator[Any, None, None]:
        """Context manager that wraps a code block in an OTel span.

        Also checks SLA threshold and logs a warning if exceeded.
        """
        t0 = time.perf_counter()

        if self._tracer is None:
            noop = _NoopSpan()
            yield noop
        else:
            with self._tracer.start_as_current_span(name) as sp:
                if attributes:
                    for k, v in attributes.items():
                        sp.set_attribute(k, str(v))
                try:
                    yield sp
                except Exception as exc:
                    sp.record_exception(exc)
                    if _OTEL_AVAILABLE:
                        from opentelemetry.trace import StatusCode
                        sp.set_status(StatusCode.ERROR, str(exc))
                    raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        threshold  = self.SLA_THRESHOLDS.get(name, self.SLA_THRESHOLDS["default"])
        if check_sla and elapsed_ms > threshold:
            logger.warning(
                "sla_breach",
                extra={"span": name, "elapsed_ms": round(elapsed_ms, 1),
                       "threshold_ms": threshold},
            )

    def instrument(self, span_name: str) -> Callable[[F], F]:
        """Decorator version of span()."""
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name):
                    return fn(*args, **kwargs)
            return wrapper  # type: ignore[return-value]
        return decorator


# ── Singleton ──────────────────────────────────────────────────────────────────

_default_tracer: HAIIPTracer | None = None


def get_tracer() -> HAIIPTracer:
    """Return the process-wide default tracer (lazy init)."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = HAIIPTracer()
    return _default_tracer
