"""Circuit breaker pattern for external service resilience.

Prevents cascading failures when downstream services (DB, Redis, LLM API,
OPC UA) become unavailable. Implements the standard three-state FSM:

  CLOSED   → normal operation, failures counted
  OPEN     → fast-fail all calls, no downstream load
  HALF_OPEN → probe: one call allowed; success → CLOSED, failure → OPEN

References:
    Martin Fowler: circuitbreaker (2014)
    Netflix Hystrix design principles
    IEC 61508 SIL-2 fault containment requirements

Usage:
    cb = CircuitBreaker(name="db", failure_threshold=5, recovery_timeout=30)

    @cb.call
    def get_sensor_reading():
        return db.query(...)

    # Or as context manager:
    with cb:
        result = external_api.call()
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when circuit is OPEN and call is rejected (fast-fail)."""


@dataclass
class CircuitBreakerStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # fast-failed while OPEN
    state_transitions: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0


class CircuitBreaker:
    """Thread-safe circuit breaker with configurable thresholds.

    Args:
        name: identifier for logging
        failure_threshold: consecutive failures before opening
        recovery_timeout: seconds to wait before attempting HALF_OPEN
        success_threshold: consecutive successes in HALF_OPEN before closing
        expected_exceptions: tuple of exception types that trip the breaker
                             (default: all exceptions)
        on_state_change: optional callback(name, old_state, new_state)
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.expected_exceptions = expected_exceptions
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
        self._stats = CircuitBreakerStats()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    def call(self, func: F) -> F:
        """Decorator — wrap a function with circuit breaker protection."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._execute(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    def __call__(self, func: F) -> F:
        return self.call(func)

    def __enter__(self) -> CircuitBreaker:
        self._before_call()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None and issubclass(exc_type, self.expected_exceptions):
            self._on_failure()
            return False  # re-raise
        if exc_type is None:
            self._on_success()
        return False

    def reset(self) -> None:
        """Manually force circuit to CLOSED state (for testing / admin recovery)."""
        with self._lock:
            old = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            if old != CircuitState.CLOSED:
                self._notify(old, CircuitState.CLOSED)

    # ── Internal FSM ──────────────────────────────────────────────────────────

    def _execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._before_call()
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions:
            self._on_failure()
            raise

    def _before_call(self) -> None:
        with self._lock:
            self._stats.total_calls += 1

            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    # Try recovery
                    old = self._state
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    self._stats.state_transitions += 1
                    self._notify(old, CircuitState.HALF_OPEN)
                    logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN (probing)", self.name)
                else:
                    self._stats.rejected_calls += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit '{self.name}' is OPEN — "
                        f"retry in {self.recovery_timeout - (time.monotonic() - self._last_failure_time):.1f}s"
                    )

    def _on_success(self) -> None:
        with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    old = self._state
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._stats.state_transitions += 1
                    self._notify(old, CircuitState.CLOSED)
                    logger.info("CircuitBreaker[%s]: HALF_OPEN → CLOSED", self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # reset on success

    def _on_failure(self) -> None:
        with self._lock:
            self._stats.failed_calls += 1
            self._last_failure_time = time.monotonic()
            self._stats.last_failure_time = self._last_failure_time
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — back to OPEN
                old = self._state
                self._state = CircuitState.OPEN
                self._stats.state_transitions += 1
                self._notify(old, CircuitState.OPEN)
                logger.warning("CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed)", self.name)

            elif (
                self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold
            ):
                old = self._state
                self._state = CircuitState.OPEN
                self._stats.state_transitions += 1
                self._notify(old, CircuitState.OPEN)
                logger.error(
                    "CircuitBreaker[%s]: CLOSED → OPEN (failures=%d >= threshold=%d)",
                    self.name,
                    self._failure_count,
                    self.failure_threshold,
                )

    def _notify(self, old: CircuitState, new: CircuitState) -> None:
        if self._on_state_change:
            try:
                self._on_state_change(self.name, old, new)
            except Exception as exc:  # noqa: BLE001
                logger.debug("CircuitBreaker state-change callback failed: %s", exc)


class CircuitBreakerRegistry:
    """Global registry of named circuit breakers — singleton per process."""

    _instance: CircuitBreakerRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    @classmethod
    def get(cls) -> CircuitBreakerRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """Get or create a named circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs,
            )
        return self._breakers[name]

    def get_breaker(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def reset_all(self) -> None:
        for cb in self._breakers.values():
            cb.reset()

    def status(self) -> dict[str, str]:
        return {name: cb.state.value for name, cb in self._breakers.items()}


# Pre-wired breakers for HAIIP services
def get_db_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get().register("db", failure_threshold=3, recovery_timeout=10.0)


def get_redis_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get().register(
        "redis", failure_threshold=5, recovery_timeout=15.0
    )


def get_llm_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get().register("llm", failure_threshold=3, recovery_timeout=60.0)


def get_opcua_breaker() -> CircuitBreaker:
    return CircuitBreakerRegistry.get().register(
        "opcua", failure_threshold=5, recovery_timeout=30.0
    )
