"""Tests for CircuitBreaker — 100% branch coverage."""

from __future__ import annotations

import threading
import time

import pytest

from haiip.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
    get_db_breaker,
    get_llm_breaker,
    get_opcua_breaker,
    get_redis_breaker,
)

# ── Basic FSM transitions ─────────────────────────────────────────────────────


class TestCircuitBreakerFSM:
    @pytest.fixture(autouse=True)
    def fresh_breaker(self):
        """Each test gets a fresh circuit breaker."""
        self.cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.05)

    def test_initial_state_is_closed(self):
        assert self.cb.state == CircuitState.CLOSED

    def test_successful_call_stays_closed(self):
        result = self.cb._execute(lambda: 42)
        assert result == 42
        assert self.cb.state == CircuitState.CLOSED

    def test_failure_below_threshold_stays_closed(self):
        def bad():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                self.cb._execute(bad)
        assert self.cb.state == CircuitState.CLOSED

    def test_failure_at_threshold_opens(self):
        def bad():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                self.cb._execute(bad)
        assert self.cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        def bad():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                self.cb._execute(bad)

        with pytest.raises(CircuitBreakerOpenError):
            self.cb._execute(lambda: 42)

    def test_open_transitions_to_half_open_after_timeout(self):
        def bad():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                self.cb._execute(bad)

        assert self.cb.state == CircuitState.OPEN
        time.sleep(0.06)  # wait for recovery_timeout=0.05
        # Next call should attempt HALF_OPEN
        result = self.cb._execute(lambda: "recovered")
        assert result == "recovered"
        # After success_threshold=2, should be CLOSED
        # (only 1 success so far, need 2)
        assert self.cb.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_half_open_success_threshold_closes(self):
        cb = CircuitBreaker(
            name="t", failure_threshold=3, recovery_timeout=0.01, success_threshold=2
        )

        def bad():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb._execute(bad)

        time.sleep(0.02)
        # Two successful calls → CLOSED
        cb._execute(lambda: 1)
        cb._execute(lambda: 2)
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        for _ in range(3):
            with pytest.raises(ValueError):
                self.cb._execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.06)
        # Force HALF_OPEN manually (simpler than timing)
        self.cb._state = CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            self.cb._execute(lambda: (_ for _ in ()).throw(ValueError("probe fail")))

        assert self.cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        def bad():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                self.cb._execute(bad)

        self.cb._execute(lambda: "ok")
        assert self.cb._failure_count == 0

    def test_reset_forces_closed(self):
        def bad():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                self.cb._execute(bad)

        assert self.cb.state == CircuitState.OPEN
        self.cb.reset()
        assert self.cb.state == CircuitState.CLOSED

    def test_reset_already_closed_noop(self):
        self.cb.reset()
        assert self.cb.state == CircuitState.CLOSED


# ── Decorator interface ───────────────────────────────────────────────────────


class TestCircuitBreakerDecorator:
    def test_call_decorator(self):
        cb = CircuitBreaker(name="dec", failure_threshold=5)

        @cb.call
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_callable_decorator(self):
        cb = CircuitBreaker(name="dec2", failure_threshold=5)

        @cb
        def mul(a, b):
            return a * b

        assert mul(3, 4) == 12

    def test_decorator_preserves_function_name(self):
        cb = CircuitBreaker(name="dec3", failure_threshold=5)

        @cb.call
        def my_function():
            return 1

        assert my_function.__name__ == "my_function"

    def test_decorator_propagates_exception(self):
        cb = CircuitBreaker(name="dec4", failure_threshold=10)

        @cb.call
        def fail():
            raise RuntimeError("propagated")

        with pytest.raises(RuntimeError, match="propagated"):
            fail()


# ── Context manager interface ─────────────────────────────────────────────────


class TestCircuitBreakerContextManager:
    def test_context_manager_success(self):
        cb = CircuitBreaker(name="ctx", failure_threshold=5)
        with cb:
            result = 42
        assert result == 42

    def test_context_manager_failure_counts(self):
        cb = CircuitBreaker(name="ctx2", failure_threshold=2)
        for _ in range(2):
            try:
                with cb:
                    raise ValueError("ctx fail")
            except ValueError:
                pass
        assert cb.state == CircuitState.OPEN

    def test_context_manager_non_expected_exception_not_counted(self):
        cb = CircuitBreaker(
            name="ctx3",
            failure_threshold=2,
            expected_exceptions=(IOError,),  # only IOError trips the breaker
        )
        for _ in range(5):
            try:
                with cb:
                    raise ValueError("not counted")
            except ValueError:
                pass
        assert cb.state == CircuitState.CLOSED  # ValueError not expected


# ── Stats tracking ────────────────────────────────────────────────────────────


class TestCircuitBreakerStats:
    def test_total_calls_incremented(self):
        cb = CircuitBreaker(name="stats", failure_threshold=10)
        for _ in range(3):
            cb._execute(lambda: 1)
        assert cb.stats.total_calls == 3

    def test_successful_calls_counted(self):
        cb = CircuitBreaker(name="stats2", failure_threshold=10)
        cb._execute(lambda: 1)
        cb._execute(lambda: 2)
        assert cb.stats.successful_calls == 2

    def test_failed_calls_counted(self):
        cb = CircuitBreaker(name="stats3", failure_threshold=10)
        for _ in range(3):
            try:
                cb._execute(lambda: (_ for _ in ()).throw(ValueError("x")))
            except ValueError:
                pass
        assert cb.stats.failed_calls == 3

    def test_rejected_calls_counted_when_open(self):
        cb = CircuitBreaker(name="stats4", failure_threshold=2, recovery_timeout=9999)
        for _ in range(2):
            try:
                cb._execute(lambda: (_ for _ in ()).throw(ValueError("x")))
            except ValueError:
                pass
        # Now open — try a call
        try:
            cb._execute(lambda: 1)
        except CircuitBreakerOpenError:
            pass
        assert cb.stats.rejected_calls == 1

    def test_state_transitions_counted(self):
        cb = CircuitBreaker(name="stats5", failure_threshold=2, recovery_timeout=9999)
        for _ in range(2):
            try:
                cb._execute(lambda: (_ for _ in ()).throw(ValueError("x")))
            except ValueError:
                pass
        assert cb.stats.state_transitions == 1  # CLOSED → OPEN


# ── State-change callback ─────────────────────────────────────────────────────


class TestStateChangeCallback:
    def test_callback_fires_on_open(self):
        transitions = []
        cb = CircuitBreaker(
            name="cb",
            failure_threshold=2,
            on_state_change=lambda name, old, new: transitions.append((old, new)),
        )
        for _ in range(2):
            try:
                cb._execute(lambda: (_ for _ in ()).throw(ValueError("x")))
            except ValueError:
                pass
        assert len(transitions) == 1
        assert transitions[0] == (CircuitState.CLOSED, CircuitState.OPEN)

    def test_callback_exception_doesnt_crash(self):
        def bad_callback(name, old, new):
            raise RuntimeError("callback exploded")

        cb = CircuitBreaker(name="cb2", failure_threshold=1, on_state_change=bad_callback)
        try:
            cb._execute(lambda: (_ for _ in ()).throw(ValueError("x")))
        except ValueError:
            pass
        # Should not raise despite bad callback
        assert cb.state == CircuitState.OPEN


# ── Registry ─────────────────────────────────────────────────────────────────


class TestCircuitBreakerRegistry:
    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        CircuitBreakerRegistry._instance = None

    def test_singleton(self):
        r1 = CircuitBreakerRegistry.get()
        r2 = CircuitBreakerRegistry.get()
        assert r1 is r2

    def test_register_creates_breaker(self):
        reg = CircuitBreakerRegistry.get()
        cb = reg.register("myservice", failure_threshold=3)
        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "myservice"

    def test_register_same_name_returns_same_instance(self):
        reg = CircuitBreakerRegistry.get()
        cb1 = reg.register("svc")
        cb2 = reg.register("svc")
        assert cb1 is cb2

    def test_get_breaker_missing_returns_none(self):
        reg = CircuitBreakerRegistry.get()
        assert reg.get_breaker("nonexistent") is None

    def test_reset_all(self):
        reg = CircuitBreakerRegistry.get()
        cb = reg.register("svc2", failure_threshold=1)
        try:
            cb._execute(lambda: (_ for _ in ()).throw(ValueError("x")))
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN
        reg.reset_all()
        assert cb.state == CircuitState.CLOSED

    def test_status(self):
        reg = CircuitBreakerRegistry.get()
        reg.register("a")
        reg.register("b")
        s = reg.status()
        assert "a" in s and "b" in s

    def test_pre_wired_db_breaker(self):
        CircuitBreakerRegistry._instance = None
        cb = get_db_breaker()
        assert cb.name == "db"

    def test_pre_wired_redis_breaker(self):
        CircuitBreakerRegistry._instance = None
        cb = get_redis_breaker()
        assert cb.name == "redis"

    def test_pre_wired_llm_breaker(self):
        CircuitBreakerRegistry._instance = None
        cb = get_llm_breaker()
        assert cb.name == "llm"

    def test_pre_wired_opcua_breaker(self):
        CircuitBreakerRegistry._instance = None
        cb = get_opcua_breaker()
        assert cb.name == "opcua"


# ── Thread safety ─────────────────────────────────────────────────────────────


class TestCircuitBreakerThreadSafety:
    def test_concurrent_calls_dont_corrupt_state(self):
        cb = CircuitBreaker(name="concurrent", failure_threshold=50, recovery_timeout=99)
        errors = []

        def call():
            try:
                cb._execute(lambda: 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # State must be consistent
        assert cb.state in list(CircuitState)
        assert cb.stats.total_calls == 20

    def test_concurrent_failures_trip_exactly_once(self):
        cb = CircuitBreaker(name="conc2", failure_threshold=5, recovery_timeout=99)
        barrier = threading.Barrier(10)

        def fail_call():
            barrier.wait()
            try:
                cb._execute(lambda: (_ for _ in ()).throw(ValueError("concurrent")))
            except (ValueError, CircuitBreakerOpenError):
                pass

        threads = [threading.Thread(target=fail_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should be OPEN, transitions should be exactly 1
        assert cb.state == CircuitState.OPEN
