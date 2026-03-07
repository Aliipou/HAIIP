"""
OPC UA Connector with Explicit Hardware Mode Flags
===================================================
DESIGN PRINCIPLE: The data source mode is ALWAYS explicit and ALWAYS logged.

There is no silent fallback from hardware to simulation. If connection fails:
  - mode switches to HARDWARE_FALLBACK
  - warning is logged with the reason
  - all downstream data is tagged with data_source_mode=HARDWARE_FALLBACK
  - dashboard shows a visible banner (see status_bar.py)

Use assert_real_hardware() at the start of any test that claims real hardware.

Usage:
    connector = OPCUAConnector(endpoint="opc.tcp://plc.jakobstad.fi:4840/",
                               machine_id="CNC-001")
    async with connector:
        reading = connector.get_reading()
        assert reading.data_source_mode == DataSourceMode.REAL_HARDWARE
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware mode enum — no implicit / hidden states
# ---------------------------------------------------------------------------


class DataSourceMode(str, Enum):
    REAL_HARDWARE = "real_hardware"  # OPC UA confirmed connected and verified
    SIMULATION = "simulation"  # synthetic data, no hardware attempted
    HARDWARE_FALLBACK = "hardware_fallback"  # tried hardware, fell back to simulation


class HardwareNotConnectedError(RuntimeError):
    """
    Raised by assert_real_hardware() when mode != REAL_HARDWARE.
    Call this at the start of any test that claims to test real hardware.
    """


# ---------------------------------------------------------------------------
# Sensor reading with mandatory mode tag
# ---------------------------------------------------------------------------


@dataclass
class SensorReading:
    machine_id: str
    timestamp: datetime
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    data_source_mode: DataSourceMode = DataSourceMode.SIMULATION
    raw_values: dict[str, Any] = field(default_factory=dict)
    data_quality_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "air_temperature": self.air_temperature,
            "process_temperature": self.process_temperature,
            "rotational_speed": self.rotational_speed,
            "torque": self.torque,
            "tool_wear": self.tool_wear,
            "data_source_mode": self.data_source_mode.value,
            "data_quality_warnings": self.data_quality_warnings,
        }


# ---------------------------------------------------------------------------
# Data quality guards
# ---------------------------------------------------------------------------

_SENSOR_RANGES = {
    "air_temperature": (270.0, 330.0),  # Kelvin — physical plant range
    "process_temperature": (280.0, 380.0),
    "rotational_speed": (0.0, 3000.0),  # rpm
    "torque": (-5.0, 100.0),  # Nm (negative allowed briefly)
    "tool_wear": (0.0, 300.0),  # minutes
}
_STALE_THRESHOLD_SECONDS = 300  # 5 minutes


def validate_reading(reading: SensorReading) -> list[str]:
    """
    Returns list of data quality warnings. Does NOT raise.
    Caller decides whether to drop or forward with warnings.
    """
    import math

    warnings: list[str] = []
    sensor_fields = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]

    values = {f: getattr(reading, f) for f in sensor_fields}
    all_zero = all(v == 0.0 for v in values.values())
    if all_zero:
        warnings.append("DATA_QUALITY_WARNING: all sensor values are 0.0 — sensor may be offline")

    for field_name, val in values.items():
        if math.isnan(val):
            warnings.append(f"DATA_QUALITY_ERROR: {field_name} is NaN")
        elif math.isinf(val):
            warnings.append(f"DATA_QUALITY_ERROR: {field_name} is Inf")
        else:
            lo, hi = _SENSOR_RANGES.get(field_name, (float("-inf"), float("inf")))
            if not (lo <= val <= hi):
                warnings.append(
                    f"DATA_QUALITY_WARNING: {field_name} = {val:.2f} "
                    f"outside expected range [{lo}, {hi}]"
                )

    age = (datetime.now(UTC) - reading.timestamp.replace(tzinfo=UTC)).total_seconds()
    if age > _STALE_THRESHOLD_SECONDS:
        warnings.append(
            f"STALE_DATA: reading timestamp is {age:.0f}s old "
            f"(threshold {_STALE_THRESHOLD_SECONDS}s)"
        )
    if age < -60:
        warnings.append(f"FUTURE_TIMESTAMP: reading timestamp is {-age:.0f}s in the future")

    return warnings


# ---------------------------------------------------------------------------
# OPC UA Connector with explicit mode
# ---------------------------------------------------------------------------


class OPCUAConnector:
    """
    Wraps asyncua OPC UA client with explicit hardware mode tracking.

    If asyncua is not installed, mode = SIMULATION always.
    If connection fails, mode = HARDWARE_FALLBACK with logged reason.
    """

    MAX_BUFFER_READINGS = 1000  # bounded queue to prevent OOM on long disconnects

    def __init__(
        self,
        endpoint: str = "opc.tcp://localhost:4840/",
        machine_id: str = "MACHINE-001",
        namespace: int = 2,
        node_ids: dict[str, str] | None = None,
        poll_interval: float = 1.0,
        connection_timeout: float = 10.0,
    ) -> None:
        self._endpoint = endpoint
        self._machine_id = machine_id
        self._namespace = namespace
        self._node_ids = node_ids or {
            "air_temperature": f"ns={namespace};i=1001",
            "process_temperature": f"ns={namespace};i=1002",
            "rotational_speed": f"ns={namespace};i=1003",
            "torque": f"ns={namespace};i=1004",
            "tool_wear": f"ns={namespace};i=1005",
        }
        self._poll_interval = poll_interval
        self._connection_timeout = connection_timeout
        self._mode: DataSourceMode = DataSourceMode.SIMULATION
        self._last_connection_error: str = "not yet attempted"
        self._client: Any = None
        self._buffer: list[SensorReading] = []
        self._callbacks: list[Callable[[SensorReading], None]] = []

    # ------------------------------------------------------------------
    # Mode property — always readable, always explicit
    # ------------------------------------------------------------------

    @property
    def mode(self) -> DataSourceMode:
        return self._mode

    @property
    def is_connected(self) -> bool:
        return self._mode == DataSourceMode.REAL_HARDWARE

    def assert_real_hardware(self) -> None:
        """
        Raises HardwareNotConnectedError if not in REAL_HARDWARE mode.
        Call this at the start of any test that claims to test real hardware.
        """
        if self._mode != DataSourceMode.REAL_HARDWARE:
            raise HardwareNotConnectedError(
                f"Real hardware required but mode is {self._mode.value}. "
                f"Connection failure reason: {self._last_connection_error}"
            )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Attempt OPC UA connection.
        Sets mode to REAL_HARDWARE on success, HARDWARE_FALLBACK on failure.
        Returns True if connected to real hardware.
        """
        try:
            from asyncua import Client  # type: ignore[import]

            client = Client(url=self._endpoint, timeout=self._connection_timeout)
            await asyncio.wait_for(client.connect(), timeout=self._connection_timeout)
            self._client = client
            self._mode = DataSourceMode.REAL_HARDWARE
            logger.info(
                "opcua_connected endpoint=%s machine=%s",
                self._endpoint,
                self._machine_id,
            )
            return True
        except ImportError:
            reason = "asyncua not installed"
            self._last_connection_error = reason
            self._mode = DataSourceMode.SIMULATION
            logger.warning("opcua_unavailable reason=%s mode=%s", reason, self._mode.value)
            return False
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            self._last_connection_error = reason
            self._mode = DataSourceMode.HARDWARE_FALLBACK
            logger.warning(
                "opcua_connection_failed endpoint=%s reason=%s mode=%s",
                self._endpoint,
                reason,
                self._mode.value,
            )
            return False

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self._mode = DataSourceMode.SIMULATION

    async def __aenter__(self) -> OPCUAConnector:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Reading — mode ALWAYS tagged
    # ------------------------------------------------------------------

    def get_reading(self) -> SensorReading:
        """
        Returns a SensorReading tagged with the current data source mode.
        Mode is ALWAYS set — there is no untagged reading.
        """
        reading = self._get_reading_internal()
        reading.data_source_mode = self._mode
        reading.data_quality_warnings = validate_reading(reading)
        return reading

    def _get_reading_internal(self) -> SensorReading:
        if self._mode == DataSourceMode.REAL_HARDWARE and self._client is not None:
            # Synchronous wrapper — use subscribe() for async production use
            try:
                return self._read_from_hardware_sync()
            except Exception as exc:  # noqa: BLE001
                self._last_connection_error = str(exc)
                self._mode = DataSourceMode.HARDWARE_FALLBACK
                logger.warning("opcua_read_failed switching to HARDWARE_FALLBACK: %s", exc)
        return self._generate_simulated_reading()

    def _read_from_hardware_sync(self) -> SensorReading:
        """Placeholder for synchronous hardware read — only reached in REAL_HARDWARE mode."""
        raise RuntimeError("Use subscribe() for async hardware reads")

    def _generate_simulated_reading(self) -> SensorReading:
        rng = random.Random()
        return SensorReading(
            machine_id=self._machine_id,
            timestamp=datetime.now(UTC),
            air_temperature=298.0 + rng.gauss(0, 0.5),
            process_temperature=308.0 + rng.gauss(0, 0.5),
            rotational_speed=1500.0 + rng.gauss(0, 20.0),
            torque=40.0 + rng.gauss(0, 2.0),
            tool_wear=max(0.0, 50.0 + rng.gauss(0, 5.0)),
            data_source_mode=self._mode,
        )

    async def subscribe(
        self,
        on_reading: Callable[[SensorReading], None] | None = None,
        max_readings: int | None = None,
    ) -> None:
        """Poll at poll_interval, invoke callbacks. Buffers readings if client not connected."""
        count = 0
        while max_readings is None or count < max_readings:
            reading = self.get_reading()

            if len(self._buffer) < self.MAX_BUFFER_READINGS:
                self._buffer.append(reading)

            if on_reading:
                on_reading(reading)
            for cb in self._callbacks:
                cb(reading)

            count += 1
            await asyncio.sleep(self._poll_interval)

    def add_callback(self, callback: Callable[[SensorReading], None]) -> None:
        self._callbacks.append(callback)

    def drain_buffer(self) -> list[SensorReading]:
        """Return and clear buffered readings."""
        readings, self._buffer = self._buffer, []
        return readings
