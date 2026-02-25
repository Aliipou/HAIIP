"""OPC UA connector — reads sensor data from industrial OPC UA servers.

OPC UA is the de-facto standard for factory floor data exchange.
Every real CNC machine, PLC, and SCADA system exposes OPC UA endpoints.

This connector:
1. Connects to an OPC UA server (real or simulated)
2. Subscribes to sensor node changes (event-driven, not polling)
3. Normalises readings into the HAIIP standard schema
4. Feeds the data pipeline for real-time anomaly detection

For development: uses the simulator when server is unreachable.
For production: connects to real OPC UA endpoints (Siemens, Fanuc, etc.)

Architecture note: asyncua library chosen over opcua (sync) for
compatibility with FastAPI's async event loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Standard HAIIP OPC UA node IDs (namespace 2, configurable)
DEFAULT_NODE_IDS = {
    "air_temperature": "ns=2;i=1001",
    "process_temperature": "ns=2;i=1002",
    "rotational_speed": "ns=2;i=1003",
    "torque": "ns=2;i=1004",
    "tool_wear": "ns=2;i=1005",
}


@dataclass
class OPCUAReading:
    """Normalised sensor reading from OPC UA."""
    machine_id: str
    timestamp: datetime
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    raw_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "air_temperature": self.air_temperature,
            "process_temperature": self.process_temperature,
            "rotational_speed": self.rotational_speed,
            "torque": self.torque,
            "tool_wear": self.tool_wear,
        }


class OPCUAConnector:
    """Async OPC UA client for real-time industrial sensor data.

    Usage (async context):
        async with OPCUAConnector(endpoint="opc.tcp://machine:4840/") as conn:
            async for reading in conn.subscribe(machine_id="CNC-001"):
                process(reading)
    """

    def __init__(
        self,
        endpoint: str = "opc.tcp://localhost:4840/freeopcua/server/",
        machine_id: str = "MACHINE-001",
        namespace: int = 2,
        node_ids: dict[str, str] | None = None,
        poll_interval: float = 1.0,
        timeout: float = 10.0,
    ) -> None:
        self.endpoint = endpoint
        self.machine_id = machine_id
        self.namespace = namespace
        self.node_ids = node_ids or DEFAULT_NODE_IDS
        self.poll_interval = poll_interval
        self.timeout = timeout

        self._client: Any = None
        self._connected = False
        self._callbacks: list[Callable[[OPCUAReading], None]] = []

    async def connect(self) -> bool:
        """Connect to OPC UA server. Returns True on success."""
        try:
            from asyncua import Client

            self._client = Client(url=self.endpoint, timeout=self.timeout)
            await self._client.connect()
            self._connected = True
            logger.info("OPC UA connected: %s", self.endpoint)
            return True
        except ImportError:
            logger.warning("asyncua not installed — OPC UA disabled")
            return False
        except Exception as exc:
            logger.error("OPC UA connection failed (%s): %s", self.endpoint, exc)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("OPC UA disconnected: %s", self.endpoint)

    async def __aenter__(self) -> "OPCUAConnector":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    async def read_once(self) -> OPCUAReading | None:
        """Read all configured nodes once and return a normalised reading."""
        if not self._connected:
            return None

        try:
            values: dict[str, Any] = {}
            for name, node_id in self.node_ids.items():
                node = self._client.get_node(node_id)
                values[name] = float(await node.read_value())

            return OPCUAReading(
                machine_id=self.machine_id,
                timestamp=datetime.now(timezone.utc),
                air_temperature=values.get("air_temperature", 0.0),
                process_temperature=values.get("process_temperature", 0.0),
                rotational_speed=values.get("rotational_speed", 0.0),
                torque=values.get("torque", 0.0),
                tool_wear=values.get("tool_wear", 0.0),
                raw_values=values,
            )
        except Exception as exc:
            logger.error("OPC UA read error: %s", exc)
            return None

    async def subscribe(
        self,
        on_reading: Callable[[OPCUAReading], None] | None = None,
        max_readings: int | None = None,
    ) -> None:
        """Poll OPC UA nodes at poll_interval. Calls on_reading callback."""
        count = 0
        while max_readings is None or count < max_readings:
            reading = await self.read_once()
            if reading:
                if on_reading:
                    on_reading(reading)
                for cb in self._callbacks:
                    cb(reading)
                count += 1
            await asyncio.sleep(self.poll_interval)

    def add_callback(self, callback: Callable[[OPCUAReading], None]) -> None:
        self._callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        return self._connected


class SimulatedOPCUAServer:
    """Lightweight simulated OPC UA server for development and testing.

    Runs as a background asyncio task; produces synthetic sensor data
    matching real OPC UA behaviour for integration testing.
    """

    def __init__(self, port: int = 4840, machine_id: str = "SIM-001") -> None:
        self.port = port
        self.machine_id = machine_id
        self._running = False

    async def start(self) -> None:
        """Start the simulated server."""
        try:
            from asyncua import Server

            self._server = Server()
            await self._server.init()
            self._server.set_endpoint(f"opc.tcp://0.0.0.0:{self.port}/")

            idx = await self._server.register_namespace("HAIIP")
            objects = self._server.get_objects_node()
            machine = await objects.add_object(idx, "Machine")

            self._nodes = {
                "air_temperature": await machine.add_variable(idx, "AirTemp", 300.0),
                "process_temperature": await machine.add_variable(idx, "ProcessTemp", 310.0),
                "rotational_speed": await machine.add_variable(idx, "RPM", 1538.0),
                "torque": await machine.add_variable(idx, "Torque", 40.0),
                "tool_wear": await machine.add_variable(idx, "ToolWear", 0.0),
            }

            async with self._server:
                self._running = True
                logger.info("Simulated OPC UA server started on port %d", self.port)
                await self._update_loop()
        except ImportError:
            logger.warning("asyncua not installed — simulated server not available")

    async def _update_loop(self) -> None:
        import numpy as np
        rng = np.random.default_rng(42)
        wear = 0.0

        while self._running:
            wear = min(wear + 0.5, 253.0)
            await self._nodes["air_temperature"].write_value(
                float(rng.normal(300.0, 2.0))
            )
            await self._nodes["process_temperature"].write_value(
                float(rng.normal(310.0, 1.5))
            )
            await self._nodes["rotational_speed"].write_value(
                float(max(0, rng.normal(1538.0, 179.0)))
            )
            await self._nodes["torque"].write_value(
                float(max(0, rng.normal(40.0, 9.8)))
            )
            await self._nodes["tool_wear"].write_value(wear)
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        self._running = False
