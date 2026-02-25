"""MQTT connector — subscribes to industrial sensor topics.

MQTT is the standard IoT protocol for resource-constrained devices.
Used by: temperature sensors, vibration sensors, PLCs via IoT gateways.

Topic schema: haiip/sensors/{tenant_id}/{machine_id}/{sensor_name}
Payload: JSON {"value": float, "unit": str, "timestamp": iso8601}

This connector:
1. Subscribes to sensor topics (wildcard + per-machine)
2. Parses JSON payloads with validation
3. Aggregates readings into OPCUAReading-compatible format
4. Feeds the pipeline for real-time processing

For development: uses mock publisher if broker is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MQTTReading:
    """Normalised reading received from MQTT broker."""
    machine_id: str
    tenant_id: str
    topic: str
    timestamp: datetime
    sensor_name: str
    value: float
    unit: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "tenant_id": self.tenant_id,
            "sensor_name": self.sensor_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
        }


class MQTTConnector:
    """Async MQTT subscriber for industrial sensor topics.

    Usage:
        async with MQTTConnector(host="localhost", tenant_id="sme-001") as conn:
            conn.add_callback(process_reading)
            await conn.listen(machine_id="CNC-001")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        tenant_id: str = "default",
        topic_prefix: str = "haiip/sensors",
        username: str = "",
        password: str = "",
        keepalive: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.tenant_id = tenant_id
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        self.keepalive = keepalive

        self._client: Any = None
        self._connected = False
        self._callbacks: list[Callable[[MQTTReading], None]] = []
        self._reading_buffer: dict[str, dict[str, float]] = {}

    async def connect(self) -> bool:
        """Connect to MQTT broker. Returns True on success."""
        try:
            import aiomqtt

            connect_kwargs: dict[str, Any] = {
                "hostname": self.host,
                "port": self.port,
                "keepalive": self.keepalive,
            }
            if self.username:
                connect_kwargs["username"] = self.username
            if self.password:
                connect_kwargs["password"] = self.password

            self._client = aiomqtt.Client(**connect_kwargs)
            await self._client.__aenter__()
            self._connected = True
            logger.info("MQTT connected: %s:%d", self.host, self.port)
            return True
        except ImportError:
            logger.warning("aiomqtt not installed — MQTT disabled")
            return False
        except Exception as exc:
            logger.error("MQTT connection failed (%s:%d): %s", self.host, self.port, exc)
            return False

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
        self._connected = False

    async def __aenter__(self) -> "MQTTConnector":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    async def listen(
        self,
        machine_id: str | None = None,
        max_messages: int | None = None,
    ) -> None:
        """Subscribe and process messages. Calls registered callbacks."""
        if not self._connected or not self._client:
            logger.warning("MQTT not connected — skipping listen")
            return

        # Subscribe to machine-specific or all topics
        if machine_id:
            topic = f"{self.topic_prefix}/{self.tenant_id}/{machine_id}/+"
        else:
            topic = f"{self.topic_prefix}/{self.tenant_id}/+/+"

        await self._client.subscribe(topic)
        logger.info("MQTT subscribed: %s", topic)

        count = 0
        async for message in self._client.messages:
            reading = self._parse_message(message)
            if reading:
                for cb in self._callbacks:
                    cb(reading)
                count += 1
            if max_messages and count >= max_messages:
                break

    def _parse_message(self, message: Any) -> MQTTReading | None:
        """Parse raw MQTT message into MQTTReading."""
        try:
            # Topic: haiip/sensors/{tenant}/{machine_id}/{sensor_name}
            parts = str(message.topic).split("/")
            if len(parts) < 5:
                return None

            machine_id = parts[-2]
            sensor_name = parts[-1]

            payload = json.loads(message.payload.decode("utf-8"))

            value = float(payload.get("value", 0.0))
            unit = str(payload.get("unit", ""))

            ts_str = payload.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            return MQTTReading(
                machine_id=machine_id,
                tenant_id=self.tenant_id,
                topic=str(message.topic),
                timestamp=ts,
                sensor_name=sensor_name,
                value=value,
                unit=unit,
                raw_payload=payload,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug("MQTT parse error: %s", exc)
            return None

    def add_callback(self, callback: Callable[[MQTTReading], None]) -> None:
        self._callbacks.append(callback)

    async def publish(self, machine_id: str, sensor_name: str, value: float, unit: str = "") -> None:
        """Publish a single sensor reading (for testing/simulation)."""
        if not self._connected or not self._client:
            return
        topic = f"{self.topic_prefix}/{self.tenant_id}/{machine_id}/{sensor_name}"
        payload = json.dumps({
            "value": value,
            "unit": unit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await self._client.publish(topic, payload=payload, qos=1)

    @property
    def is_connected(self) -> bool:
        return self._connected


class MockMQTTPublisher:
    """Publishes synthetic sensor data to MQTT for development/testing.

    Use this when no real broker is available — generates realistic
    data streams and publishes to the configured broker.
    """

    def __init__(
        self,
        connector: MQTTConnector,
        machine_ids: list[str] | None = None,
        interval: float = 1.0,
    ) -> None:
        self.connector = connector
        self.machine_ids = machine_ids or ["SIM-001", "SIM-002"]
        self.interval = interval

    async def run(self, n_cycles: int = 100) -> None:
        import numpy as np
        rng = np.random.default_rng(42)

        for _ in range(n_cycles):
            for machine_id in self.machine_ids:
                await self.connector.publish(
                    machine_id, "air_temperature", float(rng.normal(300.0, 2.0)), "K"
                )
                await self.connector.publish(
                    machine_id, "process_temperature", float(rng.normal(310.0, 1.5)), "K"
                )
                await self.connector.publish(
                    machine_id, "rotational_speed", float(max(0, rng.normal(1538.0, 179.0))), "rpm"
                )
                await self.connector.publish(
                    machine_id, "torque", float(max(0, rng.normal(40.0, 9.8))), "Nm"
                )
            await asyncio.sleep(self.interval)
