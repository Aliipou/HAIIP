"""Kafka producer — publishes sensor readings and predictions.

Graceful degradation: if confluent-kafka is not installed, logs a warning
and no-ops all publish calls (system continues without streaming).

Usage:
    producer = SensorProducer(bootstrap_servers="localhost:9092")
    producer.publish_sensor(msg)
    producer.flush()
"""

from __future__ import annotations

import json
import logging
from typing import Any

from haiip.streaming.schema import (
    TOPIC_ALERTS,
    TOPIC_ECONOMIC,
    TOPIC_PREDICTIONS,
    TOPIC_SENSORS_RAW,
    AlertMessage,
    PredictionMessage,
    SensorMessage,
)

logger = logging.getLogger(__name__)

# sentinel — set once at module level
_confluent_available: bool | None = None


def _kafka_available() -> bool:
    global _confluent_available
    if _confluent_available is None:
        try:
            import confluent_kafka  # noqa: F401
            _confluent_available = True
        except ImportError:
            _confluent_available = False
            logger.warning(
                "confluent-kafka not installed — Kafka publishing disabled. "
                "Install with: pip install confluent-kafka"
            )
    return _confluent_available


class SensorProducer:
    """Kafka producer for HAIIP sensor and prediction events.

    Thread-safe: confluent_kafka.Producer is thread-safe for produce().
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        extra_config: dict[str, Any] | None = None,
        on_delivery: Any = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self._producer: Any = None
        self._on_delivery = on_delivery or self._default_delivery_report

        if _kafka_available():
            from confluent_kafka import Producer

            config: dict[str, Any] = {
                "bootstrap.servers": bootstrap_servers,
                "acks": "all",                 # wait for all ISRs
                "retries": 5,
                "retry.backoff.ms": 200,
                "compression.type": "lz4",
                "linger.ms": 5,               # micro-batching
                "batch.size": 65536,
            }
            if extra_config:
                config.update(extra_config)

            self._producer = Producer(config)
            logger.info("Kafka producer connected to %s", bootstrap_servers)

    # ── Publish helpers ───────────────────────────────────────────────────────

    def publish_sensor(self, msg: SensorMessage) -> None:
        """Publish a sensor reading to haiip.sensors.raw."""
        self._publish(TOPIC_SENSORS_RAW, msg.machine_id, msg.to_dict())

    def publish_prediction(self, msg: PredictionMessage) -> None:
        """Publish a prediction to haiip.predictions."""
        self._publish(TOPIC_PREDICTIONS, msg.machine_id, msg.to_dict())

    def publish_alert(self, msg: AlertMessage) -> None:
        """Publish an alert to haiip.alerts."""
        self._publish(TOPIC_ALERTS, msg.machine_id, msg.to_dict())

    def publish_economic(self, machine_id: str, payload: dict[str, Any]) -> None:
        """Publish an economic decision to haiip.economic."""
        self._publish(TOPIC_ECONOMIC, machine_id, payload)

    def flush(self, timeout: float = 10.0) -> int:
        """Flush pending messages. Returns number of messages still in queue."""
        if self._producer is None:
            return 0
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("Kafka flush: %d messages still in queue after %.1fs", remaining, timeout)
        return remaining

    def poll(self, timeout: float = 0.0) -> int:
        """Poll for delivery reports. Returns number of events processed."""
        if self._producer is None:
            return 0
        return self._producer.poll(timeout=timeout)

    def close(self) -> None:
        """Flush and close the producer."""
        self.flush()
        logger.info("SensorProducer closed")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _publish(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        if self._producer is None:
            logger.debug("Kafka disabled — dropping message to %s key=%s", topic, key)
            return
        try:
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=json.dumps(payload, default=str).encode("utf-8"),
                on_delivery=self._on_delivery,
            )
            self._producer.poll(0)  # trigger callbacks
        except Exception as exc:  # noqa: BLE001
            logger.error("Kafka produce failed: topic=%s key=%s error=%s", topic, key, exc)

    @staticmethod
    def _default_delivery_report(err: Any, msg: Any) -> None:
        if err is not None:
            logger.error("Kafka delivery failed: %s", err)
        else:
            logger.debug(
                "Kafka delivered: topic=%s partition=%d offset=%d",
                msg.topic(), msg.partition(), msg.offset(),
            )
