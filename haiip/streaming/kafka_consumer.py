"""Kafka consumer — reads sensor messages and runs anomaly inference.

Designed for the HAIIP worker process. Consumes haiip.sensors.raw,
runs AnomalyDetector.predict(), and publishes back to haiip.predictions.

Usage:
    consumer = InferenceConsumer(
        bootstrap_servers="localhost:9092",
        group_id="haiip-worker",
        detector=fitted_detector,
        producer=sensor_producer,
    )
    consumer.start()   # blocks; call consumer.stop() from another thread
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

from haiip.streaming.schema import (
    TOPIC_SENSORS_RAW,
    AlertMessage,
    PredictionMessage,
    SensorMessage,
)

logger = logging.getLogger(__name__)


class InferenceConsumer:
    """Kafka consumer that runs anomaly detection on each sensor message.

    Thread model: runs in its own daemon thread; gracefully shuts down on stop().
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "haiip-worker",
        detector: Any = None,
        producer: Any = None,
        extra_config: dict[str, Any] | None = None,
        poll_timeout: float = 1.0,
        on_prediction: Callable[[PredictionMessage], None] | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._detector = detector
        self._producer = producer
        self._poll_timeout = poll_timeout
        self._on_prediction = on_prediction
        self._consumer: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._messages_processed = 0
        self._errors = 0
        self._extra_config = extra_config or {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, blocking: bool = True) -> None:
        """Start consuming. If blocking=False, runs in background thread."""
        try:
            from confluent_kafka import Consumer, KafkaError  # noqa: F401
        except ImportError:
            logger.warning("confluent-kafka not installed — InferenceConsumer disabled")
            return

        config: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 300000,
        }
        config.update(self._extra_config)

        self._consumer = Consumer(config)
        self._consumer.subscribe([TOPIC_SENSORS_RAW])
        self._running = True

        logger.info(
            "InferenceConsumer started: brokers=%s group=%s",
            self.bootstrap_servers,
            self.group_id,
        )

        if blocking:
            self._consume_loop()
        else:
            self._thread = threading.Thread(
                target=self._consume_loop, daemon=True, name="haiip-consumer"
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the consumer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._consumer:
            self._consumer.close()
            logger.info(
                "InferenceConsumer closed (processed=%d errors=%d)",
                self._messages_processed,
                self._errors,
            )

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _consume_loop(self) -> None:
        from confluent_kafka import KafkaError

        while self._running:
            msg = self._consumer.poll(timeout=self._poll_timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error: %s", msg.error())
                self._errors += 1
                continue

            self._handle_message(msg)

    def _handle_message(self, msg: Any) -> None:
        try:
            payload = json.loads(msg.value().decode("utf-8"))
            sensor_msg = SensorMessage.from_dict(payload)
        except Exception as exc:
            logger.warning("Failed to decode sensor message: %s", exc)
            self._errors += 1
            return

        # Run inference
        prediction = self._run_inference(sensor_msg)
        if prediction is None:
            return

        self._messages_processed += 1

        # Callback hook (for testing / dashboard updates)
        if self._on_prediction:
            try:
                self._on_prediction(prediction)
            except Exception as exc:  # noqa: BLE001
                logger.debug("on_prediction callback failed: %s", exc)

        # Publish result
        if self._producer is not None:
            self._producer.publish_prediction(prediction)

        # Publish alert if anomaly
        if prediction.label == "anomaly" and self._producer is not None:
            alert = AlertMessage(
                machine_id=sensor_msg.machine_id,
                tenant_id=sensor_msg.tenant_id,
                alert_type="anomaly_detected",
                severity="warning" if prediction.confidence < 0.85 else "critical",
                message=(
                    f"Anomaly detected on {sensor_msg.machine_id}: "
                    f"score={prediction.anomaly_score:.3f} "
                    f"confidence={prediction.confidence:.3f}"
                ),
                prediction_id=prediction.prediction_id,
            )
            self._producer.publish_alert(alert)

    def _run_inference(self, msg: SensorMessage) -> PredictionMessage | None:
        if self._detector is None:
            logger.debug("No detector configured — skipping inference")
            return None
        try:
            result = self._detector.predict(msg.features)
            return PredictionMessage(
                machine_id=msg.machine_id,
                tenant_id=msg.tenant_id,
                prediction_id=str(uuid.uuid4()),
                label=result["label"],
                confidence=result["confidence"],
                anomaly_score=result["anomaly_score"],
                explanation=result.get("explanation", {}),
                sensor_timestamp=msg.timestamp,
                shap_values=result.get("shap_values"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference failed for %s: %s", msg.machine_id, exc)
            self._errors += 1
            return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def messages_processed(self) -> int:
        return self._messages_processed

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def is_running(self) -> bool:
        return self._running
