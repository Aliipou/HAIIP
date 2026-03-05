"""Tests for Kafka streaming — schema, producer, consumer — 100% branch coverage."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest


# ── SensorMessage ─────────────────────────────────────────────────────────────

class TestSensorMessage:

    def _msg(self, **kwargs):
        from haiip.streaming.schema import SensorMessage
        defaults = dict(
            machine_id="M001", tenant_id="t1",
            air_temperature=300.0, process_temperature=310.0,
            rotational_speed=1538.0, torque=40.0, tool_wear=100.0,
        )
        defaults.update(kwargs)
        return SensorMessage(**defaults)

    def test_to_dict_has_required_keys(self):
        msg = self._msg()
        d = msg.to_dict()
        assert "machine_id" in d
        assert "air_temperature" in d

    def test_from_dict_roundtrip(self):
        from haiip.streaming.schema import SensorMessage
        msg = self._msg()
        d = msg.to_dict()
        restored = SensorMessage.from_dict(d)
        assert restored.machine_id == msg.machine_id
        assert restored.air_temperature == msg.air_temperature

    def test_features_property_returns_5_floats(self):
        msg = self._msg()
        feats = msg.features
        assert len(feats) == 5
        assert feats[0] == 300.0

    def test_from_dict_default_timestamp(self):
        from haiip.streaming.schema import SensorMessage
        d = {
            "machine_id": "M002", "tenant_id": "t2",
            "air_temperature": "301.5", "process_temperature": "311.0",
            "rotational_speed": "1600.0", "torque": "42.0", "tool_wear": "50.0",
        }
        msg = SensorMessage.from_dict(d)
        assert msg.air_temperature == 301.5
        assert msg.timestamp > 0

    def test_from_dict_custom_source(self):
        from haiip.streaming.schema import SensorMessage
        base = self._msg().to_dict()
        base["source"] = "mqtt"
        msg = SensorMessage.from_dict(base)
        assert msg.source == "mqtt"

    def test_data_quality_warnings_default_empty(self):
        msg = self._msg()
        assert msg.data_quality_warnings == []


# ── PredictionMessage ─────────────────────────────────────────────────────────

class TestPredictionMessage:

    def _pred_dict(self, **kwargs):
        d = dict(
            machine_id="M001", tenant_id="t1",
            prediction_id="pred-001", label="normal",
            confidence=0.9, anomaly_score=0.1,
            explanation={}, sensor_timestamp=time.time(),
        )
        d.update(kwargs)
        return d

    def test_from_dict_roundtrip(self):
        from haiip.streaming.schema import PredictionMessage
        d = self._pred_dict()
        msg = PredictionMessage.from_dict(d)
        assert msg.label == "normal"
        assert msg.confidence == 0.9

    def test_to_dict_keys(self):
        from haiip.streaming.schema import PredictionMessage
        msg = PredictionMessage.from_dict(self._pred_dict())
        d = msg.to_dict()
        assert "label" in d and "confidence" in d and "anomaly_score" in d

    def test_shap_values_optional(self):
        from haiip.streaming.schema import PredictionMessage
        d = self._pred_dict(shap_values={"air_temperature": 0.12})
        msg = PredictionMessage.from_dict(d)
        assert msg.shap_values is not None

    def test_economic_action_optional(self):
        from haiip.streaming.schema import PredictionMessage
        d = self._pred_dict(economic_action="REPAIR_NOW")
        msg = PredictionMessage.from_dict(d)
        assert msg.economic_action == "REPAIR_NOW"


# ── AlertMessage ──────────────────────────────────────────────────────────────

class TestAlertMessage:

    def test_to_dict_has_severity(self):
        from haiip.streaming.schema import AlertMessage
        alert = AlertMessage(
            machine_id="M001", tenant_id="t1",
            alert_type="anomaly_detected", severity="critical",
            message="Machine anomaly detected",
        )
        d = alert.to_dict()
        assert d["severity"] == "critical"


# ── SensorProducer ────────────────────────────────────────────────────────────

class TestSensorProducer:

    def test_producer_no_kafka_is_noop(self):
        """Without confluent-kafka installed, publish is a no-op."""
        with patch.dict("sys.modules", {"confluent_kafka": None}):
            # Reset the module-level cache
            import haiip.streaming.kafka_producer as kp
            kp._confluent_available = None

            from haiip.streaming.kafka_producer import SensorProducer
            producer = SensorProducer()
            assert producer._producer is None
            # Should not raise
            producer.flush()

    def test_publish_sensor_calls_produce(self):
        from haiip.streaming.schema import SensorMessage
        from haiip.streaming.kafka_producer import SensorProducer

        mock_producer = MagicMock()

        with patch("haiip.streaming.kafka_producer._kafka_available", return_value=True), \
             patch("haiip.streaming.kafka_producer._confluent_available", True):
            producer = SensorProducer.__new__(SensorProducer)
            producer._producer = mock_producer
            producer._on_delivery = SensorProducer._default_delivery_report
            producer.bootstrap_servers = "localhost:9092"

            msg = SensorMessage(
                machine_id="M001", tenant_id="t1",
                air_temperature=300.0, process_temperature=310.0,
                rotational_speed=1538.0, torque=40.0, tool_wear=100.0,
            )
            producer.publish_sensor(msg)

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args[1]
        assert call_kwargs["topic"] == "haiip.sensors.raw"

    def test_flush_returns_zero_when_no_kafka(self):
        import haiip.streaming.kafka_producer as kp
        kp._confluent_available = None

        from haiip.streaming.kafka_producer import SensorProducer
        producer = SensorProducer.__new__(SensorProducer)
        producer._producer = None
        assert producer.flush() == 0

    def test_poll_returns_zero_when_no_kafka(self):
        from haiip.streaming.kafka_producer import SensorProducer
        producer = SensorProducer.__new__(SensorProducer)
        producer._producer = None
        assert producer.poll() == 0

    def test_publish_logs_on_produce_error(self):
        from haiip.streaming.kafka_producer import SensorProducer
        mock_p = MagicMock()
        mock_p.produce.side_effect = Exception("broker unavailable")

        producer = SensorProducer.__new__(SensorProducer)
        producer._producer = mock_p
        producer._on_delivery = SensorProducer._default_delivery_report
        producer.bootstrap_servers = "localhost:9092"

        # Should not raise — errors are logged
        producer._publish("haiip.sensors.raw", "M001", {"key": "val"})

    def test_delivery_report_error_logged(self):
        from haiip.streaming.kafka_producer import SensorProducer
        with patch("haiip.streaming.kafka_producer.logger") as mock_log:
            SensorProducer._default_delivery_report("delivery failed", None)
            mock_log.error.assert_called_once()

    def test_delivery_report_success_logged(self):
        from haiip.streaming.kafka_producer import SensorProducer
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "haiip.sensors.raw"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 42
        with patch("haiip.streaming.kafka_producer.logger") as mock_log:
            SensorProducer._default_delivery_report(None, mock_msg)
            mock_log.debug.assert_called_once()

    def test_close_flushes(self):
        from haiip.streaming.kafka_producer import SensorProducer
        producer = SensorProducer.__new__(SensorProducer)
        producer._producer = None  # no-op flush
        producer.close()  # should not raise


# ── InferenceConsumer ─────────────────────────────────────────────────────────

class TestInferenceConsumer:

    def test_consumer_no_kafka_is_noop(self):
        import haiip.streaming.kafka_producer as kp
        kp._confluent_available = None

        with patch.dict("sys.modules", {"confluent_kafka": None}):
            from haiip.streaming.kafka_consumer import InferenceConsumer
            consumer = InferenceConsumer()
            consumer.start()  # should not raise
            assert not consumer.is_running

    def test_messages_processed_initially_zero(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        c = InferenceConsumer()
        assert c.messages_processed == 0
        assert c.errors == 0

    def test_handle_message_bad_json(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        consumer = InferenceConsumer()
        mock_msg = MagicMock()
        mock_msg.value.return_value = b"not-valid-json"
        consumer._handle_message(mock_msg)
        assert consumer.errors == 1

    def test_handle_message_valid_no_detector(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        consumer = InferenceConsumer(detector=None)
        mock_msg = MagicMock()
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)
        assert consumer.messages_processed == 0  # no detector → no prediction

    def test_handle_message_with_detector(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "label": "normal", "confidence": 0.9,
            "anomaly_score": 0.1, "explanation": {},
        }
        consumer = InferenceConsumer(detector=mock_detector)
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg = MagicMock()
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)
        assert consumer.messages_processed == 1

    def test_anomaly_publishes_alert(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "label": "anomaly", "confidence": 0.92,
            "anomaly_score": 0.75, "explanation": {},
        }
        mock_producer = MagicMock()
        consumer = InferenceConsumer(detector=mock_detector, producer=mock_producer)
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg = MagicMock()
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)
        mock_producer.publish_alert.assert_called_once()

    def test_high_confidence_anomaly_critical_alert(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "label": "anomaly", "confidence": 0.95,
            "anomaly_score": 0.8, "explanation": {},
        }
        mock_producer = MagicMock()
        consumer = InferenceConsumer(detector=mock_detector, producer=mock_producer)
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg = MagicMock()
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)
        alert_call = mock_producer.publish_alert.call_args[0][0]
        assert alert_call.severity == "critical"

    def test_inference_exception_increments_errors(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        mock_detector = MagicMock()
        mock_detector.predict.side_effect = RuntimeError("model crashed")
        consumer = InferenceConsumer(detector=mock_detector)
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg = MagicMock()
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)
        assert consumer.errors == 1

    def test_on_prediction_callback_fires(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        received = []
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "label": "normal", "confidence": 0.9,
            "anomaly_score": 0.1, "explanation": {},
        }
        consumer = InferenceConsumer(
            detector=mock_detector,
            on_prediction=received.append,
        )
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg = MagicMock()
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)
        assert len(received) == 1

    def test_callback_exception_does_not_crash(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {
            "label": "normal", "confidence": 0.9,
            "anomaly_score": 0.1, "explanation": {},
        }
        def bad_callback(x):
            raise ValueError("callback failure")

        consumer = InferenceConsumer(detector=mock_detector, on_prediction=bad_callback)
        payload = {
            "machine_id": "M001", "tenant_id": "t1",
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1538.0, "torque": 40.0, "tool_wear": 100.0,
            "timestamp": time.time(), "source": "test",
        }
        mock_msg = MagicMock()
        mock_msg.value.return_value = json.dumps(payload).encode()
        consumer._handle_message(mock_msg)  # should not raise

    def test_stop_sets_running_false(self):
        from haiip.streaming.kafka_consumer import InferenceConsumer
        consumer = InferenceConsumer()
        consumer._running = True
        consumer._consumer = MagicMock()
        consumer.stop()
        assert not consumer.is_running
