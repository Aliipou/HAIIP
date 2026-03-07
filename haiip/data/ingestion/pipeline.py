"""Data normalisation and ingestion pipeline.

Bridges raw data sources (OPC UA, MQTT, simulator) to the AI core.

Pipeline stages:
1. Receive raw reading from any source
2. Validate and normalise to standard schema
3. Apply unit conversion if needed
4. Route to anomaly detector (sync) and alert generator
5. Store prediction result via API or directly to DB

This module is source-agnostic: same pipeline handles OPC UA, MQTT, CSV.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from haiip.core.anomaly import AnomalyDetector
from haiip.core.drift import DriftDetector
from haiip.core.feedback import FeedbackEngine

logger = logging.getLogger(__name__)

# Standard feature order expected by all HAIIP models
STANDARD_FEATURE_ORDER = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]


@dataclass
class NormalisedReading:
    """Source-agnostic, validated sensor reading."""

    machine_id: str
    tenant_id: str
    timestamp: datetime
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_vector(self) -> list[float]:
        return [
            self.air_temperature,
            self.process_temperature,
            self.rotational_speed,
            self.torque,
            self.tool_wear,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "air_temperature": self.air_temperature,
            "process_temperature": self.process_temperature,
            "rotational_speed": self.rotational_speed,
            "torque": self.torque,
            "tool_wear": self.tool_wear,
            "source": self.source,
        }


@dataclass
class PipelineResult:
    reading: NormalisedReading
    anomaly_result: dict[str, Any]
    drift_detected: bool
    processing_time_ms: float
    alert_triggered: bool = False
    alert_severity: str | None = None


class IngestionPipeline:
    """Real-time sensor data ingestion and processing pipeline.

    Thread-safe for reads. Configure once, then call process() from any source.

    Usage:
        pipeline = IngestionPipeline(tenant_id="sme-001")
        pipeline.set_anomaly_detector(trained_detector)
        result = pipeline.process(normalised_reading)
    """

    def __init__(
        self,
        tenant_id: str = "default",
        anomaly_threshold: float = 0.7,
        alert_callback: Callable[[PipelineResult], None] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.anomaly_threshold = anomaly_threshold
        self.alert_callback = alert_callback

        self._anomaly_detector: AnomalyDetector | None = None
        self._drift_detector: DriftDetector | None = None
        self._feedback_engine = FeedbackEngine()
        self._processing_count = 0
        self._alert_count = 0

    def set_anomaly_detector(self, detector: AnomalyDetector) -> None:
        self._anomaly_detector = detector

    def set_drift_detector(self, detector: DriftDetector) -> None:
        self._drift_detector = detector

    def process(self, reading: NormalisedReading) -> PipelineResult:
        """Process a single normalised reading through the AI pipeline."""
        t0 = time.monotonic()

        # Step 1: Validate
        reading = self._validate(reading)

        # Step 2: Run anomaly detection
        if self._anomaly_detector:
            anomaly_result = self._anomaly_detector.predict(reading.feature_vector)
        else:
            anomaly_result = {
                "label": "normal",
                "confidence": 0.5,
                "anomaly_score": 0.0,
                "explanation": {},
            }

        # Step 3: Check concept drift (streaming)
        drift_detected = False
        if self._drift_detector:
            drift_flags = self._drift_detector.check_stream(reading.feature_vector)
            drift_detected = any(drift_flags.values())

        # Step 4: Determine alert
        alert_triggered = (
            anomaly_result.get("label") == "anomaly"
            and anomaly_result.get("confidence", 0.0) >= self.anomaly_threshold
        )

        severity = None
        if alert_triggered:
            score = anomaly_result.get("anomaly_score", 0.0)
            severity = "critical" if score > 0.9 else "high" if score > 0.7 else "medium"
            self._alert_count += 1

        processing_ms = (time.monotonic() - t0) * 1000
        self._processing_count += 1

        result = PipelineResult(
            reading=reading,
            anomaly_result=anomaly_result,
            drift_detected=drift_detected,
            processing_time_ms=round(processing_ms, 3),
            alert_triggered=alert_triggered,
            alert_severity=severity,
        )

        if alert_triggered and self.alert_callback:
            try:
                self.alert_callback(result)
            except Exception as exc:
                logger.error("Alert callback failed: %s", exc)

        logger.debug(
            "pipeline.processed machine=%s label=%s confidence=%.3f drift=%s alert=%s",
            reading.machine_id,
            anomaly_result.get("label"),
            anomaly_result.get("confidence", 0.0),
            drift_detected,
            alert_triggered,
        )

        return result

    @staticmethod
    def _validate(reading: NormalisedReading) -> NormalisedReading:
        """Clamp values to physically plausible ranges."""
        reading.air_temperature = max(-50.0, min(500.0, reading.air_temperature))
        reading.process_temperature = max(-50.0, min(1000.0, reading.process_temperature))
        reading.rotational_speed = max(0.0, min(100_000.0, reading.rotational_speed))
        reading.torque = max(0.0, min(10_000.0, reading.torque))
        reading.tool_wear = max(0.0, min(1000.0, reading.tool_wear))
        return reading

    def normalise_from_opcua(self, opcua_reading: Any) -> NormalisedReading:
        """Convert OPCUAReading to NormalisedReading."""
        return NormalisedReading(
            machine_id=opcua_reading.machine_id,
            tenant_id=self.tenant_id,
            timestamp=opcua_reading.timestamp,
            air_temperature=opcua_reading.air_temperature,
            process_temperature=opcua_reading.process_temperature,
            rotational_speed=opcua_reading.rotational_speed,
            torque=opcua_reading.torque,
            tool_wear=opcua_reading.tool_wear,
            source="opcua",
        )

    def normalise_from_mqtt_buffer(
        self, machine_id: str, buffer: dict[str, float]
    ) -> NormalisedReading | None:
        """Convert buffered MQTT readings into one NormalisedReading.

        Buffer must contain all 5 required sensors.
        Returns None if any sensor is missing.
        """
        required = set(STANDARD_FEATURE_ORDER)
        if not required.issubset(buffer.keys()):
            missing = required - buffer.keys()
            logger.debug("MQTT buffer incomplete for %s: missing %s", machine_id, missing)
            return None

        return NormalisedReading(
            machine_id=machine_id,
            tenant_id=self.tenant_id,
            timestamp=datetime.now(UTC),
            air_temperature=buffer["air_temperature"],
            process_temperature=buffer["process_temperature"],
            rotational_speed=buffer["rotational_speed"],
            torque=buffer["torque"],
            tool_wear=buffer["tool_wear"],
            source="mqtt",
        )

    def normalise_from_simulator(self, sim_reading: dict[str, Any]) -> NormalisedReading:
        """Convert simulator dict to NormalisedReading."""
        return NormalisedReading(
            machine_id=sim_reading["machine_id"],
            tenant_id=self.tenant_id,
            timestamp=datetime.fromtimestamp(sim_reading["timestamp"], tz=UTC),
            air_temperature=sim_reading["air_temperature"],
            process_temperature=sim_reading["process_temperature"],
            rotational_speed=sim_reading["rotational_speed"],
            torque=sim_reading["torque"],
            tool_wear=sim_reading["tool_wear"],
            source="simulator",
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "processing_count": self._processing_count,
            "alert_count": self._alert_count,
            "alert_rate": (
                round(self._alert_count / self._processing_count, 4)
                if self._processing_count > 0
                else 0.0
            ),
        }
