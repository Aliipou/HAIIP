"""JSON schemas and dataclasses for Kafka streaming messages.

Topics:
  haiip.sensors.raw      — raw sensor readings from OPC UA / MQTT
  haiip.predictions      — anomaly + maintenance predictions
  haiip.alerts           — triggered alert events
  haiip.economic         — economic AI decisions
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SensorMessage:
    """Kafka message on topic haiip.sensors.raw."""

    machine_id: str
    tenant_id: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    timestamp: float = field(default_factory=time.time)
    source: str = "opcua"  # opcua | mqtt | manual | simulation
    data_quality_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SensorMessage:
        return cls(
            machine_id=d["machine_id"],
            tenant_id=d["tenant_id"],
            air_temperature=float(d["air_temperature"]),
            process_temperature=float(d["process_temperature"]),
            rotational_speed=float(d["rotational_speed"]),
            torque=float(d["torque"]),
            tool_wear=float(d["tool_wear"]),
            timestamp=float(d.get("timestamp", time.time())),
            source=d.get("source", "unknown"),
            data_quality_warnings=d.get("data_quality_warnings", []),
        )

    @property
    def features(self) -> list[float]:
        return [
            self.air_temperature,
            self.process_temperature,
            self.rotational_speed,
            self.torque,
            self.tool_wear,
        ]


@dataclass
class PredictionMessage:
    """Kafka message on topic haiip.predictions."""

    machine_id: str
    tenant_id: str
    prediction_id: str
    label: str  # normal | anomaly
    confidence: float
    anomaly_score: float
    explanation: dict[str, Any]
    sensor_timestamp: float
    prediction_timestamp: float = field(default_factory=time.time)
    shap_values: dict[str, float] | None = None
    economic_action: str | None = None  # REPAIR_NOW | SCHEDULE | MONITOR | IGNORE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PredictionMessage:
        return cls(
            machine_id=d["machine_id"],
            tenant_id=d["tenant_id"],
            prediction_id=d["prediction_id"],
            label=d["label"],
            confidence=float(d["confidence"]),
            anomaly_score=float(d["anomaly_score"]),
            explanation=d.get("explanation", {}),
            sensor_timestamp=float(d["sensor_timestamp"]),
            prediction_timestamp=float(d.get("prediction_timestamp", time.time())),
            shap_values=d.get("shap_values"),
            economic_action=d.get("economic_action"),
        )


@dataclass
class AlertMessage:
    """Kafka message on topic haiip.alerts."""

    machine_id: str
    tenant_id: str
    alert_type: str  # anomaly_detected | high_anomaly_rate | drift_detected
    severity: str  # info | warning | critical
    message: str
    prediction_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Kafka topic names
TOPIC_SENSORS_RAW = "haiip.sensors.raw"
TOPIC_PREDICTIONS = "haiip.predictions"
TOPIC_ALERTS = "haiip.alerts"
TOPIC_ECONOMIC = "haiip.economic"

# Default Kafka config (override with env vars in production)
DEFAULT_KAFKA_CONFIG = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "haiip",
}
