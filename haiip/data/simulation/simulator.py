"""Synthetic sensor data simulator — always available, no network required.

Generates realistic industrial sensor streams with configurable fault injection.
Used for:
- Development and testing without real hardware
- Demo mode in the Streamlit dashboard
- Generating training data when real datasets are unavailable
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class SimulatorConfig:
    machine_id: str = "SIM-001"
    seed: int = 42
    # Normal operating ranges (AI4I 2020 distributions)
    air_temp_mean: float = 300.0
    air_temp_std: float = 2.0
    process_temp_delta: float = 10.0      # process_temp = air_temp + delta
    process_temp_std: float = 1.5
    rpm_mean: float = 1538.0
    rpm_std: float = 179.0
    torque_mean: float = 40.0
    torque_std: float = 9.8
    tool_wear_increment: float = 0.5      # minutes per cycle
    tool_wear_max: float = 253.0
    fault_probability: float = 0.03      # 3% of readings are faults


class IndustrialSimulator:
    """Stateful sensor simulator that generates realistic time-series data.

    State is maintained across calls — tool wear accumulates, faults are
    injected based on wear level and random probability.

    Usage:
        sim = IndustrialSimulator()
        for reading in sim.stream(n=100):
            print(reading)
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._tool_wear: float = 0.0
        self._cycle: int = 0

    def _next_reading(self, inject_fault: bool = False) -> dict[str, Any]:
        cfg = self.config
        self._cycle += 1
        self._tool_wear = min(self._tool_wear + cfg.tool_wear_increment, cfg.tool_wear_max)

        if inject_fault or (
            self._rng.random() < cfg.fault_probability
            or self._tool_wear > cfg.tool_wear_max * 0.9
        ):
            return self._fault_reading()

        air_temp = float(self._rng.normal(cfg.air_temp_mean, cfg.air_temp_std))
        process_temp = float(air_temp + cfg.process_temp_delta + self._rng.normal(0, cfg.process_temp_std))
        rpm = float(max(100.0, self._rng.normal(cfg.rpm_mean, cfg.rpm_std)))
        torque = float(max(0.0, self._rng.normal(cfg.torque_mean, cfg.torque_std)))

        return {
            "machine_id": cfg.machine_id,
            "cycle": self._cycle,
            "timestamp": time.time(),
            "air_temperature": round(air_temp, 2),
            "process_temperature": round(process_temp, 2),
            "rotational_speed": round(rpm, 1),
            "torque": round(torque, 2),
            "tool_wear": round(self._tool_wear, 1),
            "is_fault": False,
            "fault_type": None,
        }

    def _fault_reading(self) -> dict[str, Any]:
        cfg = self.config
        fault_types = ["TWF", "HDF", "PWF", "OSF"]
        fault_type = str(self._rng.choice(fault_types))

        # Each fault type shifts different sensors
        if fault_type == "HDF":
            air_temp = float(self._rng.normal(cfg.air_temp_mean + 5, cfg.air_temp_std))
            process_temp = float(air_temp + cfg.process_temp_delta + 8)
            rpm = float(self._rng.normal(cfg.rpm_mean * 0.7, cfg.rpm_std))
            torque = float(self._rng.normal(cfg.torque_mean * 1.4, cfg.torque_std))
        elif fault_type == "PWF":
            air_temp = float(self._rng.normal(cfg.air_temp_mean, cfg.air_temp_std))
            process_temp = float(air_temp + cfg.process_temp_delta)
            rpm = float(self._rng.normal(cfg.rpm_mean * 0.5, cfg.rpm_std * 2))
            torque = float(self._rng.normal(cfg.torque_mean * 2.5, cfg.torque_std))
        elif fault_type == "OSF":
            air_temp = float(self._rng.normal(cfg.air_temp_mean, cfg.air_temp_std))
            process_temp = float(air_temp + cfg.process_temp_delta)
            rpm = float(self._rng.normal(cfg.rpm_mean * 1.3, cfg.rpm_std))
            torque = float(self._rng.normal(cfg.torque_mean * 1.8, cfg.torque_std))
        else:  # TWF
            air_temp = float(self._rng.normal(cfg.air_temp_mean, cfg.air_temp_std))
            process_temp = float(air_temp + cfg.process_temp_delta)
            rpm = float(self._rng.normal(cfg.rpm_mean, cfg.rpm_std))
            torque = float(self._rng.normal(cfg.torque_mean * 1.2, cfg.torque_std))

        return {
            "machine_id": cfg.machine_id,
            "cycle": self._cycle,
            "timestamp": time.time(),
            "air_temperature": round(air_temp, 2),
            "process_temperature": round(process_temp, 2),
            "rotational_speed": round(max(0.0, rpm), 1),
            "torque": round(max(0.0, torque), 2),
            "tool_wear": round(self._tool_wear, 1),
            "is_fault": True,
            "fault_type": fault_type,
        }

    def next(self) -> dict[str, Any]:
        """Generate one sensor reading."""
        return self._next_reading()

    def stream(self, n: int = 100, delay: float = 0.0) -> Iterator[dict[str, Any]]:
        """Generate n readings as an iterator."""
        for _ in range(n):
            yield self._next_reading()
            if delay > 0:
                time.sleep(delay)

    def batch(self, n: int = 1000) -> list[dict[str, Any]]:
        """Generate a batch of n readings."""
        return [self._next_reading() for _ in range(n)]

    def reset(self, seed: int | None = None) -> None:
        """Reset simulator state."""
        self._tool_wear = 0.0
        self._cycle = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
