"""Tests for data/simulation/simulator.py."""

import pytest

from haiip.data.simulation.simulator import IndustrialSimulator, SimulatorConfig


def test_next_returns_dict():
    sim = IndustrialSimulator()
    reading = sim.next()
    assert isinstance(reading, dict)
    assert "machine_id" in reading
    assert "air_temperature" in reading
    assert "tool_wear" in reading


def test_stream_yields_n_readings():
    sim = IndustrialSimulator()
    readings = list(sim.stream(n=50))
    assert len(readings) == 50


def test_batch_returns_list():
    sim = IndustrialSimulator()
    batch = sim.batch(n=100)
    assert len(batch) == 100


def test_tool_wear_increases():
    sim = IndustrialSimulator()
    readings = sim.batch(n=10)
    wears = [r["tool_wear"] for r in readings]
    # Tool wear should generally increase (with small increments)
    assert wears[-1] >= wears[0]


def test_readings_within_plausible_ranges():
    sim = IndustrialSimulator()
    for reading in sim.batch(n=200):
        assert 270 < reading["air_temperature"] < 340
        assert reading["rotational_speed"] >= 0
        assert reading["torque"] >= 0
        assert 0 <= reading["tool_wear"] <= 253


def test_fault_injection():
    sim = IndustrialSimulator(
        SimulatorConfig(fault_probability=1.0, machine_id="FAULT-TEST")
    )
    reading = sim.next()
    assert reading["is_fault"] is True
    assert reading["fault_type"] is not None


def test_custom_config():
    config = SimulatorConfig(machine_id="CUSTOM-001", seed=99)
    sim = IndustrialSimulator(config)
    reading = sim.next()
    assert reading["machine_id"] == "CUSTOM-001"


def test_reset_clears_state():
    sim = IndustrialSimulator()
    sim.batch(n=100)
    assert sim._tool_wear > 0
    sim.reset()
    assert sim._tool_wear == 0.0
    assert sim._cycle == 0


def test_cycle_increments():
    sim = IndustrialSimulator()
    for i in range(1, 6):
        reading = sim.next()
        assert reading["cycle"] == i


def test_reading_has_required_keys():
    sim = IndustrialSimulator()
    reading = sim.next()
    required = {
        "machine_id", "cycle", "timestamp", "air_temperature",
        "process_temperature", "rotational_speed", "torque", "tool_wear",
        "is_fault", "fault_type",
    }
    assert required.issubset(reading.keys())
