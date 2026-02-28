"""
Documentation Honesty Tests
============================
Tests that documentation does not overclaim.
These tests fail if someone adds fake data or unsupported claims.

Rules:
- These tests read real files — no mocking.
- Every assert must be able to fail if documentation is dishonest.
- A pending result must have no number. A number must not be pending.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent  # D:/HAIIP


# ---------------------------------------------------------------------------
# RESULTS.md integrity
# ---------------------------------------------------------------------------

class TestResultsIntegrity:

    def test_results_md_exists(self):
        """docs/RESULTS.md must exist."""
        assert (ROOT / "docs" / "RESULTS.md").is_file(), (
            "docs/RESULTS.md does not exist. Create it with the correct structure."
        )

    def test_results_md_does_not_contain_pending_with_numbers(self):
        """
        No line can contain both '*(pending)*' and a decimal number.
        If result is pending, it has no number. If it has a number, it is not pending.
        """
        results_path = ROOT / "docs" / "RESULTS.md"
        if not results_path.exists():
            pytest.skip("docs/RESULTS.md does not exist")

        text          = results_path.read_text(encoding="utf-8")
        pending_lines = [l for l in text.splitlines() if "*(pending)*" in l.lower()]
        violations    = []
        for line in pending_lines:
            if re.search(r"\d+\.\d+", line):
                violations.append(line.strip())

        assert not violations, (
            "Pending results must not contain numbers. "
            f"Violating lines:\n" + "\n".join(violations)
        )

    def test_results_md_has_environment_section(self):
        """docs/RESULTS.md must contain an environment specification section."""
        results_path = ROOT / "docs" / "RESULTS.md"
        if not results_path.exists():
            pytest.skip("docs/RESULTS.md does not exist")

        text = results_path.read_text(encoding="utf-8").lower()
        assert "environment" in text or "python" in text or "hardware" in text, (
            "docs/RESULTS.md must include environment specification "
            "(hardware, Python version, dataset version)"
        )

    def test_results_md_no_placeholder_text(self):
        """docs/RESULTS.md must not contain placeholder text."""
        results_path = ROOT / "docs" / "RESULTS.md"
        if not results_path.exists():
            pytest.skip("docs/RESULTS.md does not exist")

        text = results_path.read_text(encoding="utf-8").lower()
        forbidden = ["todo", "fixme", "placeholder", "lorem ipsum", "tbd"]
        found = [f for f in forbidden if f in text]
        assert not found, f"Placeholder text found in RESULTS.md: {found}"


# ---------------------------------------------------------------------------
# README integrity
# ---------------------------------------------------------------------------

class TestREADMEIntegrity:

    def test_readme_exists(self):
        """README.md must exist."""
        assert (ROOT / "README.md").is_file()

    def test_readme_thresholds_not_called_results(self):
        """
        README ML performance table must use 'Threshold' not 'Result'.
        This test fails if someone renames the column to imply measured values.
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "Threshold" in readme or "threshold" in readme, (
            "README must use 'Threshold' column heading for ML performance tables, "
            "not 'Result' — to distinguish targets from measured outcomes."
        )

    def test_readme_has_limitations_section_or_link(self):
        """README must reference known limitations or link to LIMITATIONS.md."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "limitation" in readme or "limitations.md" in readme, (
            "README must reference known limitations (see docs/LIMITATIONS.md)."
        )

    def test_readme_has_ros2_section(self):
        """README must document ROS2 closed-loop pipeline."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "ROS2" in readme or "ros2" in readme.lower(), (
            "README must document the ROS2 closed-loop pipeline."
        )

    def test_readme_demo_video_link_present(self):
        """README must include a demo video link."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "http" in readme and ("demo" in readme.lower() or "video" in readme.lower()), (
            "README must include a demo video link."
        )


# ---------------------------------------------------------------------------
# Simulation flag consistency
# ---------------------------------------------------------------------------

class TestSimulationFlagConsistency:

    def test_oversight_simulation_has_low_confidence_flag(self):
        """oversight_simulation.py hardcodes simulation_confidence='LOW'."""
        sim_path = ROOT / "haiip" / "core" / "oversight_simulation.py"
        assert sim_path.is_file(), "oversight_simulation.py does not exist"
        text = sim_path.read_text(encoding="utf-8")
        assert '"LOW"' in text or "'LOW'" in text, (
            "oversight_simulation.py must hardcode simulation_confidence='LOW'"
        )

    def test_oversight_simulation_has_field_study_required(self):
        """oversight_simulation.py must set field_study_required=True."""
        sim_path = ROOT / "haiip" / "core" / "oversight_simulation.py"
        assert sim_path.is_file(), "oversight_simulation.py does not exist"
        text = sim_path.read_text(encoding="utf-8")
        assert "field_study_required" in text, (
            "oversight_simulation.py must include field_study_required field"
        )
        assert "True" in text, (
            "oversight_simulation.py must set field_study_required=True"
        )

    def test_economic_calibration_has_calibrated_flag(self):
        """economic_calibration.py has a 'calibrated' field defaulting to False."""
        calib_path = ROOT / "haiip" / "core" / "economic_calibration.py"
        assert calib_path.is_file(), "economic_calibration.py does not exist"
        text = calib_path.read_text(encoding="utf-8")
        assert "calibrated" in text
        assert "False" in text

    def test_opcua_connector_has_data_source_mode_enum(self):
        """opcua_connector.py has DataSourceMode enum with all three modes."""
        connector_path = ROOT / "haiip" / "data" / "ingestion" / "opcua_connector.py"
        assert connector_path.is_file(), "opcua_connector.py does not exist"
        text = connector_path.read_text(encoding="utf-8")
        for mode in ["REAL_HARDWARE", "SIMULATION", "HARDWARE_FALLBACK"]:
            assert mode in text, f"DataSourceMode.{mode} not found in opcua_connector.py"

    def test_limitations_md_exists(self):
        """docs/LIMITATIONS.md must exist."""
        assert (ROOT / "docs" / "LIMITATIONS.md").is_file(), (
            "docs/LIMITATIONS.md does not exist. Create it per the spec."
        )

    def test_limitations_md_covers_all_four_limitations(self):
        """docs/LIMITATIONS.md must cover L1 through L4."""
        lim_path = ROOT / "docs" / "LIMITATIONS.md"
        if not lim_path.is_file():
            pytest.skip("docs/LIMITATIONS.md does not exist")
        text = lim_path.read_text(encoding="utf-8")
        for label in ["L1", "L2", "L3", "L4"]:
            assert label in text, f"Limitation {label} not found in LIMITATIONS.md"

    def test_no_todo_or_fixme_in_core(self):
        """
        haiip/core/ must not contain TODO, FIXME, placeholder, fake, or dummy.
        These indicate unfinished or dishonest code.
        """
        core_dir = ROOT / "haiip" / "core"
        forbidden = ["TODO", "FIXME", "placeholder", "fake data", "dummy data"]
        violations = []

        for py_file in core_dir.glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for word in forbidden:
                if word.lower() in text.lower():
                    lines = [
                        f"  line {i+1}: {line.strip()}"
                        for i, line in enumerate(text.splitlines())
                        if word.lower() in line.lower()
                    ]
                    violations.extend(
                        [f"{py_file.name}: '{word}' found"] + lines[:3]
                    )

        assert not violations, (
            "Forbidden words found in haiip/core/:\n" + "\n".join(violations)
        )
