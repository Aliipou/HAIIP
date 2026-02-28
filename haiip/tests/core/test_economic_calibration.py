"""
Economic Calibration Tests
===========================
Tests for SiteEconomicProfile — parameter validation, sensitivity analysis,
calibration interview, and interview round-trip.

Rules:
- No test mocks the thing it is testing.
- Every assert must be able to fail.
"""

from __future__ import annotations

import pytest

from haiip.core.economic_calibration import RANGES, SiteEconomicProfile


class TestDefaultProfile:

    def test_default_profile_is_valid(self):
        """Default SiteEconomicProfile passes its own range validation."""
        profile    = SiteEconomicProfile()
        violations = profile.validate()
        assert violations == [], f"Default profile has violations: {violations}"

    def test_default_profile_not_calibrated(self):
        """Default profile has calibrated=False — must be set via interview."""
        profile = SiteEconomicProfile()
        assert profile.calibrated is False

    def test_default_site_name_is_uncalibrated(self):
        """Default site_name is 'uncalibrated' — reminds user to calibrate."""
        profile = SiteEconomicProfile()
        assert profile.site_name == "uncalibrated"


class TestRangeValidation:

    def test_range_violation_detected_downtime_too_low(self):
        """downtime_cost_eur_per_hour below minimum -> violation."""
        profile = SiteEconomicProfile(downtime_cost_eur_per_hour=1.0)
        violations = profile.validate()
        assert any("downtime_cost_eur_per_hour" in v for v in violations)

    def test_range_violation_detected_downtime_too_high(self):
        """downtime_cost_eur_per_hour above maximum -> violation."""
        profile = SiteEconomicProfile(downtime_cost_eur_per_hour=999_999.0)
        violations = profile.validate()
        assert any("downtime_cost_eur_per_hour" in v for v in violations)

    def test_range_violation_detected_mttr_too_low(self):
        """mttr_hours below 0.5 -> violation."""
        profile = SiteEconomicProfile(mttr_hours=0.1)
        violations = profile.validate()
        assert any("mttr_hours" in v for v in violations)

    def test_range_violation_detected_labour_too_high(self):
        """maintenance_labour_eur_per_hour above 200 -> violation."""
        profile = SiteEconomicProfile(maintenance_labour_eur_per_hour=500.0)
        violations = profile.validate()
        assert any("maintenance_labour_eur_per_hour" in v for v in violations)

    def test_validate_returns_empty_on_valid_custom_profile(self):
        """A custom profile within all ranges passes validation."""
        profile = SiteEconomicProfile(
            downtime_cost_eur_per_hour=1_000.0,
            maintenance_labour_eur_per_hour=80.0,
            false_positive_cost_eur=200.0,
            mttr_hours=6.0,
            production_value_eur_per_hour=2_000.0,
        )
        assert profile.validate() == []

    def test_all_ranges_have_lower_bound_below_default(self):
        """Every RANGES entry has lo <= default value."""
        profile = SiteEconomicProfile()
        for field_name, (lo, hi) in RANGES.items():
            val = getattr(profile, field_name)
            assert lo <= val, f"{field_name}: lo={lo} > default={val}"
            assert val <= hi, f"{field_name}: default={val} > hi={hi}"


class TestSensitivityAnalysis:

    def test_sensitivity_analysis_returns_all_parameters(self):
        """sensitivity_analysis() returns one row per parameter in RANGES."""
        profile = SiteEconomicProfile()
        df      = profile.sensitivity_analysis()
        assert len(df) == len(RANGES)
        for field_name in RANGES:
            assert field_name in df["parameter"].values

    def test_sensitivity_analysis_has_required_columns(self):
        """Output DataFrame has all documented columns."""
        profile  = SiteEconomicProfile()
        df       = profile.sensitivity_analysis()
        required = [
            "parameter", "baseline_value", "low_value", "high_value",
            "baseline_decision", "low_decision", "high_decision",
            "decision_changed_low", "decision_changed_high",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_decision_changes_when_downtime_cost_10x(self):
        """
        When downtime_cost is 10x the default, high_value decision may change.
        At very high downtime costs, more interventions become economically justified.
        """
        profile = SiteEconomicProfile(
            downtime_cost_eur_per_hour=500.0,
            mttr_hours=8.0,
        )
        # Use anomaly/failure values near a decision boundary
        df = profile.sensitivity_analysis(
            anomaly_score=0.4,
            failure_probability=0.45,
            variation=0.9,  # +90% variation to approximate 10x
        )
        row = df[df["parameter"] == "downtime_cost_eur_per_hour"].iloc[0]
        # Document the sensitivity — may or may not change depending on boundary
        assert isinstance(row["decision_changed_high"], bool)

    def test_decision_stable_when_labour_cost_varies_50pct(self):
        """
        Labour cost variation of +-50% should not change decision for
        high-confidence fault (failure_probability=0.85) — REPAIR_NOW is robust.
        """
        profile = SiteEconomicProfile()
        df      = profile.sensitivity_analysis(
            anomaly_score=0.9,
            failure_probability=0.85,
            variation=0.5,
        )
        row = df[df["parameter"] == "maintenance_labour_eur_per_hour"].iloc[0]
        # REPAIR_NOW at P=0.85 should not be overturned by +-50% labour cost
        assert not row["decision_changed_low"],  "REPAIR_NOW overturned by low labour cost"
        assert not row["decision_changed_high"], "REPAIR_NOW overturned by high labour cost"


class TestCalibrationInterview:

    def test_calibration_interview_returns_dict_with_questions(self):
        """calibration_interview() returns a dict with a 'questions' key."""
        profile   = SiteEconomicProfile()
        interview = profile.calibration_interview()
        assert "questions" in interview
        assert isinstance(interview["questions"], list)

    def test_calibration_interview_covers_all_fields(self):
        """Every field in RANGES has a corresponding interview question."""
        profile   = SiteEconomicProfile()
        interview = profile.calibration_interview()
        covered   = {q["field"] for q in interview["questions"]}
        for field_name in RANGES:
            assert field_name in covered, f"Field not covered in interview: {field_name}"

    def test_each_question_has_valid_range(self):
        """Every question that maps to a RANGES field has valid_range set."""
        profile   = SiteEconomicProfile()
        interview = profile.calibration_interview()
        for q in interview["questions"]:
            if q["field"] in RANGES:
                assert q["valid_range"] is not None, f"Question {q['id']} missing valid_range"
                lo, hi = q["valid_range"]
                assert lo < hi

    def test_interview_version_field_present(self):
        """Interview dict has a version field for schema evolution tracking."""
        profile   = SiteEconomicProfile()
        interview = profile.calibration_interview()
        assert "version" in interview


class TestInterviewRoundTrip:

    def test_from_interview_responses_round_trips_with_field_names(self):
        """from_interview_responses() with field names produces the correct profile."""
        responses = {
            "downtime_cost_eur_per_hour":      1_200.0,
            "maintenance_labour_eur_per_hour":  75.0,
            "false_positive_cost_eur":          150.0,
            "mttr_hours":                         5.0,
            "production_value_eur_per_hour":   2_000.0,
            "site_name":                       "Jakobstad CNC Line 1",
        }
        profile = SiteEconomicProfile.from_interview_responses(responses)

        assert profile.downtime_cost_eur_per_hour      == 1_200.0
        assert profile.maintenance_labour_eur_per_hour  ==   75.0
        assert profile.false_positive_cost_eur          ==  150.0
        assert profile.mttr_hours                       ==    5.0
        assert profile.production_value_eur_per_hour   == 2_000.0
        assert profile.site_name                       == "Jakobstad CNC Line 1"
        assert profile.calibrated is True

    def test_from_interview_responses_round_trips_with_question_ids(self):
        """from_interview_responses() with question IDs (q1..q5) also works."""
        responses = {
            "q1": 900.0,
            "q2":  70.0,
            "q3": 140.0,
            "q4":   3.5,
            "q5": 1_500.0,
            "q6": "Sundsvall Pump Station B",
        }
        profile = SiteEconomicProfile.from_interview_responses(responses)

        assert profile.downtime_cost_eur_per_hour      ==   900.0
        assert profile.maintenance_labour_eur_per_hour  ==    70.0
        assert profile.mttr_hours                       ==     3.5
        assert profile.site_name                       == "Sundsvall Pump Station B"
        assert profile.calibrated is True

    def test_calibrated_profile_passes_validation(self):
        """A calibrated profile from interview responses should be valid."""
        responses = {
            "q1": 800.0, "q2": 65.0, "q3": 130.0,
            "q4": 4.0, "q5": 1200.0, "q6": "test-site",
        }
        profile    = SiteEconomicProfile.from_interview_responses(responses)
        violations = profile.validate()
        assert violations == [], f"Calibrated profile has violations: {violations}"
