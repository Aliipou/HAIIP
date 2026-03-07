"""
Site Economic Profile — Calibration Interface
=============================================
Every monetary parameter used by EconomicDecisionEngine must be
site-calibrated before any EUR figure is quoted externally.

Default values are Nordic manufacturing medians with source citations.
Every field has: default, source, valid range, calibration method.

IMPORTANT: These defaults are WRONG for any specific site without calibration.
Run calibration_interview() and from_interview_responses() before use.

Usage:
    profile = SiteEconomicProfile()
    violations = profile.validate()
    if violations:
        raise ValueError(f"Profile out of range: {violations}")

    interview = profile.calibration_interview()
    # ... administer to site operator (15 minutes) ...
    profile = SiteEconomicProfile.from_interview_responses(responses)
    df = profile.sensitivity_analysis()
    print(df[df['decision_changed_low'] | df['decision_changed_high']])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from haiip.core.economic_ai import CostProfile, EconomicDecisionEngine

# ---------------------------------------------------------------------------
# Valid ranges — physical justification documented per field
# ---------------------------------------------------------------------------

RANGES: dict[str, tuple[float, float]] = {
    # EUR/hr: €50 = small 2-person workshop; €50k = large automotive stamping line
    "downtime_cost_eur_per_hour": (50.0, 50_000.0),
    # EUR/hr: Finnish metalworking collective agreement range (junior to senior specialist)
    "maintenance_labour_eur_per_hour": (30.0, 200.0),
    # EUR: 0.5h labour minimum; 8h maximum (full-day wasted call)
    "false_positive_cost_eur": (15.0, 1_600.0),
    # hours: 0.5h (quick sensor swap) to 72h (complex overhaul with parts lead time)
    "mttr_hours": (0.5, 72.0),
    # EUR/hr: €100 = small machine shop; €100k = semiconductor fab line
    "production_value_eur_per_hour": (100.0, 100_000.0),
}


@dataclass
class SiteEconomicProfile:
    """
    All monetary parameters for one SME site.

    IMPORTANT: These defaults are Nordic manufacturing medians.
    They are WRONG for any specific site without calibration.
    Run calibration_interview() before quoting any EUR figure.

    Source for defaults: Senvion/LUT study 2019, Nordic manufacturing SME median.
    Updated with Finnish metalworking collective agreement 2024.
    """

    # --- Downtime cost ---
    downtime_cost_eur_per_hour: float = 850.0
    # Source: Senvion/LUT study 2019, Nordic manufacturing median
    # Valid range: [50, 50000] — €50 (small workshop) to €50k (automotive line)
    # Calibrate: ask operator for last 3 unplanned downtime events, average hourly cost

    # --- Maintenance labour ---
    maintenance_labour_eur_per_hour: float = 65.0
    # Source: Finnish metalworking collective agreement 2024, median technician rate
    # Valid range: [30, 200]
    # Calibrate: from payroll / HR system, all-in hourly rate

    # --- False positive cost ---
    false_positive_cost_eur: float = 130.0
    # = 2 hours labour at default rate
    # Valid range: [0.5h, 8h] of labour cost = [15, 1600]
    # Calibrate: cost of one unnecessary maintenance dispatch (travel + time + opportunity)

    # --- Mean time to repair ---
    mttr_hours: float = 4.0
    # Source: model assumption, no direct citation
    # Confidence: LOW — varies enormously by machine type and parts availability
    # Valid range: [0.5, 72]
    # Calibrate: from maintenance logs, average of last 10 corrective events

    # --- Production value ---
    production_value_eur_per_hour: float = 1_200.0
    # Source: Senvion/LUT study 2019, mid-range SME
    # Valid range: [100, 100000]
    # Calibrate: annual revenue from machine / operating hours per year

    # --- Site metadata ---
    site_name: str = "uncalibrated"
    calibrated: bool = False
    # Set to True only after from_interview_responses() is called

    def validate(self) -> list[str]:
        """
        Returns list of range violations. Empty list = all parameters in range.
        Raises nothing — caller decides whether to abort or warn.
        """
        violations: list[str] = []
        for field_name, (lo, hi) in RANGES.items():
            val = getattr(self, field_name)
            if not (lo <= val <= hi):
                violations.append(f"{field_name} = {val:.2f} is outside valid range [{lo}, {hi}]")
        return violations

    def calibration_interview(self) -> dict[str, Any]:
        """
        Returns a structured questionnaire for site operators.
        Designed to be administered in 15 minutes.
        Maps question answers to SiteEconomicProfile fields.
        """
        return {
            "version": "1.0",
            "instructions": (
                "Answer each question based on your last 12 months of maintenance records. "
                "Estimates are acceptable. Leave blank if unknown."
            ),
            "questions": [
                {
                    "id": "q1",
                    "field": "downtime_cost_eur_per_hour",
                    "question": (
                        "When this machine stops unexpectedly, what is the cost per hour "
                        "of lost production? (Include lost revenue, idle labour, and any "
                        "contractual penalties.)"
                    ),
                    "unit": "EUR per hour",
                    "hint": "Estimate from your last 3 unplanned stops: total cost / total hours stopped.",
                    "valid_range": list(RANGES["downtime_cost_eur_per_hour"]),
                },
                {
                    "id": "q2",
                    "field": "maintenance_labour_eur_per_hour",
                    "question": (
                        "What is the all-in hourly cost of your maintenance technician? "
                        "(Include salary, social costs, and overhead.)"
                    ),
                    "unit": "EUR per hour",
                    "hint": "Annual technician cost / 1800 working hours.",
                    "valid_range": list(RANGES["maintenance_labour_eur_per_hour"]),
                },
                {
                    "id": "q3",
                    "field": "false_positive_cost_eur",
                    "question": (
                        "When the system raises a false alarm and a technician responds "
                        "unnecessarily, what is the total cost of that dispatch? "
                        "(Travel + time + opportunity cost.)"
                    ),
                    "unit": "EUR per event",
                    "hint": "Usually 1-4 hours of technician time.",
                    "valid_range": list(RANGES["false_positive_cost_eur"]),
                },
                {
                    "id": "q4",
                    "field": "mttr_hours",
                    "question": (
                        "When a real fault occurs, how many hours does it typically take "
                        "to restore the machine to operation? "
                        "(From fault detection to production restart.)"
                    ),
                    "unit": "hours",
                    "hint": "Average from your last 10 corrective maintenance events.",
                    "valid_range": list(RANGES["mttr_hours"]),
                },
                {
                    "id": "q5",
                    "field": "production_value_eur_per_hour",
                    "question": (
                        "What is the value of one hour of production on this machine? "
                        "(Revenue or contribution margin per machine-hour.)"
                    ),
                    "unit": "EUR per hour",
                    "hint": "Annual revenue from this machine / operating hours per year.",
                    "valid_range": list(RANGES["production_value_eur_per_hour"]),
                },
                {
                    "id": "q6",
                    "field": "site_name",
                    "question": "What is the name of this site or machine?",
                    "unit": "text",
                    "hint": "e.g. 'Jakobstad CNC Line 3'",
                    "valid_range": None,
                },
            ],
        }

    @classmethod
    def from_interview_responses(cls, responses: dict[str, Any]) -> SiteEconomicProfile:
        """
        Build a SiteEconomicProfile from interview response dict.

        responses: {field_name: value, ...} OR {"q1": value, "q2": value, ...}
        Sets calibrated=True on the returned profile.
        """
        interview = cls().calibration_interview()
        id_to_field = {q["id"]: q["field"] for q in interview["questions"]}

        kwargs: dict[str, Any] = {"calibrated": True}
        for key, value in responses.items():
            if key in id_to_field:
                target_field = id_to_field[key]
                kwargs[target_field] = value if target_field == "site_name" else float(value)
            elif key in RANGES:
                kwargs[key] = float(value)
            elif key == "site_name":
                kwargs[key] = value

        return cls(**kwargs)

    def sensitivity_analysis(
        self,
        anomaly_score: float = 0.6,
        failure_probability: float = 0.6,
        variation: float = 0.5,
    ) -> pd.DataFrame:
        """
        Varies each monetary parameter +/- variation (default 50%) independently.
        Returns a table showing which parameters change the ELM decision.

        Columns: parameter, baseline_value, low_value, high_value,
                 baseline_decision, low_decision, high_decision,
                 decision_changed_low, decision_changed_high
        """
        baseline_profile = self._to_cost_profile()
        baseline_engine = EconomicDecisionEngine(cost_profile=baseline_profile)
        baseline_decision = baseline_engine.decide(
            anomaly_score=anomaly_score,
            failure_probability=failure_probability,
        ).action.value

        rows: list[dict[str, Any]] = []
        for field_name in RANGES:
            base_val = getattr(self, field_name)
            low_val = base_val * (1.0 - variation)
            high_val = base_val * (1.0 + variation)

            low_decision = (
                EconomicDecisionEngine(cost_profile=self._to_cost_profile(**{field_name: low_val}))
                .decide(anomaly_score=anomaly_score, failure_probability=failure_probability)
                .action.value
            )

            high_decision = (
                EconomicDecisionEngine(cost_profile=self._to_cost_profile(**{field_name: high_val}))
                .decide(anomaly_score=anomaly_score, failure_probability=failure_probability)
                .action.value
            )

            rows.append(
                {
                    "parameter": field_name,
                    "baseline_value": base_val,
                    "low_value": round(low_val, 2),
                    "high_value": round(high_val, 2),
                    "baseline_decision": baseline_decision,
                    "low_decision": low_decision,
                    "high_decision": high_decision,
                    "decision_changed_low": low_decision != baseline_decision,
                    "decision_changed_high": high_decision != baseline_decision,
                }
            )

        return pd.DataFrame(rows)

    def _to_cost_profile(self, **overrides: float) -> CostProfile:
        """Convert to CostProfile for EconomicDecisionEngine."""
        return CostProfile(
            production_rate_eur_hr=overrides.get(
                "production_value_eur_per_hour", self.production_value_eur_per_hour
            ),
            downtime_hours_avg=overrides.get("mttr_hours", self.mttr_hours),
            labour_rate_eur_hr=overrides.get(
                "maintenance_labour_eur_per_hour", self.maintenance_labour_eur_per_hour
            ),
            labour_hours_avg=4.0,
            parts_cost_eur=250.0,
            opportunity_cost_eur=overrides.get(
                "false_positive_cost_eur", self.false_positive_cost_eur
            ),
        )
