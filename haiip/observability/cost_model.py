"""Per-prediction compute and downtime cost model.

Answers the key ROI questions:
    1. How much does each AI inference cost to compute? (compute_cost_eur)
    2. How much does a missed failure cost in downtime? (downtime_cost_eur)
    3. What is the net ROI of the AI system over a fleet? (fleet_roi)

Cost model (reference: AWS EC2 pricing 2024, Nordic energy prices):
    compute_cost = (inference_time_ms / 1000) × gpu_hourly_rate / 3600
    storage_cost = model_size_mb × storage_rate_eur_gb_month / 30 / 24 / 1000

Downtime cost model:
    avoided_downtime_eur = P(failure_detected) × C_downtime − C_maintenance
    false_negative_cost  = P(missed_failure)   × C_downtime × safety_factor

Usage::
    model  = PredictionCostModel()
    report = model.compute(
        inference_time_ms=45.2,
        failure_probability=0.72,
        was_correct=True,
    )
    print(report.net_value_eur)   # net economic value of this prediction
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CostReport:
    """Economic cost/value breakdown for one AI prediction.

    Attributes:
        report_id:           UUID for audit trail
        compute_cost_eur:    Cost of running the inference (€)
        avoided_downtime_eur: Expected downtime avoided by correct prediction (€)
        false_negative_cost_eur: Cost if this is a missed failure (€)
        net_value_eur:       avoided_downtime − compute_cost (positive = profitable)
        inference_time_ms:   Measured latency (ms)
        failure_probability: P(failure) from ML model
        was_correct:         Whether prediction matched ground truth
        metadata:            Arbitrary metadata
    """
    report_id:              str
    compute_cost_eur:       float
    avoided_downtime_eur:   float
    false_negative_cost_eur: float
    net_value_eur:          float
    inference_time_ms:      float
    failure_probability:    float
    was_correct:            bool
    metadata:               dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id":              self.report_id,
            "compute_cost_eur":       round(self.compute_cost_eur, 6),
            "avoided_downtime_eur":   round(self.avoided_downtime_eur, 2),
            "false_negative_cost_eur": round(self.false_negative_cost_eur, 2),
            "net_value_eur":          round(self.net_value_eur, 2),
            "inference_time_ms":      round(self.inference_time_ms, 2),
            "failure_probability":    round(self.failure_probability, 4),
            "was_correct":            self.was_correct,
            "metadata":               self.metadata,
        }


class PredictionCostModel:
    """Computes per-prediction economic cost and value.

    Args:
        gpu_hourly_rate_eur:   Cost of GPU/CPU compute per hour (€/h)
        downtime_cost_eur:     Cost of one unplanned failure (€)
        maintenance_cost_eur:  Cost of one planned maintenance (€)
        safety_factor:         Regulatory risk multiplier for missed failures
        model_size_mb:         Serialised model size (for storage cost)
        storage_rate_eur_gb:   Cloud storage rate (€/GB/month)
    """

    def __init__(
        self,
        gpu_hourly_rate_eur:  float = 0.50,   # t3.medium equivalent
        downtime_cost_eur:    float = 4_000.0,
        maintenance_cost_eur: float = 590.0,
        safety_factor:        float = 1.5,
        model_size_mb:        float = 10.0,
        storage_rate_eur_gb:  float = 0.023,  # AWS S3 eu-north-1
    ) -> None:
        self.gpu_hourly_rate_eur  = gpu_hourly_rate_eur
        self.downtime_cost_eur    = downtime_cost_eur
        self.maintenance_cost_eur = maintenance_cost_eur
        self.safety_factor        = safety_factor
        self.model_size_mb        = model_size_mb
        self.storage_rate_eur_gb  = storage_rate_eur_gb

    def compute(
        self,
        inference_time_ms:   float,
        failure_probability: float,
        was_correct:         bool = True,
        metadata:            dict[str, Any] | None = None,
    ) -> CostReport:
        """Compute economic value of one AI prediction.

        Args:
            inference_time_ms:   Measured inference latency (ms)
            failure_probability: P(failure) [0, 1]
            was_correct:         True if prediction matched ground truth
            metadata:            Arbitrary audit fields

        Returns:
            CostReport with full cost breakdown
        """
        p = float(np.clip(failure_probability, 0.0, 1.0))

        # Compute cost: GPU time for this inference
        gpu_rate_per_ms = self.gpu_hourly_rate_eur / 3_600_000
        compute_cost    = inference_time_ms * gpu_rate_per_ms

        # Value: expected downtime avoided if we act on this prediction
        avoided = p * self.downtime_cost_eur - self.maintenance_cost_eur if p > 0.5 else 0.0
        avoided = max(0.0, avoided)

        # Cost of false negative: we said "normal" but machine failed
        fn_cost = (
            self.downtime_cost_eur * self.safety_factor
            if (not was_correct and p < 0.5) else 0.0
        )

        net_value = avoided - compute_cost - fn_cost

        return CostReport(
            report_id              = str(uuid.uuid4()),
            compute_cost_eur       = compute_cost,
            avoided_downtime_eur   = avoided,
            false_negative_cost_eur= fn_cost,
            net_value_eur          = net_value,
            inference_time_ms      = inference_time_ms,
            failure_probability    = p,
            was_correct            = was_correct,
            metadata               = metadata or {},
        )

    def fleet_roi(
        self,
        reports: list[CostReport],
        period_days: int = 30,
    ) -> dict[str, Any]:
        """Aggregate ROI across fleet predictions for a reporting period.

        Returns:
            dict with total_compute_cost, total_avoided_downtime,
            total_false_negative_cost, net_roi_eur, roi_ratio,
            avg_inference_ms, predictions_per_day
        """
        if not reports:
            return {"net_roi_eur": 0.0, "predictions_total": 0}

        total_compute  = sum(r.compute_cost_eur       for r in reports)
        total_avoided  = sum(r.avoided_downtime_eur   for r in reports)
        total_fn       = sum(r.false_negative_cost_eur for r in reports)
        net_roi        = total_avoided - total_compute - total_fn
        roi_ratio      = net_roi / max(total_compute, 1e-9)
        avg_latency    = float(np.mean([r.inference_time_ms for r in reports]))

        # Storage cost for the period
        storage_cost = (
            self.model_size_mb / 1024
            * self.storage_rate_eur_gb
            * (period_days / 30)
        )

        return {
            "total_compute_cost_eur":      round(total_compute, 4),
            "total_avoided_downtime_eur":  round(total_avoided, 2),
            "total_false_negative_cost_eur": round(total_fn, 2),
            "storage_cost_eur":            round(storage_cost, 4),
            "net_roi_eur":                 round(net_roi, 2),
            "roi_ratio":                   round(roi_ratio, 2),
            "avg_inference_ms":            round(avg_latency, 2),
            "predictions_total":           len(reports),
            "predictions_per_day":         round(len(reports) / period_days, 1),
            "period_days":                 period_days,
        }
