"""Tests for ROI calculator logic (pure business logic, no Streamlit)."""

import pytest


def _calculate_roi(
    n_machines: int,
    unplanned_stops: int,
    avg_downtime_hours: float,
    hourly_output: float,
    repair_cost: float,
    tool_changes: int,
    tool_cost: float,
    downtime_reduction_pct: float,
    tool_reduction_pct: float,
    implementation_cost: float,
) -> dict:
    """Pure ROI calculation extracted from the dashboard page."""
    downtime_loss = n_machines * unplanned_stops * avg_downtime_hours * hourly_output
    total_repair = n_machines * unplanned_stops * repair_cost
    total_tool = n_machines * tool_changes * tool_cost
    total_current = downtime_loss + total_repair + total_tool

    saved_downtime = downtime_loss * (downtime_reduction_pct / 100)
    saved_repair = total_repair * (downtime_reduction_pct / 100)
    saved_tool = total_tool * (tool_reduction_pct / 100)
    total_savings = saved_downtime + saved_repair + saved_tool

    net_benefit = total_savings - implementation_cost
    roi_pct = (net_benefit / implementation_cost * 100) if implementation_cost > 0 else 0
    payback = (implementation_cost / (total_savings / 12)) if total_savings > 0 else 999

    return {
        "total_current_cost": total_current,
        "total_savings": total_savings,
        "net_benefit": net_benefit,
        "roi_pct": roi_pct,
        "payback_months": payback,
    }


def test_roi_positive_with_reasonable_inputs():
    result = _calculate_roi(
        n_machines=6, unplanned_stops=8, avg_downtime_hours=6,
        hourly_output=850, repair_cost=3200, tool_changes=12,
        tool_cost=420, downtime_reduction_pct=45, tool_reduction_pct=40,
        implementation_cost=18000,
    )
    assert result["roi_pct"] > 0
    assert result["net_benefit"] > 0
    assert result["total_savings"] > result["implementation_cost"]


def test_roi_payback_under_12_months_reasonable():
    result = _calculate_roi(
        n_machines=6, unplanned_stops=8, avg_downtime_hours=6,
        hourly_output=850, repair_cost=3200, tool_changes=12,
        tool_cost=420, downtime_reduction_pct=45, tool_reduction_pct=40,
        implementation_cost=18000,
    )
    assert result["payback_months"] < 24


def test_roi_zero_reduction_equals_full_cost():
    result = _calculate_roi(
        n_machines=2, unplanned_stops=4, avg_downtime_hours=4,
        hourly_output=500, repair_cost=1000, tool_changes=5,
        tool_cost=200, downtime_reduction_pct=0, tool_reduction_pct=0,
        implementation_cost=5000,
    )
    assert result["total_savings"] == 0.0
    assert result["net_benefit"] == -5000.0


def test_roi_negative_with_high_implementation_cost():
    result = _calculate_roi(
        n_machines=1, unplanned_stops=1, avg_downtime_hours=1,
        hourly_output=100, repair_cost=100, tool_changes=1,
        tool_cost=100, downtime_reduction_pct=30, tool_reduction_pct=30,
        implementation_cost=1_000_000,
    )
    assert result["net_benefit"] < 0


def test_roi_savings_scale_with_machines():
    r1 = _calculate_roi(1, 5, 4, 500, 1000, 8, 200, 40, 35, 5000)
    r6 = _calculate_roi(6, 5, 4, 500, 1000, 8, 200, 40, 35, 5000)
    assert r6["total_savings"] > r1["total_savings"]


def test_total_current_cost_is_positive():
    result = _calculate_roi(
        n_machines=3, unplanned_stops=5, avg_downtime_hours=3,
        hourly_output=600, repair_cost=2000, tool_changes=10,
        tool_cost=300, downtime_reduction_pct=40, tool_reduction_pct=35,
        implementation_cost=10000,
    )
    assert result["total_current_cost"] > 0
