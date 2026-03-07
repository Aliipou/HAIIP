"""Tests for dashboard chart components (unit-level, no Streamlit runtime)."""

from haiip.dashboard.components.charts import COLORS, PLOT_LAYOUT


def test_colors_have_required_keys():
    required = ["primary", "success", "warning", "danger", "surface", "border"]
    for key in required:
        assert key in COLORS, f"Missing color: {key}"


def test_colors_are_hex():
    for name, value in COLORS.items():
        assert value.startswith("#"), f"Color {name} is not hex: {value}"


def test_plot_layout_has_required_keys():
    assert "paper_bgcolor" in PLOT_LAYOUT
    assert "plot_bgcolor" in PLOT_LAYOUT
    assert "font" in PLOT_LAYOUT
    assert "margin" in PLOT_LAYOUT


def test_colors_primary_is_cyan():
    assert COLORS["primary"] == "#00D4FF"


def test_colors_success_is_green():
    assert COLORS["success"] == "#00C851"


def test_colors_danger_is_red():
    assert COLORS["danger"] == "#FF4444"
