"""Reusable Plotly chart components for the HAIIP dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Shared colour palette
COLORS = {
    "primary": "#00D4FF",
    "success": "#00C851",
    "warning": "#FFB400",
    "danger": "#FF4444",
    "surface": "#1C2333",
    "border": "#2D3748",
    "text": "#E2E8F0",
    "muted": "#718096",
}

PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": COLORS["text"], "family": "Inter, sans-serif"},
    "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    "legend": {
        "bgcolor": "rgba(28,35,51,0.8)",
        "bordercolor": COLORS["border"],
        "borderwidth": 1,
    },
    "xaxis": {
        "gridcolor": COLORS["border"],
        "showgrid": True,
        "zeroline": False,
    },
    "yaxis": {
        "gridcolor": COLORS["border"],
        "showgrid": True,
        "zeroline": False,
    },
}


def sensor_timeseries(
    timestamps: list,
    values: list[float],
    label: str,
    color: str = "primary",
    anomaly_mask: list[bool] | None = None,
    height: int = 220,
) -> None:
    """Line chart for a single sensor over time, with anomaly highlights."""
    fig = go.Figure()

    c = COLORS.get(color, color)

    # Normal trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            name=label,
            line={"color": c, "width": 2},
            hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
        )
    )

    # Anomaly markers
    if anomaly_mask:
        anom_x = [t for t, a in zip(timestamps, anomaly_mask) if a]
        anom_y = [v for v, a in zip(values, anomaly_mask) if a]
        if anom_x:
            fig.add_trace(
                go.Scatter(
                    x=anom_x,
                    y=anom_y,
                    mode="markers",
                    name="Anomaly",
                    marker={"color": COLORS["danger"], "size": 8, "symbol": "x"},
                    hovertemplate="⚠️ Anomaly: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(**PLOT_LAYOUT, height=height, title={"text": label, "font": {"size": 13}})
    st.plotly_chart(fig, use_container_width=True)


def multi_sensor_chart(
    timestamps: list,
    sensor_data: dict[str, list[float]],
    height: int = 300,
    title: str = "Sensor Readings",
) -> None:
    """Multi-line chart for several sensors on the same axes."""
    fig = go.Figure()
    color_cycle = [
        COLORS["primary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["danger"],
        "#A78BFA",
        "#F472B6",
    ]

    for i, (name, values) in enumerate(sensor_data.items()):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                name=name,
                line={"color": color_cycle[i % len(color_cycle)], "width": 1.8},
            )
        )

    fig.update_layout(**PLOT_LAYOUT, height=height, title={"text": title, "font": {"size": 13}})
    st.plotly_chart(fig, use_container_width=True)


def anomaly_score_gauge(score: float, title: str = "Anomaly Score") -> None:
    """Gauge chart for anomaly score [0, 1]."""
    color = (
        COLORS["success"] if score < 0.4 else COLORS["warning"] if score < 0.7 else COLORS["danger"]
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": title, "font": {"size": 13, "color": COLORS["text"]}},
            number={"suffix": "%", "font": {"size": 24, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                "bar": {"color": color},
                "bgcolor": COLORS["surface"],
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 40], "color": "#00C85122"},
                    {"range": [40, 70], "color": "#FFB40022"},
                    {"range": [70, 100], "color": "#FF444422"},
                ],
                "threshold": {
                    "line": {"color": COLORS["danger"], "width": 2},
                    "thickness": 0.8,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        font={"color": COLORS["text"]},
    )
    st.plotly_chart(fig, use_container_width=True)


def rul_bar_chart(machines: list[str], rul_values: list[int], height: int = 250) -> None:
    """Horizontal bar chart for Remaining Useful Life per machine."""
    colors = [
        COLORS["danger"] if v < 50 else COLORS["warning"] if v < 150 else COLORS["success"]
        for v in rul_values
    ]
    fig = go.Figure(
        go.Bar(
            x=rul_values,
            y=machines,
            orientation="h",
            marker_color=colors,
            text=[f"{v} cycles" for v in rul_values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>RUL: %{x} cycles<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=height,
        title={"text": "Remaining Useful Life (cycles)", "font": {"size": 13}},
        xaxis_title="Cycles remaining",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def failure_mode_pie(labels: list[str], values: list[int]) -> None:
    """Donut chart for failure mode distribution."""
    colors = [
        COLORS["danger"],
        COLORS["warning"],
        COLORS["primary"],
        "#A78BFA",
        "#F472B6",
        COLORS["muted"],
    ]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={
                "colors": colors[: len(labels)],
                "line": {"color": COLORS["surface"], "width": 2},
            },
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        font={"color": COLORS["text"]},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        title={"text": "Failure Mode Distribution", "font": {"size": 13}},
    )
    st.plotly_chart(fig, use_container_width=True)


def confidence_histogram(confidences: list[float]) -> None:
    """Histogram of model confidence scores."""
    fig = go.Figure(
        go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color=COLORS["primary"],
            opacity=0.8,
            hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=200,
        title={"text": "Confidence Distribution", "font": {"size": 13}},
        xaxis_title="Confidence",
        yaxis_title="Count",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def drift_heatmap(feature_names: list[str], psi_values: list[float]) -> None:
    """Horizontal bar showing PSI drift level per feature."""
    colors = [
        COLORS["success"] if v < 0.1 else COLORS["warning"] if v < 0.2 else COLORS["danger"]
        for v in psi_values
    ]
    fig = go.Figure(
        go.Bar(
            x=psi_values,
            y=feature_names,
            orientation="h",
            marker_color=colors,
            text=[f"PSI={v:.3f}" for v in psi_values],
            textposition="outside",
        )
    )
    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text="Monitor threshold",
        annotation_position="top right",
    )
    fig.add_vline(
        x=0.2,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text="Drift threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=max(200, len(feature_names) * 40 + 80),
        title={"text": "Feature Drift (PSI)", "font": {"size": 13}},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
