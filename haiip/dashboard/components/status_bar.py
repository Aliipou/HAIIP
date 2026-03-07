"""
Data Source Status Bar Component
=================================
Shows the hardware mode banner in the Streamlit dashboard.

DESIGN PRINCIPLE: Operators must always know whether they are looking at
real hardware data or simulation data. There is no hidden state.

Usage:
    from haiip.dashboard.components.status_bar import render_data_source_banner
    from haiip.data.ingestion.opcua_connector import DataSourceMode

    render_data_source_banner(DataSourceMode.HARDWARE_FALLBACK)
"""

from __future__ import annotations

try:
    import streamlit as st

    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False


def render_data_source_banner(mode: str | None = None) -> None:
    """
    Render a visible hardware/simulation mode banner.

    mode: DataSourceMode value or string. Defaults to 'simulation' if None.

    Banner colours:
      REAL_HARDWARE     -> green   (safe to act on)
      HARDWARE_FALLBACK -> warning (demonstration only)
      SIMULATION        -> info    (synthetic data)
    """
    if not _STREAMLIT:
        return

    mode_str = str(mode).lower() if mode else "simulation"

    if mode_str == "real_hardware":
        st.success(
            "Connected to hardware — OPC UA live. Predictions are based on real sensor data."
        )
    elif mode_str == "hardware_fallback":
        st.warning(
            "Hardware unavailable — running on simulation data. "
            "Predictions are for demonstration only and must NOT be used for operational decisions."
        )
    elif mode_str == "simulation":
        st.info("Simulation mode — synthetic sensor data. No real hardware connected.")
    else:
        st.warning(f"Unknown data source mode: {mode_str}")


def render_simulation_metrics_banner() -> None:
    """
    Show a warning banner when oversight metrics (HOG/TCS/HIR) are
    computed from simulated operator data rather than real field data.

    Must be shown whenever the oversight dashboard is rendered without
    real operator feedback data.
    """
    if not _STREAMLIT:
        return

    st.warning(
        "**Oversight metrics are based on simulation data.**  "
        "HOG, TCS, and HIR values shown here are computed from a simulated "
        "operator behaviour model, not from real operator decisions.  "
        "A field study with real operators is required before these numbers "
        "can be reported to external stakeholders.  "
        "`simulation_confidence: LOW` | `field_study_required: True`"
    )


def render_system_status(
    api_online: bool = True,
    ml_ready: bool = True,
    hardware_mode: str = "simulation",
    redis_online: bool = True,
) -> None:
    """
    Render a compact system status row showing all service states.
    Used in sidebar or dashboard header.
    """
    if not _STREAMLIT:
        return

    cols = st.columns(4)

    with cols[0]:
        st.metric(label="API Server", value="Online" if api_online else "Offline", delta=None)

    with cols[1]:
        st.metric(label="ML Models", value="Ready" if ml_ready else "Loading", delta=None)

    with cols[2]:
        mode_label = {
            "real_hardware": "Hardware",
            "hardware_fallback": "Fallback",
            "simulation": "Simulated",
        }.get(hardware_mode.lower(), hardware_mode)
        st.metric(label="Data Source", value=mode_label, delta=None)

    with cols[3]:
        st.metric(
            label="Redis/Celery",
            value="Online" if redis_online else "Offline",
            delta=None,
        )

    # Show banner below status row if not real hardware
    render_data_source_banner(hardware_mode)
