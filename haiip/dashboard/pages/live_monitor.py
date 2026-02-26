"""Live Monitor page — real-time sensor stream and anomaly detection."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st

from haiip.dashboard.components.auth import api_post, is_demo
from haiip.dashboard.components.charts import (
    anomaly_score_gauge,
    multi_sensor_chart,
    sensor_timeseries,
)
from haiip.dashboard.components.demo_data import demo_machines, demo_sensor_stream
from haiip.dashboard.components.theme import badge, kpi_card, section_title, status_dot


def render() -> None:
    st.markdown("## 📡 Live Monitor")
    st.caption("Real-time sensor stream with instant anomaly detection.")

    # ── Machine selector ──────────────────────────────────────────────────────
    machines = demo_machines() if is_demo() else []
    machine_ids = [m["machine_id"] for m in machines] or ["CNC-001"]

    col_sel, col_interval, col_start = st.columns([2, 1, 1])
    with col_sel:
        selected_machine = st.selectbox("Select machine", machine_ids)
    with col_interval:
        refresh_interval = st.selectbox("Refresh (s)", [1, 2, 5, 10], index=1)
    with col_start:
        st.markdown("<br>", unsafe_allow_html=True)
        streaming = st.toggle("▶ Live stream", value=False)

    st.markdown("---")

    # ── Init session buffer ───────────────────────────────────────────────────
    buf_key = f"sensor_buf_{selected_machine}"
    if buf_key not in st.session_state:
        st.session_state[buf_key] = demo_sensor_stream(selected_machine, n=60)

    buf = st.session_state[buf_key]

    # ── Add new reading ───────────────────────────────────────────────────────
    if streaming or st.button("⟳ Fetch reading"):
        new_reading = _fetch_reading(selected_machine)
        buf.append(new_reading)
        if len(buf) > 120:
            buf.pop(0)
        st.session_state[buf_key] = buf

    # ── Latest reading KPIs ───────────────────────────────────────────────────
    latest = buf[-1] if buf else {}
    section_title("Latest Reading")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Air Temp", f"{latest.get('air_temperature', 0):.1f} K")
    with k2:
        kpi_card("Process Temp", f"{latest.get('process_temperature', 0):.1f} K")
    with k3:
        kpi_card("Speed", f"{latest.get('rotational_speed', 0):.0f} RPM")
    with k4:
        kpi_card("Torque", f"{latest.get('torque', 0):.1f} Nm")
    with k5:
        kpi_card("Tool Wear", f"{latest.get('tool_wear', 0):.0f} min")

    # ── Anomaly status ────────────────────────────────────────────────────────
    section_title("Anomaly Status")
    score = latest.get("anomaly_score", 0.0)
    label = latest.get("label", "normal")
    conf = latest.get("confidence", 0.5)

    g_col, s_col = st.columns([1, 2])
    with g_col:
        anomaly_score_gauge(score)
    with s_col:
        s = "anomaly" if label == "anomaly" else "normal"
        st.markdown(
            f"""
            <div style="padding:1rem 0;">
                <div style="font-size:0.85rem; color:#718096; margin-bottom:0.5rem;">
                    CURRENT STATUS
                </div>
                <div style="font-size:2rem; margin-bottom:0.5rem;">
                    {badge(label.upper(), s)}
                </div>
                <div style="font-size:0.85rem; color:#718096; margin-top:0.75rem;">
                    Confidence: <strong style="color:#E2E8F0;">{conf*100:.1f}%</strong>
                </div>
                <div style="font-size:0.85rem; color:#718096; margin-top:0.25rem;">
                    Machine: <strong style="color:#E2E8F0;">{selected_machine}</strong>
                </div>
                <div style="font-size:0.85rem; color:#718096; margin-top:0.25rem;">
                    Updated: <strong style="color:#E2E8F0;">{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Time-series charts ────────────────────────────────────────────────────
    section_title("Sensor History (last 120 readings)")

    timestamps = [r["timestamp"] for r in buf]
    anomaly_mask = [r.get("label") == "anomaly" for r in buf]

    tab1, tab2, tab3 = st.tabs(["Temperature", "Mechanical", "Tool Wear"])

    with tab1:
        multi_sensor_chart(
            timestamps,
            {
                "Air Temp (K)": [r["air_temperature"] for r in buf],
                "Process Temp (K)": [r["process_temperature"] for r in buf],
            },
            title="Temperature Sensors",
        )

    with tab2:
        multi_sensor_chart(
            timestamps,
            {
                "Speed (RPM)": [r["rotational_speed"] for r in buf],
                "Torque (Nm)": [r["torque"] for r in buf],
            },
            title="Mechanical Sensors",
        )

    with tab3:
        sensor_timeseries(
            timestamps,
            [r["tool_wear"] for r in buf],
            label="Tool Wear (min)",
            color="warning",
            anomaly_mask=anomaly_mask,
        )

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("📊 Raw data table"):
        import pandas as pd
        df = pd.DataFrame(buf[-20:])
        cols = ["timestamp", "air_temperature", "process_temperature",
                "rotational_speed", "torque", "tool_wear", "label", "confidence"]
        st.dataframe(
            df[[c for c in cols if c in df.columns]].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

    # Auto-refresh
    if streaming:
        time.sleep(refresh_interval)
        st.rerun()


def _fetch_reading(machine_id: str) -> dict:
    """Fetch one reading from simulator or API."""
    if is_demo():
        from haiip.data.simulation.simulator import IndustrialSimulator, SimulatorConfig
        sim = IndustrialSimulator(SimulatorConfig(machine_id=machine_id, seed=int(time.time()) % 10000))
        r = sim.next()
        from haiip.core.anomaly import AnomalyDetector
        det = AnomalyDetector()
        features = [r["air_temperature"], r["process_temperature"],
                    r["rotational_speed"], r["torque"], r["tool_wear"]]
        result = det.predict(features)
        r.update({
            "label": result["label"],
            "confidence": result["confidence"],
            "anomaly_score": result["anomaly_score"],
        })
        return r
    else:
        import numpy as np
        rng = np.random.default_rng()
        payload = {
            "machine_id": machine_id,
            "air_temperature": float(rng.normal(300.0, 2.0)),
            "process_temperature": float(rng.normal(310.0, 1.5)),
            "rotational_speed": float(max(0, rng.normal(1538.0, 179.0))),
            "torque": float(max(0, rng.normal(40.0, 9.8))),
            "tool_wear": float(rng.uniform(0, 253)),
        }
        result = api_post("/api/v1/predict", payload) or {}
        data = result.get("data", {})
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": machine_id,
            **payload,
            "label": data.get("prediction_label", "normal"),
            "confidence": data.get("confidence", 0.5),
            "anomaly_score": data.get("anomaly_score", 0.0),
        }
