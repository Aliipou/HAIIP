"""Overview page — fleet KPI summary and machine health at a glance."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_get, is_demo
from haiip.dashboard.components.charts import confidence_histogram, failure_mode_pie
from haiip.dashboard.components.demo_data import (
    demo_alerts,
    demo_kpis,
    demo_machines,
    demo_predictions,
)
from haiip.dashboard.components.theme import badge, kpi_card, section_title, status_dot


def render() -> None:
    st.markdown("## 🏠 Fleet Overview")
    st.caption("Real-time summary of all machines across your production sites.")

    # ── Load data ─────────────────────────────────────────────────────────────
    if is_demo():
        kpis = demo_kpis()
        machines = demo_machines()
        alerts = demo_alerts()
        predictions = demo_predictions()
    else:
        machine_data = api_get("/api/v1/metrics/machines") or []
        alert_data = api_get("/api/v1/metrics/alerts/summary") or {}
        api_get("/api/v1/metrics/health") or {}
        predictions = (api_get("/api/v1/predictions", params={"size": 50}) or {}).get("items", [])

        kpis = {
            "total_machines": len(machine_data),
            "anomaly_rate": (
                sum(m.get("anomaly_rate", 0) for m in machine_data) / max(len(machine_data), 1)
            ),
            "active_alerts": alert_data.get("critical", 0) + alert_data.get("high", 0),
            "avg_rul_cycles": 0,
            "model_accuracy": 0.0,
            "uptime_pct": 0.0,
            "predictions_today": sum(m.get("total_predictions", 0) for m in machine_data),
            "feedback_count": 0,
        }
        machines = [
            {
                "machine_id": m["machine_id"],
                "status": "anomaly" if m.get("anomaly_rate", 0) > 0.1 else "normal",
                "uptime_pct": 0.0,
            }
            for m in machine_data
        ]
        alerts = []

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("⟳ Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # ── KPI row ───────────────────────────────────────────────────────────────
    section_title("Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi_card("Machines Online", str(kpis["total_machines"]), delta="All sites")
    with c2:
        ar = kpis["anomaly_rate"] * 100
        kpi_card(
            "Anomaly Rate",
            f"{ar:.1f}%",
            delta="▲ +0.2% vs yesterday" if ar > 3 else "▼ normal",
            delta_dir="up" if ar > 5 else "neu",
        )
    with c3:
        ac = kpis["active_alerts"]
        kpi_card(
            "Active Alerts",
            str(ac),
            delta="⚠️ Needs attention" if ac > 0 else "✓ All clear",
            delta_dir="up" if ac > 2 else "neu",
        )
    with c4:
        kpi_card("Predictions Today", f"{kpis['predictions_today']:,}")
    with c5:
        acc = kpis.get("model_accuracy", 0.943)
        kpi_card("Model Accuracy", f"{acc * 100:.1f}%", delta="Last 7 days", delta_dir="down")
    with c6:
        kpi_card("Human Feedback", str(kpis.get("feedback_count", 47)))

    # ── Machine fleet status ───────────────────────────────────────────────────
    section_title("Machine Fleet Status")

    status_color = {"normal": "green", "warning": "amber", "anomaly": "red"}
    status_label = {"normal": "Normal", "warning": "Warning", "anomaly": "Anomaly"}

    for machine in machines:
        s = machine.get("status", "normal")
        dot_color = status_color.get(s, "green")
        lbl = status_label.get(s, "Normal")
        uptime = machine.get("uptime_pct", 0.0)

        st.markdown(
            f"""
            <div class="machine-row">
                <div>
                    {status_dot(dot_color)}
                    <strong style="color:#E2E8F0;">{machine["machine_id"]}</strong>
                    &nbsp;&nbsp;
                    <span style="color:#718096; font-size:0.8rem;">
                        {machine.get("location", "")}
                    </span>
                </div>
                <div style="display:flex; gap:1rem; align-items:center;">
                    <span style="font-size:0.8rem; color:#718096;">
                        Uptime: {uptime:.1f}%
                    </span>
                    {badge(lbl, s)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Charts row ────────────────────────────────────────────────────────────
    section_title("Analytics")
    chart_l, chart_r = st.columns(2)

    with chart_l:
        failure_labels = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        if predictions:
            from collections import Counter

            counts = Counter(p.get("prediction_label", "no_failure") for p in predictions)
            failure_values = [
                counts.get(k, 0) for k in ["no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
            ]
        else:
            failure_values = [420, 18, 12, 9, 7, 4]
        failure_mode_pie(failure_labels, failure_values)

    with chart_r:
        confs = (
            [p.get("confidence", 0.9) for p in predictions]
            if predictions
            else [round(0.7 + i * 0.003, 3) for i in range(100)]
        )
        confidence_histogram(confs)

    # ── Active alerts preview ─────────────────────────────────────────────────
    unacked = [a for a in alerts if not a.get("is_acknowledged")]
    if unacked:
        section_title(f"Active Alerts ({len(unacked)})")
        sev_color = {
            "critical": "danger",
            "high": "warning",
            "medium": "medium",
            "low": "low",
        }
        for alert in unacked[:5]:
            sev = alert.get("severity", "medium")
            st.markdown(
                f"""
                <div style="background:#1C2333; border:1px solid #2D3748; border-left:3px solid
                    {"#FF4444" if sev == "critical" else "#FFB400" if sev == "high" else "#718096"};
                    border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between;">
                        <strong style="color:#E2E8F0;">{alert.get("title", "")}</strong>
                        {badge(sev.upper(), sev_color.get(sev, "low"))}
                    </div>
                    <div style="font-size:0.8rem; color:#718096; margin-top:4px;">
                        {alert.get("machine_id", "")} &nbsp;·&nbsp; {alert.get("created_at", "")[:19]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
