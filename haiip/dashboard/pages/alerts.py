"""Alerts page — view, filter, and acknowledge alerts."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_get, api_post, is_demo
from haiip.dashboard.components.demo_data import demo_alerts
from haiip.dashboard.components.theme import badge, section_title

SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEV_COLOR = {"critical": "danger", "high": "warning", "medium": "medium", "low": "low"}
SEV_BORDER = {
    "critical": "#FF4444",
    "high": "#FF8C00",
    "medium": "#FFB400",
    "low": "#2D3748",
}


def render() -> None:
    st.markdown("## 🚨 Alerts")
    st.caption("Monitor, filter, and acknowledge machine alerts across all sites.")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        sev_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
    with fc2:
        ack_filter = st.selectbox("Status", ["All", "Unacknowledged", "Acknowledged"])
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳ Refresh alerts", use_container_width=True):
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    if is_demo():
        alerts = demo_alerts()
    else:
        params: dict = {}
        if sev_filter != "All":
            params["severity"] = sev_filter.lower()
        if ack_filter == "Unacknowledged":
            params["unacknowledged_only"] = "true"
        resp = api_get("/api/v1/alerts", params=params)
        alerts = (resp or {}).get("items", [])

    # ── Apply local filters ───────────────────────────────────────────────────
    if sev_filter != "All":
        alerts = [a for a in alerts if a.get("severity") == sev_filter.lower()]
    if ack_filter == "Unacknowledged":
        alerts = [a for a in alerts if not a.get("is_acknowledged")]
    elif ack_filter == "Acknowledged":
        alerts = [a for a in alerts if a.get("is_acknowledged")]

    alerts = sorted(
        alerts,
        key=lambda a: (
            SEV_ORDER.get(a.get("severity", "low"), 3),
            a.get("created_at", ""),
        ),
    )

    # ── Summary badges ────────────────────────────────────────────────────────
    all_alerts = demo_alerts() if is_demo() else alerts
    s_counts = {
        s: sum(1 for a in all_alerts if a.get("severity") == s)
        for s in ["critical", "high", "medium", "low"]
    }

    st.markdown(
        f"""
        <div style="display:flex; gap:1rem; margin:0.5rem 0 1rem; flex-wrap:wrap;">
            <span class="badge badge-critical">Critical: {s_counts["critical"]}</span>
            <span class="badge badge-high">High: {s_counts["high"]}</span>
            <span class="badge badge-medium">Medium: {s_counts["medium"]}</span>
            <span class="badge badge-low">Low: {s_counts["low"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not alerts:
        st.success("✅ No alerts match the current filter.")
        return

    section_title(f"{len(alerts)} Alert(s)")

    # ── Alert cards ───────────────────────────────────────────────────────────
    for alert in alerts:
        sev = alert.get("severity", "low")
        border_color = SEV_BORDER.get(sev, "#2D3748")
        acked = alert.get("is_acknowledged", False)
        alert_id = alert.get("id", "")

        with st.container():
            st.markdown(
                f"""
                <div style="background:#1C2333; border:1px solid #2D3748;
                            border-left:4px solid {border_color};
                            border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.6rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="color:#E2E8F0; font-size:0.95rem;">
                            {alert.get("title", "Alert")}
                        </strong>
                        <div style="display:flex; gap:0.5rem; align-items:center;">
                            {badge(sev.upper(), SEV_COLOR.get(sev, "low"))}
                            {'<span class="badge badge-low">ACK</span>' if acked else ""}
                        </div>
                    </div>
                    <div style="color:#A0AEC0; font-size:0.82rem; margin-top:6px;">
                        {alert.get("message", "")}
                    </div>
                    <div style="color:#718096; font-size:0.75rem; margin-top:8px;">
                        🖥 {alert.get("machine_id", "")} &nbsp;·&nbsp;
                        🕐 {alert.get("created_at", "")[:19]}
                        {" &nbsp;·&nbsp; ✅ Acknowledged" if acked else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not acked:
                col_ack, col_space = st.columns([1, 4])
                with col_ack:
                    if st.button("✓ Acknowledge", key=f"ack_{alert_id}"):
                        _acknowledge(alert_id)

    # ── Create manual alert ───────────────────────────────────────────────────
    with st.expander("➕ Create manual alert"):
        with st.form("create_alert_form"):
            machine_id = st.text_input("Machine ID", placeholder="CNC-001")
            col_sev, col_title = st.columns([1, 3])
            with col_sev:
                severity = st.selectbox("Severity", ["critical", "high", "medium", "low"])
            with col_title:
                title = st.text_input("Title")
            message = st.text_area("Message", height=80)
            submitted = st.form_submit_button("Create Alert", use_container_width=True)

        if submitted and machine_id and title:
            if is_demo():
                st.success(f"✅ Alert '{title}' created (demo mode — not persisted).")
            else:
                resp = api_post(
                    "/api/v1/alerts",
                    {
                        "machine_id": machine_id,
                        "severity": severity,
                        "title": title,
                        "message": message,
                    },
                )
                if resp:
                    st.success("Alert created successfully.")
                    st.rerun()
                else:
                    st.error("Failed to create alert.")


def _acknowledge(alert_id: str) -> None:
    if is_demo():
        st.success("✅ Alert acknowledged (demo mode).")
        return
    resp = api_post(f"/api/v1/alerts/{alert_id}/acknowledge", {})
    if resp:
        st.success("Alert acknowledged.")
        st.rerun()
    else:
        st.error("Failed to acknowledge alert.")
