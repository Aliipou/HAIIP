"""Audit Trail page — EU AI Act compliance log."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_get, is_demo
from haiip.dashboard.components.demo_data import demo_audit_log
from haiip.dashboard.components.theme import badge, kpi_card, section_title

ACTION_COLORS = {
    "prediction.created":  ("primary", "🔮"),
    "feedback.submitted":  ("normal",  "👤"),
    "alert.acknowledged":  ("medium",  "🚨"),
    "model.retrained":     ("warning", "🔄"),
    "document.ingested":   ("normal",  "📄"),
    "user.login":          ("low",     "🔑"),
    "user.register":       ("low",     "👤"),
    "alert.created":       ("high",    "⚠️"),
}


def render() -> None:
    st.markdown("## 📋 Audit Trail")
    st.caption(
        "EU AI Act compliance log — complete record of all AI decisions, "
        "human overrides, and system events. Immutable, timestamped, exportable."
    )

    # ── EU AI Act info panel ──────────────────────────────────────────────────
    with st.expander("ℹ️ EU AI Act Compliance Context"):
        st.markdown(
            """
            HAIIP is classified as a **Limited-Risk AI System** under the EU AI Act
            (Article 52). Requirements met by this audit trail:

            | Requirement | Status |
            |---|---|
            | Human oversight record | ✅ All predictions logged |
            | Explanation availability | ✅ Confidence + feature importance stored |
            | Data traceability | ✅ Input features hashed and stored |
            | Incident logging | ✅ All alerts and acknowledgements logged |
            | Human override record | ✅ Feedback and corrections tracked |
            | Model version tracking | ✅ ModelRegistry records all deployments |

            Retention period: **5 years** (GDPR Art. 17 balanced with AI Act Art. 12).
            """
        )

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        action_filter = st.selectbox(
            "Filter by action",
            ["All", "prediction.created", "feedback.submitted", "alert.acknowledged",
             "model.retrained", "document.ingested", "user.login"],
        )
    with fc2:
        date_from = st.date_input("From date")
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳ Refresh", use_container_width=True):
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    if is_demo():
        logs = demo_audit_log()
    else:
        resp = api_get("/api/v1/audit") or []
        logs = resp if isinstance(resp, list) else []

    if action_filter != "All":
        logs = [l for l in logs if l.get("action") == action_filter]

    # ── Stats row ─────────────────────────────────────────────────────────────
    section_title("Compliance Statistics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Events", str(len(demo_audit_log()) if is_demo() else len(logs)))
    with k2:
        pred_count = sum(1 for l in demo_audit_log() if l["action"] == "prediction.created") if is_demo() else 0
        kpi_card("AI Decisions", str(pred_count))
    with k3:
        fb_count = sum(1 for l in demo_audit_log() if l["action"] == "feedback.submitted") if is_demo() else 0
        kpi_card("Human Reviews", str(fb_count))
    with k4:
        kpi_card("Compliance", "✅ EU AI Act")

    section_title(f"Event Log ({len(logs)} events)")

    if not logs:
        st.info("No audit events match the current filter.")
        return

    # ── Event timeline ────────────────────────────────────────────────────────
    for log in logs:
        action = log.get("action", "unknown")
        color, icon = ACTION_COLORS.get(action, ("low", "📌"))
        ts = log.get("created_at", "")[:19]
        user = log.get("user_id", "system")
        details = log.get("details", "")
        resource = log.get("resource_type", "")

        st.markdown(
            f"""
            <div style="background:#1C2333; border:1px solid #2D3748; border-radius:8px;
                        padding:0.75rem 1rem; margin-bottom:0.5rem;
                        display:flex; align-items:flex-start; gap:0.75rem;">
                <span style="font-size:1.2rem; line-height:1.4;">{icon}</span>
                <div style="flex:1;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <code style="color:#00D4FF; font-size:0.8rem;">{action}</code>
                        <span style="color:#718096; font-size:0.75rem;">{ts} UTC</span>
                    </div>
                    <div style="font-size:0.8rem; color:#A0AEC0; margin-top:4px;">
                        Resource: <strong>{resource}</strong>
                        &nbsp;·&nbsp; User: <strong>{user[:16]}</strong>
                    </div>
                    {f'<div style="font-size:0.75rem; color:#718096; margin-top:4px; font-family:monospace;">{details[:120]}</div>' if details else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_export, col_space = st.columns([1, 3])
    with col_export:
        if st.button("📥 Export audit log (CSV)", use_container_width=True):
            import pandas as pd
            df = pd.DataFrame(logs)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="haiip_audit_log.csv",
                mime="text/csv",
            )
