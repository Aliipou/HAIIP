"""HAIIP Streamlit Dashboard — entry point.

Multi-page app with sidebar navigation.
Pages:
    1. Overview          — KPI summary, fleet health at a glance
    2. Live Monitor      — real-time sensor stream + anomaly detection
    3. Maintenance       — predictive maintenance, failure modes, RUL
    4. Alerts            — alert list, acknowledge, history
    5. Query (RAG)       — natural-language knowledge base search
    6. Feedback          — human-in-the-loop prediction correction
    7. ROI Calculator    — SME business case calculator
    8. Audit Trail       — EU AI Act compliance log
    9. Admin             — tenant management (admin only)
"""

import streamlit as st

from haiip.dashboard.components.auth import render_login_page, require_login
from haiip.dashboard.components.sidebar import render_sidebar
from haiip.dashboard.components.theme import apply_theme

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAIIP — Industrial AI Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/haiip/haiip",
        "Report a bug": "https://github.com/haiip/haiip/issues",
        "About": "HAIIP v0.1.0 — Human-Aligned Industrial Intelligence Platform",
    },
)

apply_theme()

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not require_login():
    render_login_page()
    st.stop()

# ── Navigation ────────────────────────────────────────────────────────────────
page = render_sidebar()

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Overview":
    from haiip.dashboard.pages.overview import render

    render()
elif page == "Live Monitor":
    from haiip.dashboard.pages.live_monitor import render

    render()
elif page == "Maintenance":
    from haiip.dashboard.pages.maintenance import render

    render()
elif page == "Alerts":
    from haiip.dashboard.pages.alerts import render

    render()
elif page == "Query":
    from haiip.dashboard.pages.query import render

    render()
elif page == "Feedback":
    from haiip.dashboard.pages.feedback import render

    render()
elif page == "ROI Calculator":
    from haiip.dashboard.pages.roi_calculator import render

    render()
elif page == "Audit Trail":
    from haiip.dashboard.pages.audit_trail import render

    render()
elif page == "Admin":
    from haiip.dashboard.pages.admin import render

    render()
