"""Sidebar navigation component."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import logout

NAV_ITEMS = [
    ("🏠", "Overview"),
    ("📡", "Live Monitor"),
    ("🔧", "Maintenance"),
    ("🚨", "Alerts"),
    ("💬", "Query"),
    ("👤", "Feedback"),
    ("💶", "ROI Calculator"),
    ("📋", "Audit Trail"),
    ("⚙️", "Admin"),
]


def render_sidebar() -> str:
    """Render sidebar and return selected page name."""
    with st.sidebar:
        # Logo / brand
        st.markdown(
            """
            <div style="padding:0.5rem 0 1.5rem; text-align:center;">
                <span style="font-size:1.6rem; font-weight:800; color:#00D4FF;">⚙️ HAIIP</span>
                <div style="font-size:0.7rem; color:#718096; margin-top:2px;">
                    Industrial Intelligence Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tenant badge
        tenant = st.session_state.get("tenant_slug", "unknown")
        demo = st.session_state.get("demo_mode", False)
        demo_tag = " · 🧪 demo" if demo else ""
        st.markdown(
            f"""
            <div style="background:#1C2333; border:1px solid #2D3748; border-radius:8px;
                        padding:0.6rem 0.8rem; margin-bottom:1.2rem; font-size:0.8rem;">
                <span style="color:#718096;">Workspace</span><br>
                <strong style="color:#E2E8F0;">{tenant}{demo_tag}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "Overview"

        for icon, name in NAV_ITEMS:
            _ = st.session_state["nav_page"] == name
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
                help=name,
            ):
                st.session_state["nav_page"] = name
                st.rerun()

        st.markdown("---")

        # System status
        st.markdown(
            """
            <div style="font-size:0.72rem; color:#718096; margin-bottom:0.5rem;">
                SYSTEM STATUS
            </div>
            """,
            unsafe_allow_html=True,
        )
        _status_row("API Server", "green")
        _status_row("ML Models", "green")
        _status_row("OPC UA", "amber")
        _status_row("MQTT", "green")

        st.markdown("---")

        # User + logout
        email = st.session_state.get("user_email", "")
        st.markdown(
            f'<div style="font-size:0.8rem; color:#718096; margin-bottom:0.5rem;">👤 {email}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", use_container_width=True):
            logout()

        st.markdown(
            '<div style="font-size:0.68rem; color:#4A5568; text-align:center; margin-top:1rem;">'
            "v0.1.0 · NextIndustriAI</div>",
            unsafe_allow_html=True,
        )

    return st.session_state.get("nav_page", "Overview")


def _status_row(label: str, color: str) -> None:
    color_map = {"green": "#00C851", "amber": "#FFB400", "red": "#FF4444"}
    hex_color = color_map.get(color, "#718096")
    text_map = {"green": "Online", "amber": "Connecting", "red": "Offline"}
    status_text = text_map.get(color, "Unknown")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    font-size:0.75rem; margin-bottom:4px;">
            <span style="color:#A0AEC0;">{label}</span>
            <span style="color:{hex_color};">● {status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
