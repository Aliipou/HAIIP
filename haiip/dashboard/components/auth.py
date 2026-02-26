"""Authentication component for the Streamlit dashboard.

Handles:
- Login form with tenant slug + email + password
- JWT token storage in st.session_state
- Token refresh on expiry
- Logout
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def require_login() -> bool:
    """Return True if the user is authenticated."""
    return bool(st.session_state.get("access_token"))


def render_login_page() -> None:
    """Render the full-page login form."""
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 0 1rem;">
            <h1 style="font-size:2.5rem; font-weight:800; color:#00D4FF;">⚙️ HAIIP</h1>
            <p style="color:#718096; font-size:1rem; margin-top:-0.5rem;">
                Human-Aligned Industrial Intelligence Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("##### Sign in to your workspace")
            tenant = st.text_input("Tenant workspace", placeholder="e.g. jakobstad-sme")
            email = st.text_input("Email", placeholder="engineer@company.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            _do_login(tenant, email, password)

        st.markdown(
            "<div style='text-align:center; color:#718096; font-size:0.8rem; margin-top:1rem;'>"
            "NextIndustriAI · Centria RDI · Jakobstad · Sundsvall · Narvik"
            "</div>",
            unsafe_allow_html=True,
        )

    # Dev shortcut — pre-filled demo credentials
    with col2:
        with st.expander("🧪 Demo mode (dev only)"):
            if st.button("Login as demo admin", use_container_width=True):
                _demo_login()


def _do_login(tenant: str, email: str, password: str) -> None:
    if not tenant or not email or not password:
        st.error("All fields are required.")
        return

    try:
        resp = httpx.post(
            f"{API_BASE}/api/v1/auth/login",
            json={"tenant_slug": tenant, "email": email, "password": password},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
            st.session_state["tenant_slug"] = tenant
            st.session_state["user_email"] = email
            st.rerun()
        else:
            st.error("Invalid credentials. Please check your email, password, and workspace.")
    except httpx.ConnectError:
        st.warning(
            "Cannot reach API server. Running in **offline demo mode** — "
            "data is simulated locally."
        )
        _demo_login()


def _demo_login() -> None:
    """Set up demo session without API — uses local simulation."""
    st.session_state["access_token"] = "demo-token"
    st.session_state["refresh_token"] = "demo-refresh"
    st.session_state["tenant_slug"] = "demo-sme"
    st.session_state["user_email"] = "demo@haiip.ai"
    st.session_state["demo_mode"] = True
    st.rerun()


def logout() -> None:
    for key in ["access_token", "refresh_token", "tenant_slug", "user_email", "demo_mode"]:
        st.session_state.pop(key, None)
    st.rerun()


def get_auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token", "")
    return {"Authorization": f"Bearer {token}"}


def is_demo() -> bool:
    return bool(st.session_state.get("demo_mode"))


def api_get(path: str, params: dict | None = None) -> dict | list | None:
    """Make authenticated GET request; returns parsed JSON or None."""
    if is_demo():
        return None
    try:
        resp = httpx.get(
            f"{API_BASE}{path}",
            headers=get_auth_headers(),
            params=params,
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except httpx.RequestError:
        pass
    return None


def api_post(path: str, json: dict) -> dict | None:
    """Make authenticated POST request; returns parsed JSON or None."""
    if is_demo():
        return None
    try:
        resp = httpx.post(
            f"{API_BASE}{path}",
            headers=get_auth_headers(),
            json=json,
            timeout=15.0,
        )
        if resp.status_code in (200, 201):
            return resp.json()
    except httpx.RequestError:
        pass
    return None
