"""Admin page — tenant management, user management, model registry."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_get, api_post, is_demo
from haiip.dashboard.components.theme import badge, kpi_card, section_title


def render() -> None:
    st.markdown("## ⚙️ Admin Panel")
    st.caption("Tenant management, user administration, and model registry. Admin role required.")

    tab_users, tab_models, tab_system = st.tabs(["Users", "Model Registry", "System"])

    # ── Users tab ─────────────────────────────────────────────────────────────
    with tab_users:
        section_title("User Management")

        if is_demo():
            demo_users = [
                {
                    "email": "admin@demo-sme.com",
                    "full_name": "Demo Admin",
                    "role": "admin",
                    "is_active": True,
                },
                {
                    "email": "engineer@demo-sme.com",
                    "full_name": "Demo Engineer",
                    "role": "engineer",
                    "is_active": True,
                },
                {
                    "email": "operator@demo-sme.com",
                    "full_name": "Demo Operator",
                    "role": "operator",
                    "is_active": True,
                },
            ]
        else:
            demo_users = []

        for user in demo_users:
            role = user.get("role", "operator")
            role_colors = {
                "admin": "danger",
                "engineer": "warning",
                "operator": "normal",
                "viewer": "low",
            }
            st.markdown(
                f"""
                <div style="background:#1C2333; border:1px solid #2D3748; border-radius:8px;
                            padding:0.75rem 1rem; margin-bottom:0.5rem;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="color:#E2E8F0;">{user["full_name"]}</strong>
                        <span style="color:#718096; font-size:0.8rem; margin-left:0.5rem;">
                            {user["email"]}
                        </span>
                    </div>
                    <div style="display:flex; gap:0.5rem; align-items:center;">
                        {badge(role.upper(), role_colors.get(role, "low"))}
                        {'<span style="color:#00C851; font-size:0.8rem;">● Active</span>' if user.get("is_active") else '<span style="color:#718096;">○ Inactive</span>'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section_title("Register New User")
        with st.form("register_form"):
            col_e, col_n = st.columns(2)
            with col_e:
                new_email = st.text_input("Email")
            with col_n:
                new_name = st.text_input("Full name")
            col_p, col_r = st.columns(2)
            with col_p:
                new_password = st.text_input("Password", type="password")
            with col_r:
                new_role = st.selectbox("Role", ["operator", "engineer", "admin", "viewer"])
            submitted = st.form_submit_button("Register User", use_container_width=True)

        if submitted:
            if not new_email or not new_name or not new_password:
                st.error("All fields are required.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                if is_demo():
                    st.success(f"User '{new_email}' registered (demo mode).")
                else:
                    resp = api_post(
                        "/api/v1/auth/register",
                        {
                            "email": new_email,
                            "full_name": new_name,
                            "password": new_password,
                            "role": new_role,
                        },
                    )
                    if resp:
                        st.success(f"✅ User '{new_email}' registered as {new_role}.")
                    else:
                        st.error("Registration failed. Check password requirements.")

    # ── Model Registry tab ────────────────────────────────────────────────────
    with tab_models:
        section_title("Model Registry")
        st.caption("All trained models and their performance metrics.")

        if is_demo():
            demo_models = [
                {
                    "model_name": "AnomalyDetector",
                    "model_version": "v1.2.0",
                    "is_active": True,
                    "trained_at": "2026-02-20T08:30:00",
                    "metrics": '{"contamination": 0.05, "n_estimators": 100}',
                },
                {
                    "model_name": "MaintenancePredictor",
                    "model_version": "v1.1.0",
                    "is_active": True,
                    "trained_at": "2026-02-18T14:00:00",
                    "metrics": '{"accuracy": 0.943, "f1_macro": 0.891}',
                },
                {
                    "model_name": "AnomalyDetector",
                    "model_version": "v1.1.0",
                    "is_active": False,
                    "trained_at": "2026-02-10T10:00:00",
                    "metrics": '{"contamination": 0.05, "n_estimators": 100}',
                },
            ]
        else:
            demo_models = []

        for model in demo_models:
            is_active = model.get("is_active", False)
            st.markdown(
                f"""
                <div style="background:#1C2333; border:1px solid #2D3748; border-radius:8px;
                            padding:0.75rem 1rem; margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong style="color:#E2E8F0;">{model["model_name"]}</strong>
                            <code style="color:#00D4FF; font-size:0.78rem; margin-left:0.5rem;">
                                {model["model_version"]}
                            </code>
                        </div>
                        {'<span class="badge badge-normal">ACTIVE</span>' if is_active else '<span class="badge badge-low">ARCHIVED</span>'}
                    </div>
                    <div style="font-size:0.78rem; color:#718096; margin-top:6px;">
                        Trained: {model["trained_at"][:10]}
                        &nbsp;·&nbsp; Metrics: <code>{model.get("metrics", "{}")[:80]}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_train, col_space = st.columns([1, 2])
        with col_train:
            if st.button("🔄 Train model on AI4I 2020", use_container_width=True):
                if is_demo():
                    st.info("Training task queued (demo mode — Celery not running).")
                else:
                    resp = api_post("/api/v1/workers/train", {"model_type": "anomaly"})
                    st.info("Training task queued via Celery.")

    # ── System tab ────────────────────────────────────────────────────────────
    with tab_system:
        section_title("System Configuration")

        if not is_demo():
            health = api_get("/api/v1/metrics/health") or {}
        else:
            health = {
                "status": "healthy",
                "database": True,
                "redis": False,
                "model_loaded": True,
                "uptime_seconds": 3847.2,
                "version": "0.1.0",
            }

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("API Status", "🟢 " + health.get("status", "unknown").title())
        with k2:
            kpi_card("Database", "🟢 Connected" if health.get("database") else "🔴 Offline")
        with k3:
            kpi_card("Redis", "🟢 Connected" if health.get("redis") else "🔴 Offline")
        with k4:
            uptime = health.get("uptime_seconds", 0)
            hours = int(uptime // 3600)
            mins = int((uptime % 3600) // 60)
            kpi_card("Uptime", f"{hours}h {mins}m")

        section_title("Tenant Info")
        tenant_slug = st.session_state.get("tenant_slug", "unknown")
        st.info(
            f"**Workspace**: `{tenant_slug}`\n\n"
            f"**Version**: {health.get('version', '0.1.0')}\n\n"
            f"**Platform**: HAIIP — NextIndustriAI · Centria RDI"
        )

        if is_demo():
            st.warning(
                "🧪 **Demo Mode** — Running with simulated data. "
                "Configure `.env.local` and connect to the API for full functionality."
            )
