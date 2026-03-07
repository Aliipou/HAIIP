"""Feedback page — human-in-the-loop prediction correction."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_get, api_post, is_demo
from haiip.dashboard.components.demo_data import demo_predictions
from haiip.dashboard.components.theme import badge, kpi_card, section_title

FAILURE_MODES = ["no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def render() -> None:
    st.markdown("## 👤 Human Feedback")
    st.caption(
        "Review AI predictions and submit corrections. "
        "Feedback adjusts model confidence and triggers retraining when accuracy drops."
    )

    # ── Load unverified predictions ───────────────────────────────────────────
    if is_demo():
        predictions = [p for p in demo_predictions(n=15) if not p.get("human_verified")]
    else:
        resp = api_get("/api/v1/predictions", params={"size": 30}) or {}
        all_preds = resp.get("items", [])
        predictions = [p for p in all_preds if not p.get("human_verified")]

    # ── KPI row ───────────────────────────────────────────────────────────────
    fb_resp = api_get("/api/v1/feedback") if not is_demo() else []
    total_fb = len(fb_resp or [])
    correct_fb = sum(1 for f in (fb_resp or []) if f.get("was_correct"))
    accuracy = correct_fb / total_fb if total_fb > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Awaiting Review", str(len(predictions)))
    with k2:
        kpi_card("Feedback Given", str(total_fb if not is_demo() else 47))
    with k3:
        kpi_card("Model Accuracy", f"{(accuracy or 0.943) * 100:.1f}%")
    with k4:
        needs_retrain = accuracy < 0.80 if total_fb > 50 else False
        kpi_card(
            "Retraining Needed",
            "YES ⚠️" if needs_retrain else "No ✓",
            delta_dir="up" if needs_retrain else "neu",
        )

    if not predictions:
        st.success("✅ All predictions have been reviewed. No feedback pending.")
        _show_feedback_history()
        return

    section_title(f"Predictions Awaiting Human Review ({len(predictions)})")

    # ── Feedback cards ────────────────────────────────────────────────────────
    for pred in predictions[:10]:  # show max 10 at a time
        _feedback_card(pred)

    _show_feedback_history()


def _feedback_card(pred: dict) -> None:
    pred_id = pred.get("id", "")
    machine = pred.get("machine_id", "")
    label = pred.get("prediction_label", "no_failure")
    conf = pred.get("confidence", 0.5)
    score = pred.get("anomaly_score", 0.0)
    created = pred.get("created_at", "")[:19]
    s = "anomaly" if label != "no_failure" else "normal"

    with st.container():
        st.markdown(
            f"""
            <div style="background:#1C2333; border:1px solid #2D3748; border-radius:10px;
                        padding:1rem 1.2rem; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="color:#E2E8F0;">{machine}</strong>
                        &nbsp;&nbsp;
                        {badge(label, s)}
                    </div>
                    <div style="font-size:0.8rem; color:#718096;">{created}</div>
                </div>
                <div style="display:flex; gap:2rem; margin-top:0.75rem; font-size:0.82rem; color:#A0AEC0;">
                    <span>Confidence: <strong style="color:#E2E8F0;">{conf * 100:.1f}%</strong></span>
                    <span>Anomaly score: <strong style="color:#E2E8F0;">{score:.3f}</strong></span>
                    <span>ID: <code style="font-size:0.72rem;">{pred_id[:12]}…</code></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_correct, col_wrong, col_label, col_note = st.columns([1, 1, 2, 2])
        with col_correct:
            if st.button("✅ Correct", key=f"correct_{pred_id}", use_container_width=True):
                _submit_feedback(pred_id, was_correct=True, corrected_label=None, notes=None)
        with col_wrong:
            if st.button("❌ Wrong", key=f"wrong_{pred_id}", use_container_width=True):
                st.session_state[f"show_correct_{pred_id}"] = True

        # Show correction form if "Wrong" was clicked
        if st.session_state.get(f"show_correct_{pred_id}"):
            with col_label:
                corrected = st.selectbox(
                    "Correct label",
                    FAILURE_MODES,
                    key=f"clabel_{pred_id}",
                )
            with col_note:
                note = st.text_input(
                    "Notes (optional)",
                    placeholder="Explain why",
                    key=f"note_{pred_id}",
                )
            if st.button("Submit correction", key=f"submit_{pred_id}"):
                _submit_feedback(
                    pred_id,
                    was_correct=False,
                    corrected_label=corrected,
                    notes=note or None,
                )
                st.session_state.pop(f"show_correct_{pred_id}", None)

        st.markdown("---")


def _submit_feedback(
    prediction_id: str,
    was_correct: bool,
    corrected_label: str | None,
    notes: str | None,
) -> None:
    payload = {
        "prediction_id": prediction_id,
        "was_correct": was_correct,
    }
    if corrected_label:
        payload["corrected_label"] = corrected_label
    if notes:
        payload["notes"] = notes

    if is_demo():
        st.success("✅ Feedback recorded (demo mode).")
        st.rerun()
        return

    resp = api_post("/api/v1/feedback", payload)
    if resp:
        st.success("✅ Feedback submitted. Thank you for improving the model.")
        st.rerun()
    else:
        st.error("Failed to submit feedback.")


def _show_feedback_history() -> None:
    """Show recent submitted feedback."""
    if is_demo():
        return

    with st.expander("📋 Recent feedback history"):
        resp = api_get("/api/v1/feedback", params={"limit": 20})
        items = resp or []
        if not items:
            st.info("No feedback submitted yet.")
            return

        import pandas as pd

        df = pd.DataFrame(items)
        display_cols = ["prediction_id", "was_correct", "corrected_label", "created_at"]
        df_show = df[[c for c in display_cols if c in df.columns]]
        df_show["was_correct"] = df_show["was_correct"].apply(lambda x: "✅" if x else "❌")
        st.dataframe(df_show, use_container_width=True, hide_index=True)
