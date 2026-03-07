"""Predictive Maintenance page — failure mode prediction and RUL estimation."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.auth import api_post, is_demo
from haiip.dashboard.components.charts import (
    drift_heatmap,
    failure_mode_pie,
    rul_bar_chart,
)
from haiip.dashboard.components.demo_data import (
    demo_drift_results,
    demo_predictions,
    demo_rul_per_machine,
)
from haiip.dashboard.components.theme import badge, kpi_card, section_title


def render() -> None:
    st.markdown("## 🔧 Predictive Maintenance")
    st.caption(
        "AI-powered failure mode classification (TWF, HDF, PWF, OSF, RNF) "
        "and Remaining Useful Life estimation."
    )

    tab_fleet, tab_single, tab_drift = st.tabs(
        ["Fleet Overview", "Single Machine Analysis", "Concept Drift"]
    )

    # ── Fleet tab ─────────────────────────────────────────────────────────────
    with tab_fleet:
        rul_data = demo_rul_per_machine() if is_demo() else {}
        predictions = demo_predictions() if is_demo() else []

        section_title("Remaining Useful Life — All Machines")

        if rul_data:
            machines = list(rul_data.keys())
            ruls = list(rul_data.values())
            rul_bar_chart(machines, ruls)

            # Critical machines
            critical = [(m, r) for m, r in rul_data.items() if r < 50]
            if critical:
                st.warning(
                    f"⚠️ **{len(critical)} machine(s) approaching failure**: "
                    + ", ".join(f"**{m}** ({r} cycles)" for m, r in critical)
                )

        section_title("Failure Mode Distribution")
        col_l, col_r = st.columns([1, 1])
        with col_l:
            from collections import Counter

            counts = Counter(p.get("prediction_label", "no_failure") for p in predictions)
            labels = ["no_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
            display = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
            values = [counts.get(k, 0) for k in labels]
            if any(v > 0 for v in values):
                failure_mode_pie(display, values)
            else:
                st.info("No predictions available yet.")

        with col_r:
            section_title("Failure Mode Guide")
            for code, desc in [
                ("TWF", "Tool Wear Failure — excessive tool wear beyond threshold"),
                ("HDF", "Heat Dissipation Failure — insufficient cooling"),
                ("PWF", "Power Failure — torque×speed product too high"),
                ("OSF", "Overstrain Failure — product of tool wear and torque"),
                ("RNF", "Random Failure — stochastic, no clear cause"),
            ]:
                st.markdown(
                    f"**{code}** — <span style='color:#A0AEC0;'>{desc}</span>",
                    unsafe_allow_html=True,
                )

        # Predictions table
        section_title("Recent Predictions")
        if predictions:
            import pandas as pd

            df = pd.DataFrame(predictions)
            display_cols = [
                "machine_id",
                "prediction_label",
                "confidence",
                "rul_cycles",
                "human_verified",
                "created_at",
            ]
            df_display = df[[c for c in display_cols if c in df.columns]].copy()
            df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x * 100:.1f}%")
            df_display["human_verified"] = df_display["human_verified"].apply(
                lambda x: "✅" if x else "⬜"
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ── Single Machine Analysis tab ───────────────────────────────────────────
    with tab_single:
        section_title("Run Maintenance Prediction")
        st.caption("Enter sensor readings to get failure mode + RUL prediction.")

        col_a, col_b = st.columns(2)
        with col_a:
            machine_id = st.text_input("Machine ID", value="CNC-001")
            air_temp = st.number_input(
                "Air Temperature (K)",
                value=300.0,
                min_value=250.0,
                max_value=400.0,
                step=0.1,
            )
            proc_temp = st.number_input(
                "Process Temperature (K)",
                value=310.0,
                min_value=250.0,
                max_value=500.0,
                step=0.1,
            )
        with col_b:
            rpm = st.number_input(
                "Rotational Speed (RPM)",
                value=1538.0,
                min_value=0.0,
                max_value=5000.0,
                step=10.0,
            )
            torque = st.number_input(
                "Torque (Nm)", value=40.0, min_value=0.0, max_value=200.0, step=0.5
            )
            tool_wear = st.number_input(
                "Tool Wear (min)", value=100.0, min_value=0.0, max_value=253.0, step=1.0
            )

        if st.button("🔍 Predict Failure Mode", use_container_width=True):
            _run_single_prediction(machine_id, air_temp, proc_temp, rpm, torque, tool_wear)

    # ── Drift tab ─────────────────────────────────────────────────────────────
    with tab_drift:
        section_title("Concept Drift Monitor")
        st.caption(
            "Population Stability Index (PSI) per feature. "
            "PSI > 0.2 indicates significant distribution shift — retraining required."
        )

        drift_results = demo_drift_results() if is_demo() else []

        if drift_results:
            feature_names = [d["feature"] for d in drift_results]
            psi_values = [d["psi"] for d in drift_results]
            drift_heatmap(feature_names, psi_values)

            # Status summary
            drifted = [d for d in drift_results if d["severity"] == "drift"]
            monitoring = [d for d in drift_results if d["severity"] == "monitoring"]
            stable = [d for d in drift_results if d["severity"] == "stable"]

            d1, d2, d3 = st.columns(3)
            with d1:
                kpi_card(
                    "Drifting Features",
                    str(len(drifted)),
                    delta_dir="up" if drifted else "neu",
                )
            with d2:
                kpi_card("Monitoring", str(len(monitoring)))
            with d3:
                kpi_card("Stable", str(len(stable)), delta_dir="down")

            if drifted:
                st.error(
                    f"⚠️ Drift detected on: {', '.join(d['feature'] for d in drifted)}. "
                    "Consider triggering model retraining."
                )
                if st.button("🔄 Trigger Retraining"):
                    st.info(
                        "Retraining task queued via Celery worker. Monitor progress in Audit Trail."
                    )
        else:
            st.info("No drift data available. Run a drift check from the workers.")


def _run_single_prediction(
    machine_id: str,
    air_temp: float,
    proc_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
) -> None:
    """Run prediction and display results."""
    if is_demo():
        from haiip.core.maintenance import MaintenancePredictor

        predictor = MaintenancePredictor()
        result_data = predictor.predict([air_temp, proc_temp, rpm, torque, tool_wear])
    else:
        payload = {
            "readings": [
                {
                    "machine_id": machine_id,
                    "air_temperature": air_temp,
                    "process_temperature": proc_temp,
                    "rotational_speed": rpm,
                    "torque": torque,
                    "tool_wear": tool_wear,
                }
            ],
            "model_type": "predictive_maintenance",
        }
        resp = api_post("/api/v1/predict/batch", payload)
        result_data = (resp or {}).get("data", [{}])[0] if resp else {}

    label = result_data.get("label", result_data.get("prediction_label", "no_failure"))
    confidence = result_data.get("confidence", 0.5)
    rul = result_data.get("rul_cycles")
    fail_prob = result_data.get("failure_probability", 0.0)
    explanation = result_data.get("explanation", {})

    st.markdown("---")
    section_title("Prediction Result")

    r1, r2, r3, r4 = st.columns(4)
    s = "anomaly" if label != "no_failure" else "normal"
    with r1:
        st.markdown(f"**Failure Mode**<br>{badge(label, s)}", unsafe_allow_html=True)
    with r2:
        kpi_card("Confidence", f"{confidence * 100:.1f}%")
    with r3:
        kpi_card("Failure Prob.", f"{fail_prob * 100:.1f}%")
    with r4:
        kpi_card("RUL", f"{rul} cycles" if rul is not None else "N/A")

    if explanation:
        top_feats = explanation.get("top_features", {})
        if top_feats:
            st.markdown("**Top contributing features:**")
            for feat, importance in top_feats.items():
                st.markdown(
                    f"- `{feat}`: importance = {importance:.4f}",
                    unsafe_allow_html=False,
                )

    if label != "no_failure":
        st.error(
            f"⚠️ **{label} failure predicted** with {confidence * 100:.1f}% confidence. "
            "Schedule maintenance immediately."
        )
    else:
        st.success(f"✅ Machine operating normally. Confidence: {confidence * 100:.1f}%")
