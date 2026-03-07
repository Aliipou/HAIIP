"""ROI Calculator — SME business case for HAIIP deployment."""

from __future__ import annotations

import streamlit as st

from haiip.dashboard.components.theme import kpi_card, section_title


def render() -> None:
    st.markdown("## 💶 SME ROI Calculator")
    st.caption(
        "Calculate the business value of deploying HAIIP at your production site. "
        "Based on industry benchmarks from Nordic manufacturing SMEs."
    )

    st.markdown("---")

    # ── Input parameters ──────────────────────────────────────────────────────
    section_title("Your Production Parameters")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Production Profile**")
        n_machines = st.slider("Number of machines", 1, 50, 6)
        shifts_per_day = st.slider("Shifts per day", 1, 3, 2)
        production_days = st.slider("Production days per year", 200, 365, 250)
        hourly_output = st.number_input(
            "Production value per machine-hour (€)",
            value=850,
            min_value=50,
            max_value=10000,
            step=50,
        )

    with col_b:
        st.markdown("**Current Failure Profile**")
        unplanned_stops_year = st.slider("Unplanned stops per machine per year", 1, 30, 8)
        avg_downtime_hours = st.slider("Average downtime per stop (hours)", 1, 48, 6)
        repair_cost_per_stop = st.number_input(
            "Avg repair cost per stop (€)",
            value=3200,
            min_value=100,
            max_value=50000,
            step=100,
        )
        tool_changes_year = st.slider("Reactive tool changes per machine per year", 0, 50, 12)
        tool_cost = st.number_input(
            "Cost per emergency tool change (€)",
            value=420,
            min_value=10,
            max_value=5000,
            step=10,
        )

    with st.expander("⚙️ HAIIP Performance Assumptions (adjustable)"):
        col_c, col_d = st.columns(2)
        with col_c:
            downtime_reduction = st.slider(
                "Unplanned downtime reduction (%)",
                20,
                80,
                45,
                help="Industry benchmark: 35–55% reduction with predictive maintenance",
            )
            tool_waste_reduction = st.slider("Emergency tool change reduction (%)", 20, 70, 40)
        with col_d:
            detection_accuracy = st.slider("Model detection accuracy (%)", 80, 99, 94)
            implementation_cost = st.number_input(
                "HAIIP implementation cost (€/year)",
                value=18000,
                min_value=1000,
                step=1000,
            )

    # ── Calculations ──────────────────────────────────────────────────────────
    # Current costs
    current_downtime_hours = n_machines * unplanned_stops_year * avg_downtime_hours
    current_downtime_loss = current_downtime_hours * hourly_output
    current_repair_cost = n_machines * unplanned_stops_year * repair_cost_per_stop
    current_tool_cost = n_machines * tool_changes_year * tool_cost
    total_current_cost = current_downtime_loss + current_repair_cost + current_tool_cost

    # HAIIP savings
    saved_downtime_loss = current_downtime_loss * (downtime_reduction / 100)
    saved_repair_cost = current_repair_cost * (downtime_reduction / 100)
    saved_tool_cost = current_tool_cost * (tool_waste_reduction / 100)
    total_savings = saved_downtime_loss + saved_repair_cost + saved_tool_cost

    net_benefit = total_savings - implementation_cost
    roi_pct = (net_benefit / implementation_cost * 100) if implementation_cost > 0 else 0
    payback_months = (implementation_cost / (total_savings / 12)) if total_savings > 0 else 999

    # OEE improvement estimate
    total_production_hours = n_machines * shifts_per_day * 8 * production_days
    current_oee_loss_pct = (current_downtime_hours / total_production_hours) * 100
    new_oee_loss_pct = current_oee_loss_pct * (1 - downtime_reduction / 100)
    oee_improvement = current_oee_loss_pct - new_oee_loss_pct

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    section_title("Financial Impact")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi_card(
            "Annual Savings",
            f"€{total_savings:,.0f}",
            delta=f"vs €{total_current_cost:,.0f} current cost",
            delta_dir="down",
        )
    with r2:
        kpi_card(
            "Net Benefit",
            f"€{net_benefit:,.0f}",
            delta_dir="down" if net_benefit > 0 else "up",
        )
    with r3:
        kpi_card(
            "ROI",
            f"{roi_pct:.0f}%",
            delta="First year",
            delta_dir="down" if roi_pct > 0 else "up",
        )
    with r4:
        kpi_card(
            "Payback Period",
            f"{payback_months:.1f} months" if payback_months < 100 else "N/A",
        )

    section_title("Cost Breakdown")

    col_curr, col_haiip = st.columns(2)

    with col_curr:
        st.markdown("**📊 Current Annual Costs**")
        rows = [
            ("Downtime production loss", current_downtime_loss),
            ("Repair & maintenance", current_repair_cost),
            ("Emergency tool changes", current_tool_cost),
        ]
        for name, cost in rows:
            pct = cost / total_current_cost * 100 if total_current_cost > 0 else 0
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; padding:0.5rem 0;
                            border-bottom:1px solid #2D3748; font-size:0.85rem;">
                    <span style="color:#A0AEC0;">{name}</span>
                    <span style="color:#E2E8F0;">€{cost:,.0f}
                        <span style="color:#718096; font-size:0.75rem;">({pct:.0f}%)</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; padding:0.6rem 0;
                        font-weight:700; font-size:0.9rem;">
                <span style="color:#E2E8F0;">Total</span>
                <span style="color:#FF4444;">€{total_current_cost:,.0f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_haiip:
        st.markdown("**✅ With HAIIP**")
        rows_saved = [
            ("Downtime production loss", current_downtime_loss - saved_downtime_loss),
            ("Repair & maintenance", current_repair_cost - saved_repair_cost),
            ("Emergency tool changes", current_tool_cost - saved_tool_cost),
            ("HAIIP platform cost", implementation_cost),
        ]
        new_total = sum(c for _, c in rows_saved)
        for name, cost in rows_saved:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; padding:0.5rem 0;
                            border-bottom:1px solid #2D3748; font-size:0.85rem;">
                    <span style="color:#A0AEC0;">{name}</span>
                    <span style="color:#E2E8F0;">€{cost:,.0f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; padding:0.6rem 0;
                        font-weight:700; font-size:0.9rem;">
                <span style="color:#E2E8F0;">Total</span>
                <span style="color:#00C851;">€{new_total:,.0f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── OEE & operational KPIs ────────────────────────────────────────────────
    section_title("Operational Impact")
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        kpi_card("OEE Improvement", f"+{oee_improvement:.1f}pp")
    with o2:
        kpi_card("Downtime Reduced", f"{downtime_reduction}%")
    with o3:
        saved_stops = int(n_machines * unplanned_stops_year * downtime_reduction / 100)
        kpi_card("Stops Prevented/yr", str(saved_stops))
    with o4:
        kpi_card("Detection Accuracy", f"{detection_accuracy}%")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("📥 Export report (CSV)", use_container_width=False):
        import pandas as pd

        report = pd.DataFrame(
            {
                "Metric": [
                    "Machines",
                    "Annual savings (€)",
                    "Net benefit (€)",
                    "ROI (%)",
                    "Payback (months)",
                    "OEE improvement (pp)",
                    "Implementation cost (€)",
                ],
                "Value": [
                    n_machines,
                    total_savings,
                    net_benefit,
                    roi_pct,
                    payback_months,
                    oee_improvement,
                    implementation_cost,
                ],
            }
        )
        csv = report.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="haiip_roi_report.csv",
            mime="text/csv",
        )
