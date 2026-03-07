"""Global theme and CSS for the HAIIP dashboard."""

import streamlit as st

THEME_CSS = """
<style>
:root {
    --primary:    #00D4FF;
    --success:    #00C851;
    --warning:    #FFB400;
    --danger:     #FF4444;
    --bg:         #0E1117;
    --surface:    #1C2333;
    --surface2:   #242C3E;
    --border:     #2D3748;
    --text:       #E2E8F0;
    --text-muted: #718096;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.kpi-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}
.kpi-delta { font-size: 0.8rem; margin-top: 0.25rem; font-weight: 500; }
.kpi-delta.up   { color: var(--danger); }
.kpi-delta.down { color: var(--success); }
.kpi-delta.neu  { color: var(--text-muted); }
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-critical { background: #FF4444; color: #fff; }
.badge-high     { background: #FF8C00; color: #fff; }
.badge-medium   { background: #FFB400; color: #000; }
.badge-low      { background: #2D3748; color: #CBD5E0; }
.badge-normal   { background: #00C851; color: #fff; }
.badge-anomaly  { background: #FF4444; color: #fff; }
.badge-primary  { background: #00D4FF; color: #000; }
.badge-warning  { background: #FFB400; color: #000; }
.badge-danger   { background: #FF4444; color: #fff; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.dot-green  { background: var(--success); box-shadow: 0 0 6px var(--success); }
.dot-red    { background: var(--danger);  box-shadow: 0 0 6px var(--danger); }
.dot-amber  { background: var(--warning); box-shadow: 0 0 6px var(--warning); }
.dot-blue   { background: var(--primary); box-shadow: 0 0 6px var(--primary); }
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    border-left: 3px solid var(--primary);
    padding-left: 0.6rem;
    margin: 1.5rem 0 0.75rem;
}
.machine-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}
.stButton > button { border-radius: 8px; font-weight: 600; }
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", delta_dir: str = "neu") -> None:
    delta_html = f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def badge(label: str, level: str = "normal") -> str:
    return f'<span class="badge badge-{level}">{label}</span>'


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def status_dot(color: str = "green") -> str:
    return f'<span class="dot dot-{color}"></span>'
