"""
Modern dark UI theme for the ML Trainer app.
Inject via apply_theme() once at app startup.
All helper functions return raw HTML strings.
"""
import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0b0f19",
    "surface":   "#111827",
    "surface2":  "#1a2236",
    "border":    "rgba(255,255,255,0.06)",
    "border2":   "rgba(255,255,255,0.10)",
    "primary":   "#6366f1",
    "secondary": "#06b6d4",
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "error":     "#ef4444",
    "violet":    "#8b5cf6",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
    "muted2":    "#94a3b8",
}

CSS = f"""
<style>
/* ── Google Font ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}

.stApp {{
    background: {C["bg"]} !important;
}}

[data-testid="stAppViewContainer"] > .main {{
    background: {C["bg"]};
    padding-top: 0 !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: 1px solid {C["border"]} !important;
}}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {C["surface"]} !important;
    border-right: 1px solid {C["border2"]} !important;
}}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label {{
    color: {C["muted2"]} !important;
    font-size: 13px !important;
}}

[data-testid="stSidebar"] h1 {{
    background: linear-gradient(135deg, {C["primary"]}, {C["violet"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 22px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {C["text"]} !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-top: 4px !important;
}}

/* ── Main text ────────────────────────────────────────────────────────────── */
h1, h2, h3 {{ color: {C["text"]} !important; }}

p, li, .stMarkdown {{
    color: {C["muted2"]};
    font-size: 14px;
}}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["surface"]} !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 2px !important;
    border: 1px solid {C["border2"]} !important;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    color: {C["muted"]} !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    background: transparent !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: {C["text"]} !important;
    background: {C["surface2"]} !important;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {C["primary"]}, {C["violet"]}) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {C["primary"]} 0%, {C["violet"]} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
    letter-spacing: 0.01em !important;
}}

[data-testid="stButton"] > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.45) !important;
    filter: brightness(1.1) !important;
}}

[data-testid="stButton"] > button:active {{
    transform: translateY(0px) !important;
}}

/* Secondary / download buttons */
[data-testid="stDownloadButton"] > button {{
    background: {C["surface2"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border2"]} !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stDownloadButton"] > button:hover {{
    border-color: {C["primary"]} !important;
    color: {C["primary"]} !important;
    transform: translateY(-1px) !important;
}}

/* ── Inputs ───────────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    background: {C["surface2"]} !important;
    border: 1px solid {C["border2"]} !important;
    border-radius: 10px !important;
    color: {C["text"]} !important;
    font-size: 13px !important;
}}

[data-testid="stNumberInput"] > div {{
    background: {C["surface2"]} !important;
    border: 1px solid {C["border2"]} !important;
    border-radius: 10px !important;
}}

[data-testid="stTextInput"] > div > div > input {{
    background: {C["surface2"]} !important;
    border: 1px solid {C["border2"]} !important;
    border-radius: 10px !important;
    color: {C["text"]} !important;
    font-size: 13px !important;
}}

[data-testid="stTextInput"] > div > div > input:focus {{
    border-color: {C["primary"]} !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}}

/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background: {C["primary"]} !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.2) !important;
}}

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border2"]} !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s !important;
}}

[data-testid="metric-container"]:hover {{
    border-color: {C["primary"]}60 !important;
}}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    color: {C["muted"]} !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C["text"]} !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}}

/* ── Expanders ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border2"]} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}}

[data-testid="stExpander"] summary {{
    color: {C["muted2"]} !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
}}

[data-testid="stExpander"] summary:hover {{
    color: {C["text"]} !important;
}}

/* ── Alerts ───────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    font-size: 13px !important;
}}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] > div {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid {C["border2"]} !important;
}}

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] > div {{
    background: {C["surface"]} !important;
    border: 2px dashed {C["border2"]} !important;
    border-radius: 14px !important;
    transition: all 0.2s !important;
}}

[data-testid="stFileUploader"] > div:hover {{
    border-color: {C["primary"]} !important;
    background: rgba(99,102,241,0.04) !important;
}}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: {C["surface2"]};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {C["muted"]}; }}

/* ── Radio buttons ────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label {{
    color: {C["muted2"]} !important;
    font-size: 13px !important;
}}

[data-testid="stRadio"] [data-baseweb="radio"] div:first-child {{
    border-color: {C["border2"]} !important;
    background: transparent !important;
}}

/* ── Caption text ─────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {{
    color: {C["muted"]} !important;
    font-size: 12px !important;
}}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {{
    color: {C["muted2"]} !important;
}}

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr {{
    border-color: {C["border"]} !important;
    margin: 20px 0 !important;
}}

/* ── Plotly charts transparent bg ────────────────────────────────────────── */
.js-plotly-plot .plotly {{
    border-radius: 12px;
}}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


# ── HTML component helpers ────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="
        padding: 32px 0 24px 0;
        border-bottom: 1px solid {C['border']};
        margin-bottom: 28px;
    ">
        <h1 style="
            background: linear-gradient(135deg, {C['text']} 40%, {C['muted2']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 30px;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        ">{title}</h1>
        {f'<p style="color:{C["muted"]};font-size:14px;margin:6px 0 0 0">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def section_title(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin: 24px 0 16px 0">
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <div style="
                width: 3px;
                height: 20px;
                background: linear-gradient(180deg, {C['primary']}, {C['violet']});
                border-radius: 2px;
            "></div>
            <span style="
                color: {C['text']};
                font-size: 16px;
                font-weight: 600;
                letter-spacing: -0.2px;
            ">{title}</span>
        </div>
        {f'<p style="color:{C["muted"]};font-size:13px;margin:6px 0 0 14px">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def metric_card(title: str, value: str, subtitle: str = "", accent: str = None):
    color = accent or C["primary"]
    st.markdown(f"""
    <div style="
        background: {C['surface']};
        border: 1px solid {C['border2']};
        border-radius: 16px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    ">
        <div style="
            position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, {color}, {color}55);
        "></div>
        <div style="
            font-size: 11px;
            color: {C['muted']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 8px;
        ">{title}</div>
        <div style="
            font-size: 34px;
            font-weight: 800;
            color: {C['text']};
            letter-spacing: -1px;
            line-height: 1;
        ">{value}</div>
        {f'<div style="font-size:12px;color:{C["muted"]};margin-top:6px">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def metrics_row(items: list[dict]):
    """items: list of {title, value, subtitle?, accent?}"""
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(
                item["title"], item["value"],
                item.get("subtitle", ""), item.get("accent")
            )


def status_badge(label: str, kind: str = "info"):
    colors = {
        "success": (C["success"],  "rgba(16,185,129,0.12)"),
        "warning": (C["warning"],  "rgba(245,158,11,0.12)"),
        "error":   (C["error"],    "rgba(239,68,68,0.12)"),
        "info":    (C["secondary"],"rgba(6,182,212,0.12)"),
        "primary": (C["primary"],  "rgba(99,102,241,0.12)"),
    }
    fg, bg = colors.get(kind, colors["info"])
    st.markdown(f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {bg};
        color: {fg};
        border: 1px solid {fg}40;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    ">
        <span style="width:6px;height:6px;border-radius:50%;background:{fg};display:inline-block"></span>
        {label}
    </span>
    """, unsafe_allow_html=True)


def card(content_fn, padding: str = "24px"):
    """Wrap a block of Streamlit content in a styled card."""
    st.markdown(f"""
    <div style="
        background: {C['surface']};
        border: 1px solid {C['border2']};
        border-radius: 16px;
        padding: {padding};
        margin-bottom: 16px;
    ">
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str):
    st.markdown(f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 64px 32px;
        text-align: center;
    ">
        <div style="font-size: 48px; margin-bottom: 16px; opacity: 0.6">{icon}</div>
        <div style="
            color: {C['text']};
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        ">{title}</div>
        <div style="
            color: {C['muted']};
            font-size: 14px;
            max-width: 320px;
            line-height: 1.6;
        ">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def divider():
    st.markdown(
        f'<div style="height:1px;background:{C["border"]};margin:24px 0"></div>',
        unsafe_allow_html=True
    )


def tag(text: str, color: str = None):
    c = color or C["primary"]
    return (f'<span style="background:{c}20;color:{c};border:1px solid {c}40;'
            f'border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600">{text}</span>')
