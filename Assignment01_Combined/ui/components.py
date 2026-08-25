from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from .theme import TOKENS


def inject_global_styles():
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {TOKENS["background"]};
            --surface: {TOKENS["surface"]};
            --surface-2: {TOKENS["surface_elevated"]};
            --border: {TOKENS["border"]};
            --text: {TOKENS["primary_text"]};
            --muted: {TOKENS["secondary_text"]};
            --accent: {TOKENS["accent"]};
            --accent-2: {TOKENS["accent_secondary"]};
            --success: {TOKENS["success"]};
            --warning: {TOKENS["warning"]};
            --danger: {TOKENS["danger"]};
            --radius: {TOKENS["radius"]};
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: radial-gradient(circle at top left, rgba(56,189,248,0.10), transparent 34rem),
                        radial-gradient(circle at 85% 10%, rgba(129,140,248,0.08), transparent 28rem),
                        var(--bg);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        [data-testid="stHeader"] {{ background: rgba(11,15,23,0.55); backdrop-filter: blur(16px); }}
        [data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(11,15,23,0.98));
            border-right: 1px solid var(--border);
        }}
        .block-container {{
            max-width: 1220px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3 {{ letter-spacing: 0; color: var(--text); }}
        p, li, label, .stMarkdown {{ color: var(--muted); }}
        div[data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
        }}
        div[data-testid="stMetric"] label {{ color: var(--muted) !important; }}
        .stButton > button {{
            min-height: 46px;
            border-radius: 14px;
            border: 1px solid rgba(56,189,248,0.35);
            background: linear-gradient(135deg, #2563EB, #38BDF8);
            color: white;
            font-weight: 750;
            width: 100%;
            transition: transform 160ms ease, filter 160ms ease;
        }}
        .stButton > button:hover {{
            filter: brightness(1.08);
            transform: translateY(-1px);
            border-color: rgba(56,189,248,0.65);
            color: white;
        }}
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {{
            background: rgba(15,23,42,0.82) !important;
            color: var(--text) !important;
            border-color: rgba(255,255,255,0.10) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpander"] {{
            background: rgba(17,24,39,0.72);
            border: 1px solid var(--border);
            border-radius: 16px;
        }}
        .hero {{
            padding: 30px 28px;
            border: 1px solid var(--border);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(21,28,43,0.96), rgba(17,24,39,0.72));
            box-shadow: 0 18px 55px rgba(0,0,0,0.22);
            margin-bottom: 22px;
        }}
        .eyebrow {{
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .55rem;
        }}
        .hero-title {{
            color: var(--text);
            font-size: clamp(2rem, 4vw, 2.7rem);
            line-height: 1.06;
            font-weight: 850;
            margin: 0 0 .75rem 0;
        }}
        .hero-subtitle {{
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 760px;
            line-height: 1.62;
            margin: 0;
        }}
        .card {{
            background: rgba(17,24,39,0.84);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            min-height: 100%;
            box-shadow: 0 12px 36px rgba(0,0,0,0.16);
        }}
        .card-title {{
            color: var(--muted);
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 800;
            margin-bottom: 9px;
        }}
        .card-value {{
            color: var(--text);
            font-size: 1.55rem;
            line-height: 1.15;
            font-weight: 820;
            margin-bottom: 6px;
        }}
        .card-caption {{ color: var(--muted); font-size: .88rem; line-height: 1.45; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 14px;
            margin: 14px 0 20px;
        }}
        .metric-card {{
            background: rgba(17,24,39,0.88);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
        }}
        .metric-label {{ color: var(--muted); font-size: .78rem; font-weight: 750; text-transform: uppercase; letter-spacing: .06em; }}
        .metric-value {{ color: var(--text); font-size: 1.85rem; font-weight: 850; margin-top: 8px; }}
        .metric-help {{ color: var(--muted); font-size: .78rem; margin-top: 6px; }}
        .badge-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
        .badge {{
            display: inline-flex; align-items: center; gap: 8px;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 7px 11px;
            background: rgba(15,23,42,0.74);
            color: var(--text);
            font-weight: 720;
            font-size: .82rem;
        }}
        .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: var(--accent); }}
        .dot.success {{ background: var(--success); }}
        .dot.warning {{ background: var(--warning); }}
        .dot.danger {{ background: var(--danger); }}
        .section-title {{
            margin: 30px 0 12px;
            color: var(--text);
            font-size: 1.38rem;
            font-weight: 840;
        }}
        .soft-note {{
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.18);
            border-radius: 16px;
            padding: 14px 16px;
            color: #BAE6FD;
            margin: 12px 0 18px;
        }}
        .result-card {{
            border-radius: 20px;
            padding: 24px;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(17,24,39,0.96), rgba(21,28,43,0.78));
            margin: 18px 0;
        }}
        .result-card.success {{ border-color: rgba(52,211,153,0.38); }}
        .result-card.danger {{ border-color: rgba(251,113,133,0.38); }}
        .result-kicker {{ color: var(--muted); font-size: .78rem; font-weight: 850; letter-spacing: .09em; text-transform: uppercase; }}
        .result-main {{ color: var(--text); font-size: clamp(2.2rem, 7vw, 4rem); font-weight: 900; line-height: 1; margin: 10px 0; }}
        .result-sub {{ color: var(--muted); font-size: .96rem; }}
        .progress-track {{ width: 100%; height: 9px; background: rgba(255,255,255,.08); border-radius: 999px; overflow: hidden; margin-top: 12px; }}
        .progress-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
        .graph-shell {{
            border: 1px solid var(--border);
            border-radius: 20px;
            background: rgba(17,24,39,0.72);
            padding: 18px;
            overflow-x: auto;
        }}
        .graph-lanes {{
            display: grid;
            grid-template-columns: repeat(5, minmax(140px, 1fr));
            gap: 16px;
            align-items: stretch;
            min-width: 680px;
        }}
        .graph-lane-title {{ color: var(--muted); font-size: .74rem; font-weight: 850; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px; }}
        .graph-node {{
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 11px 12px;
            color: var(--text);
            background: rgba(15,23,42,0.82);
            margin-bottom: 9px;
            transition: border-color .15s ease, transform .15s ease;
        }}
        .graph-node:hover {{ border-color: rgba(56,189,248,.55); transform: translateY(-1px); }}
        .graph-node.model {{ box-shadow: inset 3px 0 0 var(--accent); }}
        .graph-node.feature {{ box-shadow: inset 3px 0 0 var(--accent-2); }}
        .graph-node.target {{ box-shadow: inset 3px 0 0 var(--success); }}
        .graph-node.outcome {{ box-shadow: inset 3px 0 0 var(--warning); }}
        .graph-node.representation {{ box-shadow: inset 3px 0 0 #A78BFA; }}
        .recent-card {{
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(17,24,39,0.78);
            margin-bottom: 10px;
        }}
        .recent-top {{ display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:.78rem; }}
        .recent-main {{ color:var(--text); font-weight:800; margin-top:6px; }}
        .recent-sub {{ color:var(--muted); margin-top:4px; font-size:.86rem; }}
        .footer {{ color: var(--muted); text-align:center; margin: 34px 0 8px; font-size:.86rem; }}
        @media (max-width: 768px) {{
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
            .hero {{ padding: 22px 18px; }}
            .metric-grid {{ grid-template-columns: 1fr; }}
            .graph-lanes {{ grid-template-columns: 1fr; min-width: 0; }}
            .recent-top {{ flex-direction: column; gap: 2px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _attrs(items: dict[str, str] | None = None) -> str:
    if not items:
        return ""
    return " ".join(f'{escape(k)}="{escape(str(v))}"' for k, v in items.items())


def render_page_header(eyebrow: str, title: str, subtitle: str, badges: list[str] | None = None):
    badge_html = ""
    if badges:
        badge_html = "<div class='badge-row'>" + "".join(f"<span class='badge'>{escape(badge)}</span>" for badge in badges) + "</div>"
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">{escape(eyebrow)}</div>
            <h1 class="hero-title">{escape(title)}</h1>
            <p class="hero-subtitle">{escape(subtitle)}</p>
            {badge_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, status: str, kind: str = "success"):
    st.markdown(
        f"<span class='badge'><span class='dot {escape(kind)}'></span>{escape(label)} · {escape(status)}</span>",
        unsafe_allow_html=True,
    )


def render_model_badge(is_final: bool):
    if is_final:
        st.markdown("<span class='badge'><span class='dot success'></span>Scientific Final Model</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge'><span class='dot'></span>Comparison Model</span>", unsafe_allow_html=True)


def render_section_header(title: str, caption: str | None = None):
    caption_html = f"<div class='card-caption'>{escape(caption)}</div>" if caption else ""
    st.markdown(f"<div class='section-title'>{escape(title)}</div>{caption_html}", unsafe_allow_html=True)


def render_info_banner(text: str):
    st.markdown(f"<div class='soft-note'>{escape(text)}</div>", unsafe_allow_html=True)


def render_card(title: str, value: str, caption: str = ""):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{escape(title)}</div>
            <div class="card-value">{escape(value)}</div>
            <div class="card-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[tuple[str, str, str]]):
    cards = []
    for label, value, help_text in metrics:
        cards.append(
            f"""
            <div class="metric-card">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{escape(value)}</div>
                <div class="metric-help">{escape(help_text)}</div>
            </div>
            """
        )
    st.markdown("<div class='metric-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_prediction_card(kicker: str, main: str, sub: str, status: str = "success", progress: float | None = None):
    progress_html = ""
    if progress is not None:
        pct = max(0, min(100, progress * 100))
        progress_html = f"<div class='progress-track'><div class='progress-fill' style='width:{pct:.1f}%'></div></div>"
    st.markdown(
        f"""
        <div class="result-card {escape(status)}">
            <div class="result-kicker">{escape(kicker)}</div>
            <div class="result-main">{escape(main)}</div>
            <div class="result-sub">{escape(sub)}</div>
            {progress_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_schema_graph(lanes: list[tuple[str, list[tuple[str, str]]]]):
    lane_html = []
    for title, nodes in lanes:
        node_html = "".join(
            f"<div class='graph-node {escape(group)}'>{escape(name)}</div>"
            for name, group in nodes
        )
        lane_html.append(
            f"<div><div class='graph-lane-title'>{escape(title)}</div>{node_html}</div>"
        )
    st.markdown(
        "<div class='graph-shell'><div class='graph-lanes'>" + "".join(lane_html) + "</div></div>",
        unsafe_allow_html=True,
    )


def render_graph_legend(items: list[tuple[str, str]]):
    html = "<div class='badge-row'>" + "".join(
        f"<span class='badge'><span class='dot {escape(kind)}'></span>{escape(label)}</span>"
        for label, kind in items
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_recent_prediction_card(time_label: str, model: str, prediction: str, detail: str):
    st.markdown(
        f"""
        <div class="recent-card">
            <div class="recent-top"><span>{escape(time_label)}</span><span>{escape(model)}</span></div>
            <div class="recent-main">{escape(prediction)}</div>
            <div class="recent-sub">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, detail: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-value">{escape(title)}</div>
            <div class="card-caption">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown("<div class='footer'>Assignment 01 · Intelligent System Development</div>", unsafe_allow_html=True)


def format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def format_decimal(value: float) -> str:
    return f"{float(value):.4f}"


def format_time(value) -> str:
    if not value:
        return "Recent"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d %b · %H:%M")
    except ValueError:
        return text[:16]
