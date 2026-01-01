 # dashboard_langgraph_app.py
# ✅ Updated dashboard to give REAL LLM value:
# - When "Use LLM" is ON, extra_notes is passed raw so LLM can interpret (no hardcoded phrase rules).
# - Deterministic extra_note_flags parsing is OPTIONAL (debug-only).
# - Ticker mentions in notes require explicit confirmation before becoming excluded_assets.
# - Works with portfolio_langgraph.py + llm_client.py where action set_objective_key is allowed.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ✅ IMPORTANT: adjust this import to your real file name
# from portfolio_langgraph_withllm import run_graph
from portfolio_langgraph_withllm import run_graph

DATA_DIR = Path("data/processed_yahoo")


@st.cache_data
def load_available_tickers() -> list[str]:
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    return list(map(str, summary.index))


def _safe_normalize_current_inputs(df: pd.DataFrame, mode: str) -> Optional[dict[str, float]]:
    if df is None or df.empty or "Value" not in df.columns:
        return None

    x = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if mode == "Percent (%)":
        x = x / 100.0

    s = float(x.sum())
    if s <= 0:
        return None

    w = x / s
    return {str(t): float(v) for t, v in w.items()}


def _extract_weights_and_metrics(graph_state: Dict[str, Any]):
    optimization_result = graph_state.get("optimization_result") or {}
    objective_key = graph_state.get("objective_key", "maxsharpe")

    weights_series = None
    portfolio_metrics = None

    if optimization_result and objective_key in optimization_result:
        port = optimization_result[objective_key]
        w = pd.Series(port.get("weights", {}), dtype=float)
        w = w[w.abs() > 1e-6].sort_values(ascending=False)
        weights_series = w

        portfolio_metrics = {
            "objective_key": objective_key,
            "return": float(port.get("return", np.nan)),
            "vol": float(port.get("vol", np.nan)),
            "sharpe": port.get("sharpe", None),
            "used_assets": int(len(w)),
            "universe_assets": int(len(graph_state.get("selected_tickers", []))),
        }

    return optimization_result, objective_key, weights_series, portfolio_metrics


def _active_portfolio_label(is_refined: bool) -> str:
    return "Refined Portfolio" if is_refined else "Base Portfolio"


def _portfolio_summary_from_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not state:
        return None

    opt_res = state.get("optimization_result") or {}
    obj = state.get("objective_key", "maxsharpe")
    if obj not in opt_res:
        return None

    port = opt_res[obj]
    w = pd.Series(port.get("weights", {}), dtype=float)
    w = w[w.abs() > 1e-6]

    if w.empty:
        return None

    w = w.sort_values(ascending=False)
    eff_n = float(1.0 / np.sum(np.square(w.values)))
    max_w = float(w.max())

    sharpe = port.get("sharpe", None)
    sharpe = float(sharpe) if sharpe is not None else None

    return {
        "objective_key": obj,
        "return": float(port.get("return", np.nan)),
        "vol": float(port.get("vol", np.nan)),
        "sharpe": sharpe,
        "active_assets": int(len(w)),
        "max_weight": max_w,
        "effective_n": eff_n,
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x*100:.1f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.2f}"


# ============================================================
# Helper: conflict-safe pain points logic
# ============================================================
PP_TOO_RISKY = "It feels too risky"
PP_TOO_CONSERVATIVE = "It feels too conservative"
PP_TOO_CONCENTRATED = "It’s too concentrated in a few assets"
PP_DISLIKE_ASSETS = "I don’t like some of the assets"
PP_NOT_SURE = "I’m not sure — I just want something safer/smoother"


def _sanitize_pain_points(raw: list[str]) -> list[str]:
    if not raw:
        return []
    s = set(raw)

    if PP_NOT_SURE in s:
        return [PP_NOT_SURE]

    if (PP_TOO_RISKY in s) and (PP_TOO_CONSERVATIVE in s):
        s.remove(PP_TOO_CONSERVATIVE)

    return list(s)


def _infer_targets_from_feedback(
    pain_points: list[str],
    comfort_pref: Optional[str],
    growth_pref: Optional[str],
    concentration_hint: Optional[str],
) -> tuple[str, str, str]:
    goal = "best_tradeoff"
    stability = "balanced"
    concentration = "low"

    if (PP_TOO_RISKY in pain_points) or (PP_NOT_SURE in pain_points):
        goal = "lowest_risk"
        concentration = "low"

        if comfort_pref in ("A smoother ride overall", "Lower ups & downs, even if returns drop"):
            stability = "stable"
        else:
            stability = "balanced"

        if comfort_pref == "Limit big positions":
            concentration = "low"

    if PP_TOO_CONSERVATIVE in pain_points:
        goal = "best_tradeoff"
        if growth_pref == "Higher ups & downs for better returns":
            stability = "swingy"
        else:
            stability = "balanced"
        if growth_pref == "Bigger positions in strong assets":
            concentration = "high"

    if PP_TOO_CONCENTRATED in pain_points:
        concentration = "low"

    return goal, stability, concentration


# ============================================================
# Optional: notes parsing (DEBUG only, NOT required for LLM value)
# ============================================================
def _extract_tickers_from_notes(extra_notes: str, universe: list[str], max_n: int = 10) -> list[str]:
    if not extra_notes:
        return []
    candidates = re.findall(r"\b[A-Z]{1,5}\b", extra_notes.upper())
    if not candidates:
        return []
    universe_set = set(map(str, universe))
    found = []
    for t in candidates:
        if t in universe_set and t not in found:
            found.append(t)
        if len(found) >= max_n:
            break
    return found


def _extra_note_flags(extra_notes: str) -> dict[str, Any]:
    text = (extra_notes or "").lower().strip()
    if not text:
        return {}

    def has_any(words: list[str]) -> bool:
        return any(w in text for w in words)

    flags = {
        "avoid_drawdowns": has_any(["drawdown", "crash", "downside", "big loss", "lose", "loss"]),
        "safer_smoother": has_any(["safer", "smooth", "stable", "less volatile", "low volatility", "sleep at night"]),
        "prefer_diversification": has_any(["diversif", "spread", "less concentrated", "more assets", "many assets"]),
        "avoid_big_positions": has_any(["limit big", "cap", "no more than", "max weight", "too big position"]),
        "still_want_growth": has_any(["still want growth", "long term", "growth", "upside"]),
    }
    return {k: v for k, v in flags.items() if v}


# ---------------- PAGE ----------------
st.set_page_config(page_title="Financial Risk & Portfolio Optimizer", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #050816; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .card {
        background: #0b1020;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #20263a;
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #0b1020;
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid #20263a;
        text-align: left;
    }
    .metric-label { font-size: 0.85rem; color: #8b9ac5; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f7f9ff; }
    .metric-sub { font-size: 0.8rem; color: #9aa6d4; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e6ff;
        margin-bottom: 0.5rem;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #f7f9ff; margin-bottom: 0.2rem; }
    .header-sub { font-size: 0.95rem; color: #9aa6d4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state placeholders
if "base_state" not in st.session_state:
    st.session_state["base_state"] = None
if "refined_state" not in st.session_state:
    st.session_state["refined_state"] = None
if "current_input_df" not in st.session_state:
    st.session_state["current_input_df"] = None
if "pain_points" not in st.session_state:
    st.session_state["pain_points"] = []

# ---------------- LAYOUT ----------------
col_left, col_mid, col_right = st.columns([1.25, 1.05, 1.05])

# ---------------- LEFT: CONTROLS ----------------
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎛 Setup</div>', unsafe_allow_html=True)

    all_tickers = load_available_tickers()

    selected_tickers = st.multiselect(
        "Universe (stocks to include)",
        options=all_tickers,
        default=all_tickers,
    )

    st.markdown(
        '<div class="section-title" style="margin-top:0.8rem;">🧾 Current Portfolio (optional)</div>',
        unsafe_allow_html=True,
    )
    use_current = st.checkbox("I have an existing portfolio (compare vs optimized)", value=False)

    current_weights_dict = None
    current_mode = None

    if use_current:
        current_mode = st.selectbox(
            "How do you want to enter your current portfolio?",
            ["Percent (%)", "Amount (EUR)", "Weight (0-1)"],
            index=0,
        )

    if use_current and selected_tickers:
        if st.session_state["current_input_df"] is None:
            st.session_state["current_input_df"] = (
                pd.DataFrame({"Ticker": selected_tickers, "Value": [0.0] * len(selected_tickers)})
                .set_index("Ticker")
            )
        else:
            st.session_state["current_input_df"] = (
                st.session_state["current_input_df"].reindex(selected_tickers).fillna(0.0)
            )

        if current_mode == "Percent (%)":
            col_label, step, fmt = "Portfolio share (%)", 1.0, "%.2f"
        elif current_mode == "Amount (EUR)":
            col_label, step, fmt = "Invested amount (EUR)", 50.0, "%.2f"
        else:
            col_label, step, fmt = "Weight (0–1)", 0.01, "%.4f"

        edited_df = st.data_editor(
            st.session_state["current_input_df"],
            num_rows="fixed",
            column_config={"Value": st.column_config.NumberColumn(col_label, min_value=0.0, step=step, format=fmt)},
            width="stretch",
        )
        st.session_state["current_input_df"] = edited_df.copy()
        current_weights_dict = _safe_normalize_current_inputs(st.session_state["current_input_df"], current_mode)

    st.markdown(
        '<div class="section-title" style="margin-top:0.8rem;">⚙️ Optimization settings</div>',
        unsafe_allow_html=True,
    )

    rf = st.number_input(
        "Risk-free rate (annual)",
        value=0.02,
        min_value=-0.05,
        max_value=0.20,
        step=0.005,
        format="%.3f",
    )
    w_max = st.slider("Max weight per asset (hard cap)", min_value=0.05, max_value=1.00, value=0.30, step=0.05)

    run_base = st.button("🧱 Run Base Portfolio", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RUN BASE ----------------
if run_base and selected_tickers:
    base_state = run_graph(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        preferences={},
        current_weights=current_weights_dict,
        max_iterations=0,
        clarification_answers=None,
        mode="base",
        use_llm=False,
    )

    st.session_state["base_state"] = base_state
    st.session_state["refined_state"] = None
    st.session_state["pain_points"] = []
    st.rerun()

# ---------------- HEADER ----------------
is_refined_active = st.session_state["refined_state"] is not None
active_label = _active_portfolio_label(is_refined_active)

st.markdown(
    f"""
    <div class="card" style="margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="header-title">📈 Financial Risk & Portfolio Optimizer</div>
          <div class="header-sub">Two-step UX: Run Base → then Refine with user feedback (LangGraph loop)</div>
          <div class="header-sub" style="margin-top:0.35rem;">Currently showing: <b>{active_label}</b></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- ACTIVE STATE ----------------
graph_state = st.session_state["refined_state"] or st.session_state["base_state"]
is_refined_active = st.session_state["refined_state"] is not None
active_label = _active_portfolio_label(is_refined_active)

optimization_result = None
portfolio_weights = None
portfolio_metrics = None
baseline_metrics = None
optimized_metrics = None
current_metrics = None
objective_key = "maxsharpe"

if graph_state is not None:
    optimization_result, objective_key, portfolio_weights, portfolio_metrics = _extract_weights_and_metrics(graph_state)
    baseline_metrics = graph_state.get("baseline_metrics")
    optimized_metrics = graph_state.get("optimized_metrics")
    current_metrics = graph_state.get("current_metrics")

# ---------------- MID: COMPOSITION ----------------
with col_mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">🧱 Portfolio Composition — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("Click **Run Base Portfolio** to generate the first portfolio.")
    else:
        pie_df = portfolio_weights.reset_index()
        pie_df.columns = ["Ticker", "Weight"]
        fig = px.pie(pie_df, names="Ticker", values="Weight", hole=0.6)
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0b1020",
            plot_bgcolor="#0b1020",
            font=dict(color="#E2E6FF"),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT: METRICS ----------------
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📌 Risk & Performance — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_metrics is None:
        st.info("Metrics will appear after base portfolio runs.")
    else:
        opt = portfolio_metrics
        obj_label = "Max Sharpe" if opt["objective_key"] == "maxsharpe" else "Min Variance"
        st.caption(f"Objective: **{obj_label}**")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{float(opt["sharpe"]):.2f}</div>'
                if opt["sharpe"] is not None
                else '<div class="metric-value">–</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="metric-sub">Risk-adjusted return</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Return</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{opt["return"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{opt["vol"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized std dev</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Assets</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{opt["used_assets"]} / {opt["universe_assets"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="metric-sub">Active / Universe</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Refinement UI
# ============================================================
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔁 Refine (after base portfolio)</div>', unsafe_allow_html=True)

if st.session_state["base_state"] is None:
    st.info("Run **Base Portfolio** first. Then you can refine based on what you see.")
else:
    happy_ui = st.radio(
        "Are you happy with this portfolio?",
        ["✅ Yes, this looks good", "❌ No, I’d like to adjust it"],
        index=0,
        horizontal=True,
    )
    is_happy = happy_ui.startswith("✅")

    # ✅ LLM toggle (only relevant if user says "no")
    use_llm_refine = st.checkbox(
        "🤖 Use LLM to interpret my feedback + notes",
        value=True,
        disabled=is_happy,
        help="When enabled, the model can interpret your free-text notes (e.g., 'avoid drawdowns') and may switch objective to Min-Variance if appropriate.",
    )

    # ✅ Optional: debug mode for deterministic note parsing (NOT the main path)
    parse_notes_into_flags = st.checkbox(
        "🧪 Debug: also extract keyword flags from notes",
        value=False,
        disabled=is_happy or (not use_llm_refine),
        help="Optional. Only for debugging; main value is the LLM reading raw notes.",
    )

    if is_happy:
        st.success("Keeping the current portfolio as-is (ACCEPT).")
        st.caption("If you change your mind, select “No” above to adjust it.")
    else:
        st.markdown(
            '<div class="section-title" style="margin-top:0.6rem;">What doesn’t feel right?</div>',
            unsafe_allow_html=True,
        )

        current_pp = list(st.session_state.get("pain_points", []))
        base_options = [
            PP_TOO_RISKY,
            PP_TOO_CONSERVATIVE,
            PP_TOO_CONCENTRATED,
            PP_DISLIKE_ASSETS,
            PP_NOT_SURE,
        ]

        if PP_NOT_SURE in current_pp:
            pain_points = st.multiselect(
                "Select all that apply",
                options=[PP_NOT_SURE],
                default=[PP_NOT_SURE],
                help="If you’re not sure, we’ll aim for a safer/smoother portfolio first.",
            )
            pain_points = [PP_NOT_SURE]
        else:
            options = list(base_options)
            if PP_TOO_RISKY in current_pp:
                options = [o for o in options if o != PP_TOO_CONSERVATIVE]
            if PP_TOO_CONSERVATIVE in current_pp:
                options = [o for o in options if o != PP_TOO_RISKY]

            pain_points = st.multiselect(
                "Select all that apply (optional)",
                options=options,
                default=[p for p in current_pp if p in options],
                help="Optional. If you leave this empty and just write notes, the LLM will still interpret your notes.",
            )

        pain_points = _sanitize_pain_points(pain_points)
        st.session_state["pain_points"] = pain_points

        comfort_pref = None
        if (PP_TOO_RISKY in pain_points) or (PP_NOT_SURE in pain_points):
            comfort_pref = st.selectbox(
                "What would make you more comfortable?",
                [
                    "Lower ups & downs, even if returns drop",
                    "Limit big positions",
                    "A smoother ride overall",
                ],
                index=0,
            )

        growth_pref = None
        if PP_TOO_CONSERVATIVE in pain_points:
            growth_pref = st.selectbox(
                "What are you willing to accept for higher returns?",
                [
                    "Higher ups & downs for better returns",
                    "Bigger positions in strong assets",
                    "I’m investing long-term",
                ],
                index=0,
            )

        concentration_hint = None
        if PP_TOO_CONCENTRATED in pain_points:
            concentration_hint = st.selectbox(
                "How concentrated is acceptable?",
                [
                    "Strict cap per asset",
                    "Balanced",
                    "Some big positions are fine",
                ],
                index=0,
            )

        excluded_assets: list[str] = []
        if PP_DISLIKE_ASSETS in pain_points:
            excluded_assets = st.multiselect(
                "Which assets would you like to exclude?",
                options=selected_tickers,
                default=[],
            )

        extra_notes = st.text_area(
            "Extra notes (optional, free text)",
            placeholder="e.g. I don't want big drawdowns, keep it smoother.",
            height=90,
        ).strip()

        # ✅ We still detect ticker mentions for explicit confirmation (safe)
        notes_tickers = _extract_tickers_from_notes(extra_notes, selected_tickers)

        if notes_tickers and (PP_DISLIKE_ASSETS not in pain_points):
            st.warning(
                f"Your notes mention these tickers: {', '.join(notes_tickers)}. "
                "Do you want to exclude them?"
            )
            confirm_exclude_from_notes = st.checkbox(
                "✅ Exclude tickers mentioned in notes",
                value=False,
                help="This adds them to excluded assets (explicit confirmation).",
            )
            if confirm_exclude_from_notes:
                if PP_DISLIKE_ASSETS not in pain_points:
                    pain_points = list(set(pain_points + [PP_DISLIKE_ASSETS]))
                    st.session_state["pain_points"] = pain_points
                for t in notes_tickers:
                    if t not in excluded_assets:
                        excluded_assets.append(t)

        # ✅ Infer targets automatically (still useful for deterministic base behavior)
        goal, stability, concentration = _infer_targets_from_feedback(
            pain_points=pain_points,
            comfort_pref=comfort_pref,
            growth_pref=growth_pref,
            concentration_hint=concentration_hint,
        )

        # ✅ Optional debug flags (NOT required)
        note_flags = _extra_note_flags(extra_notes) if (use_llm_refine and parse_notes_into_flags) else {}

        if note_flags:
            with st.expander("🧪 Debug: keyword flags from notes"):
                st.json(note_flags)

        max_iterations = st.slider("Max refinement loops", min_value=0, max_value=5, value=2, step=1)
        apply_refine = st.button("✅ Apply Refinements", width="stretch")

        if apply_refine and selected_tickers:
            refined_answers = {
                # deterministic mapping inputs (perception node uses these)
                "goal": goal,
                "stability": stability,
                "concentration": concentration,
                "excluded_assets": excluded_assets,

                # feedback bundle (LLM + rules can use)
                "satisfaction": "no",
                "pain_points": pain_points,
                "comfort_pref": comfort_pref,
                "growth_pref": growth_pref,
                "concentration_hint": concentration_hint,

                # ✅ key: raw notes for the LLM (main value path)
                "extra_notes": extra_notes,
                # optional debug structure (can be empty)
                "extra_note_flags": note_flags,
                "notes_tickers": notes_tickers,
            }

            refined_state = run_graph(
                selected_tickers=selected_tickers,
                rf=float(rf),
                w_max=float(w_max),
                preferences={},
                current_weights=current_weights_dict,
                max_iterations=int(max_iterations),
                clarification_answers=refined_answers,
                mode="refine",
                use_llm=bool(use_llm_refine),
            )
            st.session_state["refined_state"] = refined_state
            st.success("Refinement applied. Scroll up to see updated portfolio & charts.")
            st.rerun()

        if st.session_state.get("refined_state") is not None:
            rs = st.session_state["refined_state"]
            with st.expander("🔧 Refinement summary (what changed?)"):
                st.write(f"Iterations used: **{rs.get('iteration', 0)}** / {rs.get('max_iterations', 0)}")

                obj_key = rs.get("objective_key")
                obj_label = "Max Sharpe" if obj_key == "maxsharpe" else "Min Variance"
                st.write(f"Objective (final): **{obj_label}** (`{obj_key}`)")

                try:
                    st.write(f"w_max (final): **{float(rs.get('w_max', np.nan)):.2f}**")
                except Exception:
                    st.write("w_max: **–**")

                st.write("Applied actions:")
                st.json(rs.get("changes_applied", []))

                if rs.get("changes_rejected"):
                    st.write("Rejected actions:")
                    st.json(rs.get("changes_rejected", []))

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Evaluation: Base vs Refine ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Evaluation — Base vs Refine</div>', unsafe_allow_html=True)

base_sum = _portfolio_summary_from_state(st.session_state.get("base_state"))
ref_sum = _portfolio_summary_from_state(st.session_state.get("refined_state"))

if base_sum is None:
    st.info("Run **Base Portfolio** first to enable evaluation.")
else:
    col1, col2, col3 = st.columns([1.0, 1.0, 1.0])

    with col1:
        st.markdown("**Base**")
        st.write(f"- Objective: `{base_sum['objective_key']}`")
        st.write(f"- Return: {_fmt_pct(base_sum['return'])}")
        st.write(f"- Vol: {_fmt_pct(base_sum['vol'])}")
        st.write(f"- Sharpe: {_fmt_num(base_sum['sharpe'])}")
        st.write(f"- Active assets: {base_sum['active_assets']}")
        st.write(f"- Max weight: {_fmt_pct(base_sum['max_weight'])}")
        st.write(f"- Effective N: {base_sum['effective_n']:.1f}")

    with col2:
        st.markdown("**Refined**")
        if ref_sum is None:
            st.write("Not computed yet.")
        else:
            st.write(f"- Objective: `{ref_sum['objective_key']}`")
            st.write(f"- Return: {_fmt_pct(ref_sum['return'])}")
            st.write(f"- Vol: {_fmt_pct(ref_sum['vol'])}")
            st.write(f"- Sharpe: {_fmt_num(ref_sum['sharpe'])}")
            st.write(f"- Active assets: {ref_sum['active_assets']}")
            st.write(f"- Max weight: {_fmt_pct(ref_sum['max_weight'])}")
            st.write(f"- Effective N: {ref_sum['effective_n']:.1f}")

    with col3:
        st.markdown("**Delta (Refined − Base)**")
        if ref_sum is None:
            st.write("Run refinement to see deltas.")
        else:
            d_ret = ref_sum["return"] - base_sum["return"]
            d_vol = ref_sum["vol"] - base_sum["vol"]
            d_sh = (
                (ref_sum["sharpe"] - base_sum["sharpe"])
                if (ref_sum["sharpe"] is not None and base_sum["sharpe"] is not None)
                else None
            )
            d_eff = ref_sum["effective_n"] - base_sum["effective_n"]
            d_mx = ref_sum["max_weight"] - base_sum["max_weight"]
            d_act = ref_sum["active_assets"] - base_sum["active_assets"]

            st.write(f"- Δ Return: {_fmt_pct(d_ret)}")
            st.write(f"- Δ Vol: {_fmt_pct(d_vol)}")
            st.write(f"- Δ Sharpe: {_fmt_num(d_sh)}")
            st.write(f"- Δ Max weight: {_fmt_pct(d_mx)}")
            st.write(f"- Δ Effective N: {d_eff:+.1f}")
            st.write(f"- Δ Active assets: {d_act:+d}")

with st.expander("📦 Export run logs (JSON)"):
    if st.session_state.get("base_state") is not None:
        st.download_button(
            "Download BASE state (JSON)",
            data=json.dumps(st.session_state["base_state"], indent=2, default=str),
            file_name="base_state.json",
            mime="application/json",
        )
    if st.session_state.get("refined_state") is not None:
        st.download_button(
            "Download REFINED state (JSON)",
            data=json.dumps(st.session_state["refined_state"], indent=2, default=str),
            file_name="refined_state.json",
            mime="application/json",
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Efficient Frontier ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-title">📊 Expected Return vs Risk (Efficient Frontier) — {active_label}</div>',
    unsafe_allow_html=True,
)

if (optimization_result is None) or (not optimization_result.get("frontier")) or (portfolio_metrics is None):
    st.info("Run base/refine to visualize the efficient frontier.")
else:
    frontier_df = pd.DataFrame(optimization_result["frontier"])
    y_col = (
        "realized_return"
        if "realized_return" in frontier_df.columns
        else ("return" if "return" in frontier_df.columns else frontier_df.columns[-1])
    )
    fig_frontier = px.line(frontier_df, x="vol", y=y_col, markers=True)

    port = optimization_result[portfolio_metrics["objective_key"]]
    port_y = float(port.get("return", np.nan))

    fig_frontier.add_trace(
        go.Scatter(
            x=[float(port.get("vol", np.nan))],
            y=[port_y],
            mode="markers+text",
            name=active_label,
            text=[active_label],
            textposition="top left",
            marker=dict(size=10),
        )
    )

    if current_metrics is not None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[float(current_metrics["vol"])],
                y=[float(current_metrics["return"])],
                mode="markers+text",
                name="Current",
                text=["Current"],
                textposition="bottom right",
                marker=dict(size=10),
            )
        )
    elif baseline_metrics is not None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[float(baseline_metrics["vol"])],
                y=[float(baseline_metrics["return"])],
                mode="markers+text",
                name="Baseline (Equal Weight)",
                text=["Baseline (Equal Weight)"],
                textposition="bottom right",
                marker=dict(size=10),
            )
        )

    fig_frontier.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Risk (Volatility, σ)",
        yaxis_title="Expected Return (µ)",
    )
    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_frontier, width="stretch")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Risk Contribution by Asset ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">📊 Risk Contribution by Asset — {active_label}</div>', unsafe_allow_html=True)

if optimized_metrics is None or not optimized_metrics:
    st.info("Run base/refine to see risk contributions.")
else:
    tickers_rc = optimized_metrics["tickers"]
    active_rc = np.array(optimized_metrics["rc_pct"], dtype=float)

    main_col = "Active"
    df_rc = pd.DataFrame({"Ticker": tickers_rc, main_col: active_rc})

    compare_label = None
    if current_metrics is not None:
        df_rc["Current"] = np.array(current_metrics["rc_pct"], dtype=float)
        compare_label = "Current"
    elif baseline_metrics is not None:
        df_rc["Baseline (Equal Weight)"] = np.array(baseline_metrics["rc_pct"], dtype=float)
        compare_label = "Baseline (Equal Weight)"

    df_long = df_rc.melt(id_vars="Ticker", var_name="Portfolio", value_name="Risk Contribution")
    fig_rc = px.bar(df_long, x="Ticker", y="Risk Contribution", color="Portfolio", barmode="group")
    fig_rc.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Risk contribution (share of total σ)",
        xaxis_title="",
        legend_title="",
    )
    fig_rc.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rc, width="stretch")

    if compare_label:
        st.caption(
            f"Bars show each asset’s share of total portfolio risk (σ). Comparison: **{compare_label}** vs **{active_label}**."
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Bottom: Weights + Explanation ----------------
st.markdown("")
bottom_left, bottom_right = st.columns([1.3, 1.0])

with bottom_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📉 Weights — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("No portfolio yet.")
    else:
        df_weights = portfolio_weights.to_frame("Weight")
        st.dataframe(df_weights.style.format("{:.3f}"), width="stretch", height=360)

    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">💬 Explanation — {active_label}</div>', unsafe_allow_html=True)

    if graph_state is None:
        st.info("Explanation will appear after base/refine.")
    else:
        st.write(graph_state.get("explanation", "No explanation generated."))

        with st.expander("🧠 LLM decision (accept vs refine)"):
            st.json(graph_state.get("llm_decision", {}))

        with st.expander("📰 News signals (placeholder)"):
            st.json(graph_state.get("news_signals", {}))

        with st.expander("🔍 Debug notes (graph trace)"):
            notes = graph_state.get("debug_notes", [])
            if not notes:
                st.write("No debug notes.")
            else:
                for n in notes:
                    st.write(f"- {n}")

    st.markdown("</div>", unsafe_allow_html=True) 