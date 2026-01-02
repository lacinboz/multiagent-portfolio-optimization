# dashboard_langgraph_app.py
# ✅ Updated for A/B Candidate Selection + LLM-in-the-loop (Option A compatible)
# - Base run: deterministic single portfolio (maxsharpe) for baseline comparison.
# - Refine run: deterministic candidates (maxsharpe + minvar) + LLM selects the best candidate.
# - No refine-actions / parameter updates. Selection only.
#
# FIXES INCLUDED:
# 1) ✅ Import run_graph from `portfolio_langgraph_withllm` (your current backend file)
#    - If you renamed it to portfolio_langgraph.py, change the import accordingly.
# 2) ✅ Efficient Frontier marker uses FINAL metrics consistently (x_vol and y_ret both from optimized_metrics when present).
# 3) ✅ Risk Contribution chart safely falls back: if optimized_metrics missing, show a helpful message (no crash).
# 4) ✅ Minor safety: frontier marker won’t plot if either x or y missing.

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

# ✅ IMPORTANT: your backend runner (you said your portfolio code is in portfolio_langgraph_withllm)
from portfolio_langgraph_withllm import run_graph
# If you later move it to portfolio_langgraph.py, swap to:
# from portfolio_langgraph import run_graph

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


def _get_chosen_candidate(state: Dict[str, Any]) -> str:
    chosen = state.get("chosen_candidate") or state.get("objective_key") or "maxsharpe"
    chosen = str(chosen).lower().strip()
    return chosen or "maxsharpe"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _extract_weights_and_metrics(state: Dict[str, Any]):
    """
    ✅ Option A compatible extraction.
    Prefer finalized results:
      - optimized_weights (final selection)
      - optimized_metrics (final selection risk metrics)
    Fallback:
      - optimization_result[chosen] if optimized_* absent.
    """
    optimization_result = state.get("optimization_result") or {}
    chosen = _get_chosen_candidate(state)

    weights_series = None
    portfolio_metrics = None

    # 1) Prefer finalized weights/metrics
    opt_w = state.get("optimized_weights") or {}
    opt_m = state.get("optimized_metrics") or {}

    if opt_w:
        w = pd.Series(opt_w, dtype=float)
        w = w[w.abs() > 1e-6].sort_values(ascending=False)
        weights_series = w

        sharpe = _safe_float(opt_m.get("sharpe", None))
        ret = _safe_float(opt_m.get("return", None))
        vol = _safe_float(opt_m.get("vol", None))

        # active_assets might exist; otherwise derive from weights
        active_assets = opt_m.get("active_assets", None)
        try:
            active_assets = int(active_assets) if active_assets is not None else int(len(w))
        except Exception:
            active_assets = int(len(w))

        portfolio_metrics = {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "sharpe": sharpe,
            "used_assets": int(len(w)),
            "universe_assets": int(len(state.get("selected_tickers", []))),
            "active_assets": active_assets,
        }
        return optimization_result, chosen, weights_series, portfolio_metrics

    # 2) Fallback: read directly from optimization_result[chosen]
    if optimization_result and chosen in optimization_result:
        port = optimization_result[chosen]
        w = pd.Series(port.get("weights", {}), dtype=float)
        w = w[w.abs() > 1e-6].sort_values(ascending=False)
        weights_series = w

        sharpe = _safe_float(port.get("sharpe", None))
        ret = _safe_float(port.get("return", None))
        vol = _safe_float(port.get("vol", None))

        portfolio_metrics = {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "sharpe": sharpe,
            "used_assets": int(len(w)),
            "universe_assets": int(len(state.get("selected_tickers", []))),
            "active_assets": int(len(w)),
        }

    return optimization_result, chosen, weights_series, portfolio_metrics


def _active_portfolio_label(is_refined: bool) -> str:
    return "Refined Portfolio" if is_refined else "Base Portfolio"


def _portfolio_summary_from_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    ✅ For Evaluation section.
    Prefer optimized_* (final) to avoid mismatch.
    """
    if not state:
        return None

    chosen = _get_chosen_candidate(state)
    opt_m = state.get("optimized_metrics") or {}
    opt_w = state.get("optimized_weights") or {}

    if opt_w:
        w = pd.Series(opt_w, dtype=float)
        w = w[w.abs() > 1e-6]
        if w.empty:
            return None

        w = w.sort_values(ascending=False)
        eff_n = float(1.0 / np.sum(np.square(w.values)))
        max_w = float(w.max())

        sharpe = _safe_float(opt_m.get("sharpe", None))
        ret = _safe_float(opt_m.get("return", None))
        vol = _safe_float(opt_m.get("vol", None))

        active_assets = opt_m.get("active_assets", None)
        try:
            active_assets = int(active_assets) if active_assets is not None else int(len(w))
        except Exception:
            active_assets = int(len(w))

        return {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "sharpe": sharpe,
            "active_assets": active_assets,
            "max_weight": max_w,
            "effective_n": eff_n,
        }

    # fallback older-style
    opt_res = state.get("optimization_result") or {}
    if chosen not in opt_res:
        return None

    port = opt_res[chosen]
    w = pd.Series(port.get("weights", {}), dtype=float)
    w = w[w.abs() > 1e-6]
    if w.empty:
        return None

    w = w.sort_values(ascending=False)
    eff_n = float(1.0 / np.sum(np.square(w.values)))
    max_w = float(w.max())

    sharpe = _safe_float(port.get("sharpe", None))
    ret = _safe_float(port.get("return", None))
    vol = _safe_float(port.get("vol", None))

    return {
        "candidate": chosen,
        "return": ret if ret is not None else float(np.nan),
        "vol": vol if vol is not None else float(np.nan),
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
# Pain point labels (must match backend constants)
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

def _rc_series_aligned_to(tickers_target: list[str], metrics: dict):
    """
    metrics: {'tickers': [...], 'rc_pct': [...]}
    returns: rc_pct aligned to tickers_target order (np.ndarray)
    """
    if not metrics:
        return None

    src_t = metrics.get("tickers")
    src_rc = metrics.get("rc_pct")
    if src_t is None or src_rc is None:
        return None

    src_t = list(map(str, src_t))
    src_rc = np.array(src_rc, dtype=float)

    if len(src_t) != len(src_rc):
        return None

    m = {t: float(v) for t, v in zip(src_t, src_rc)}
    return np.array([m.get(str(t), np.nan) for t in tickers_target], dtype=float)

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
          <div class="header-sub">Two-step UX: Run Base → then Refine with A/B candidate selection (LLM-in-the-loop)</div>
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
chosen_candidate = "maxsharpe"

if graph_state is not None:
    optimization_result, chosen_candidate, portfolio_weights, portfolio_metrics = _extract_weights_and_metrics(graph_state)
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
        cand = opt["candidate"]
        obj_label = "Max Sharpe" if cand == "maxsharpe" else "Min Variance"
        st.caption(f"Selected candidate: **{obj_label}** (`{cand}`)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{float(opt["sharpe"]):.2f}</div>'
                if opt.get("sharpe") is not None
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
# Refinement UI (Candidate selection)
# ============================================================
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔁 Refine (after base portfolio)</div>', unsafe_allow_html=True)

if st.session_state["base_state"] is None:
    st.info("Run **Base Portfolio** first. Then you can refine using A/B candidate selection.")
else:
    happy_ui = st.radio(
        "Are you happy with this portfolio?",
        ["✅ Yes, this looks good", "❌ No, I’d like to adjust it"],
        index=0,
        horizontal=True,
    )
    is_happy = happy_ui.startswith("✅")

    use_llm_refine = st.checkbox(
        "🤖 Use LLM to choose among candidate portfolios",
        value=True,
        disabled=is_happy,
        help="When enabled, the model compares candidates (Max-Sharpe vs Min-Variance) and selects the one most aligned with your feedback/notes.",
    )

    if is_happy:
        st.success("Keeping the current portfolio as-is (ACCEPT).")
        st.caption("If you change your mind, select “No” above to compare alternatives.")
    else:
        st.markdown(
            '<div class="section-title" style="margin-top:0.6rem;">What doesn’t feel right?</div>',
            unsafe_allow_html=True,
        )

        current_pp = list(st.session_state.get("pain_points", []))
        base_options = [PP_TOO_RISKY, PP_TOO_CONSERVATIVE, PP_TOO_CONCENTRATED, PP_DISLIKE_ASSETS, PP_NOT_SURE]

        if PP_NOT_SURE in current_pp:
            pain_points = st.multiselect(
                "Select all that apply",
                options=[PP_NOT_SURE],
                default=[PP_NOT_SURE],
                help="If you’re not sure, we’ll prioritize safer / smoother candidates.",
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
                help="Optional. If empty, your notes still help the LLM choose a candidate.",
            )

        pain_points = _sanitize_pain_points(pain_points)
        st.session_state["pain_points"] = pain_points

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

        notes_tickers = _extract_tickers_from_notes(extra_notes, selected_tickers)

        if notes_tickers and (PP_DISLIKE_ASSETS not in pain_points):
            st.warning(
                f"Your notes mention these tickers: {', '.join(notes_tickers)}. "
                "Do you want to exclude them?"
            )
            confirm_exclude_from_notes = st.checkbox(
                "✅ Exclude tickers mentioned in notes",
                value=False,
                help="This adds them to excluded assets only with explicit confirmation.",
            )
            if confirm_exclude_from_notes:
                if PP_DISLIKE_ASSETS not in pain_points:
                    pain_points = list(set(pain_points + [PP_DISLIKE_ASSETS]))
                    st.session_state["pain_points"] = pain_points
                for t in notes_tickers:
                    if t not in excluded_assets:
                        excluded_assets.append(t)

        apply_refine = st.button("✅ Run Candidate Selection (Refine)", width="stretch")

        if apply_refine and selected_tickers:
            refined_answers = {
                "satisfaction": "no",
                "pain_points": pain_points,
                "excluded_assets": excluded_assets,
                "extra_notes": extra_notes,
                "notes_tickers": notes_tickers,
            }

            refined_state = run_graph(
                selected_tickers=selected_tickers,
                rf=float(rf),
                w_max=float(w_max),
                preferences={},
                current_weights=current_weights_dict,
                clarification_answers=refined_answers,
                mode="refine",
                use_llm=bool(use_llm_refine),
            )
            st.session_state["refined_state"] = refined_state
            st.success("Refinement applied. Scroll up to see the selected candidate portfolio.")
            st.rerun()

        if st.session_state.get("refined_state") is not None:
            rs = st.session_state["refined_state"]
            with st.expander("🔧 Selection summary (what was chosen?)"):
                chosen = _get_chosen_candidate(rs)
                st.write(f"Chosen candidate: **`{chosen}`**")

                llm_decision = rs.get("llm_decision") or {}
                if llm_decision:
                    st.write("LLM decision payload:")
                    st.json(llm_decision)

                cand_keys = list((rs.get("optimization_result") or {}).keys())
                st.write(f"Available candidates: {cand_keys}")

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
        st.write(f"- Candidate: `{base_sum['candidate']}`")
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
            st.write(f"- Candidate: `{ref_sum['candidate']}`")
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

frontier = None
if isinstance(optimization_result, dict):
    frontier = optimization_result.get("frontier")

if (frontier is None) or (portfolio_metrics is None):
    st.info("Run base/refine to visualize the efficient frontier.")
else:
    frontier_df = pd.DataFrame(frontier)
    y_col = (
        "realized_return"
        if "realized_return" in frontier_df.columns
        else ("return" if "return" in frontier_df.columns else frontier_df.columns[-1])
    )
    fig_frontier = px.line(frontier_df, x="vol", y=y_col, markers=True)

    # Mark chosen portfolio point: ✅ use FINAL metrics consistently when present
    chosen = portfolio_metrics["candidate"]
    port = optimization_result.get(chosen, {}) if isinstance(optimization_result, dict) else {}

    # Defaults from optimizer
    x_vol = _safe_float(port.get("vol", None))
    y_ret = _safe_float(port.get("return", None))

    # ✅ Override BOTH X and Y from optimized_metrics if available
    if optimized_metrics is not None and optimized_metrics:
        x_final = _safe_float(optimized_metrics.get("vol", None))
        y_final = _safe_float(optimized_metrics.get("return", None))
        if x_final is not None:
            x_vol = x_final
        if y_final is not None:
            y_ret = y_final

    if x_vol is not None and y_ret is not None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[x_vol],
                y=[y_ret],
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
st.markdown(
    f'<div class="section-title">📊 Risk Contribution by Asset — {active_label}</div>',
    unsafe_allow_html=True,
)

if optimized_metrics is None or not optimized_metrics:
    st.info("Risk contributions are available after the graph computes `optimized_metrics` (risk_agent output).")
else:
    # --- Active (final portfolio) ---
    tickers_rc = list(map(str, optimized_metrics.get("tickers", [])))
    active_rc = np.array(optimized_metrics.get("rc_pct", []), dtype=float)

    if len(tickers_rc) == 0 or len(active_rc) == 0 or len(tickers_rc) != len(active_rc):
        st.info("Risk contribution data is missing or malformed (tickers/rc_pct mismatch).")
    else:
        df_rc = pd.DataFrame({"Ticker": tickers_rc, "Active": active_rc})

        compare_label = None

        # --- Align current_metrics to active tickers (avoid length mismatch) ---
        if current_metrics is not None:
            aligned = _rc_series_aligned_to(tickers_rc, current_metrics)
            if aligned is not None:
                df_rc["Current"] = aligned
                compare_label = "Current"
            else:
                st.info("Current RC cannot be aligned (current_metrics should include 'tickers' + 'rc_pct').")

        # --- Align baseline_metrics to active tickers (avoid length mismatch) ---
        elif baseline_metrics is not None:
            aligned = _rc_series_aligned_to(tickers_rc, baseline_metrics)
            if aligned is not None:
                df_rc["Baseline (Equal Weight)"] = aligned
                compare_label = "Baseline (Equal Weight)"
            else:
                st.info("Baseline RC cannot be aligned (baseline_metrics should include 'tickers' + 'rc_pct').")

        # Plot
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
                f"Bars show each asset’s share of total portfolio risk (σ). "
                f"Comparison: **{compare_label}** vs **{active_label}**."
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

        with st.expander("🧠 LLM decision (candidate selection)"):
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
