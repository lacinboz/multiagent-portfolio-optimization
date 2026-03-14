# dashboard_langgraph_app.py
# ✅ Updated for:
# - Option A candidate selection (LLM chooses maxsharpe/minvar)
# - SAFE metric rendering (no crashes when sharpe/metrics missing)
# - Insight panel supports BOTH:
#     - structured JSON insight in state["insight"]
#     - narrative insight in state["insight_raw_text"]
# - Streamlit-safe container widths (use_container_width=True)
# - Safe delta computations (no None - None crashes)
# - ✅ NEW: News Snapshot / News Risk Check UI rendering
#     - supports BOTH placeholder news_signals and your new llm_client.generate_news_snapshot() output
#     - shows a short snapshot + per-ticker risk flags if present
# - ✅ NEW: News is OPTIONAL in refine (gated by UI checkbox)
#     - sends refined_answers["use_news"] = "yes"/"no"
#     - passes use_news=... into run_graph
#
# Notes:
# - Base run: mode="base" use_llm=False use_news=False
# - Refine run: mode="refine" use_llm=True/False use_news=True/False
# - Prefer finalized outputs:
#     state["optimized_weights"], state["optimized_metrics"], state["insight"/"insight_raw_text"]
#   fallback to optimization_result[chosen] only if optimized_* absent.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ✅ IMPORTANT: backend runner
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


def _get_chosen_candidate(state: Dict[str, Any]) -> str:
    chosen = state.get("chosen_candidate") or state.get("objective_key") or "maxsharpe"
    chosen = str(chosen).lower().strip()
    return chosen or "maxsharpe"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    aa = _safe_float(a)
    bb = _safe_float(b)
    if aa is None or bb is None:
        return None
    return float(aa - bb)


def _extract_weights_and_metrics(state: Dict[str, Any]):
    """
    ✅ Option A compatible extraction.
    Prefer FINAL outputs:
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

        sharpe = _safe_float(opt_m.get("sharpe"))
        ret = _safe_float(opt_m.get("return"))
        vol = _safe_float(opt_m.get("vol"))

        ret_pct = _safe_float(opt_m.get("return_pct"))
        vol_pct = _safe_float(opt_m.get("vol_pct"))

        active_assets = opt_m.get("active_assets", None)
        try:
            active_assets = int(active_assets) if active_assets is not None else int(len(w))
        except Exception:
            active_assets = int(len(w))

        portfolio_metrics = {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "return_pct": ret_pct,
            "vol_pct": vol_pct,
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

        sharpe = _safe_float(port.get("sharpe"))
        ret = _safe_float(port.get("return"))
        vol = _safe_float(port.get("vol"))

        portfolio_metrics = {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "return_pct": None,
            "vol_pct": None,
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

        sharpe = _safe_float(opt_m.get("sharpe"))
        ret = _safe_float(opt_m.get("return"))
        vol = _safe_float(opt_m.get("vol"))

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

    sharpe = _safe_float(port.get("sharpe"))
    ret = _safe_float(port.get("return"))
    vol = _safe_float(port.get("vol"))

    return {
        "candidate": chosen,
        "return": ret if ret is not None else float(np.nan),
        "vol": vol if vol is not None else float(np.nan),
        "sharpe": sharpe,
        "active_assets": int(len(w)),
        "max_weight": max_w,
        "effective_n": eff_n,
    }


def _fmt_pct_from_decimal(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x*100:.1f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.2f}"


def _fmt_pct_from_pct_field(x_pct: Optional[float]) -> str:
    # for fields already in percent units (e.g., 17.7)
    if x_pct is None or not np.isfinite(x_pct):
        return "–"
    return f"{float(x_pct):.1f}%"


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

def _detect_mentioned_tickers(text: str, universe: list[str], max_n: int = 6) -> list[str]:
    if not text:
        return []
    text_u = text.upper()

    # en basit ve güvenli yöntem: universe içindeki ticker'ları text içinde ara
    found = []
    for t in universe:
        tt = str(t).upper().strip()
        if not tt:
            continue
        if tt in text_u and tt not in found:
            found.append(tt)
        if len(found) >= max_n:
            break
    return found

# ✅ NEW: News rendering helpers (supports placeholder + real LLM news snapshot output)
def _extract_news_snapshot_and_risk(state: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not state:
        return None, None

    # ✅ NEW: prefer explicit fields from backend
    snapshot_text = None
    # ✅ 0) UI-friendly raw snapshot (if backend provides it)
    if isinstance(state.get("news_snapshot_text_raw"), str) and state["news_snapshot_text_raw"].strip():
        snapshot_text = state["news_snapshot_text_raw"].strip()
    if isinstance(state.get("news_snapshot_text"), str) and state["news_snapshot_text"].strip():
        snapshot_text = state["news_snapshot_text"].strip()
    elif isinstance(state.get("news_snapshot"), str) and state["news_snapshot"].strip():  # future-proof
        snapshot_text = state["news_snapshot"].strip()

    # ✅ NEW: prefer explicit risk json
    risk_json = state.get("news_risk_json") if isinstance(state.get("news_risk_json"), dict) else None
    if risk_json is None:
        # fallback legacy
        risk_json = state.get("news_signals") if isinstance(state.get("news_signals"), dict) else None

    # If risk_json has summary, use it as snapshot if snapshot_text missing
    if (not snapshot_text) and isinstance(risk_json, dict):
        if isinstance(risk_json.get("summary"), str) and risk_json["summary"].strip():
            snapshot_text = risk_json["summary"].strip()

    # Placeholder schema fallback: global risk_flags/vol_regime -> build small text if still none
    if (not snapshot_text) and isinstance(risk_json, dict):
        glob = risk_json.get("global")
        if isinstance(glob, dict):
            flags = glob.get("risk_flags")
            vr = glob.get("vol_regime")
            if isinstance(flags, list) and flags:
                snapshot_text = f"Detected {len(flags)} potential event-risk flag(s). Vol regime: {vr or 'normal'}."

    return snapshot_text, risk_json



def _news_section(state: Dict[str, Any]):
    st.markdown('<div class="section-title">📰 News Snapshot & Risk Check</div>', unsafe_allow_html=True)

    snapshot_text, risk_json = _extract_news_snapshot_and_risk(state)

    if (snapshot_text is None) and (not isinstance(risk_json, dict) or not risk_json):
        st.caption("No news snapshot available (yet).")
        return

    if snapshot_text:
        st.write(snapshot_text)

    if isinstance(risk_json, dict) and risk_json:
        glob = risk_json.get("global") if isinstance(risk_json.get("global"), dict) else {}
        by_ticker = risk_json.get("by_ticker") if isinstance(risk_json.get("by_ticker"), dict) else {}

        # global
        if isinstance(glob, dict) and glob:
            vol_regime = str(glob.get("vol_regime") or "normal")
            st.caption(f"Global regime: **{vol_regime}**")

            flags = glob.get("risk_flags")
            if isinstance(flags, list) and flags:
                with st.expander("Global risk flags"):
                    st.json(flags)

        # per ticker
        if isinstance(by_ticker, dict) and by_ticker:
            rows = []
            for t, v in by_ticker.items():
                if not isinstance(v, dict):
                    continue
                rows.append(
                    {
                        "Ticker": str(t),
                        "Risk flag": str(v.get("risk_flag") or "none"),
                        "Confidence": v.get("confidence"),
                    }
                )
            if rows:
                df = pd.DataFrame(rows)
                # sort: highest confidence first
                if "Confidence" in df.columns:
                    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
                    df = df.sort_values(by="Confidence", ascending=False, na_position="last")
                st.dataframe(df, use_container_width=True, height=220)

def _get_news_items_by_id(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build lookup map for BOTH:
      - evidence_id (preferred; e.g. APP_9d1b573b)
      - raw id (fallback; e.g. cea6dfaf1ff9)

    This fixes: "(not found in news_items_by_id)" when actions reference evidence_ids.
    """
    m = state.get("news_items_by_id") or state.get("items_by_id")
    if isinstance(m, dict) and m:
        out = {}
        for k, v in m.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out

    raw = (
        state.get("news_raw_items")
        or state.get("news_snapshot_raw_items")
        or state.get("news_items")
        or state.get("news_raw")
        or state.get("news")
        or []
    )

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue

            eid = it.get("evidence_id")
            rid = it.get("id")

            # ✅ Prefer evidence_id key
            if eid is not None and str(eid).strip():
                out[str(eid)] = it


    return out



def _render_action_evidence(
    *,
    action: Dict[str, Any],
    news_items_by_id: Dict[str, Dict[str, Any]],
    universe_tickers: List[str],
):
    # ✅ 1) Prefer structured evidence list (your backend output)
    ev_list = action.get("evidence")
    eids: List[str] = []

    if isinstance(ev_list, list) and ev_list:
        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("evidence_id") or ev.get("id") or "").strip()
            if eid:
                eids.append(eid)

    # ✅ 2) Fallback legacy field if present
    if not eids:
        raw_eids = action.get("evidence_ids") or []
        if isinstance(raw_eids, list):
            eids = [str(x).strip() for x in raw_eids if str(x).strip()]

    if not eids:
        st.caption("No evidence attached.")
        return

    lines: List[str] = []
    for eid in eids[:6]:  # show a few
        item = news_items_by_id.get(eid)

        # If we can't find the full item, still show the id (debug)
        if not isinstance(item, dict):
            lines.append(f"- `{eid}` (not found in news_items_by_id)")
            continue

        ticker = str(item.get("ticker") or "UNK").strip()
        date = str(item.get("date") or "unknown").strip()
        source = str(item.get("source") or item.get("provider") or "unknown").strip()
        headline = str(item.get("headline") or "headline").strip()
        summary = str(item.get("summary") or "").strip()
        url = str(item.get("url") or "").strip()

        # ✅ Cross-mention detection (headline + summary)
        mentioned = _detect_mentioned_tickers(f"{headline} {summary}", universe_tickers)
        # item’in kendi ticker’ını “mentions”tan çıkar
        also_mentions = [t for t in mentioned if t != ticker.upper()]

        if url:
            lines.append(f"- {ticker} ({date} | {source}) [{headline}]({url})")
        else:
            lines.append(f"- {ticker} ({date} | {source}) {headline} (no url)")

        if also_mentions:
            lines.append(f"  - _Mentions:_ {', '.join(also_mentions)}")
        else:
            lines.append("  - _Mentions: 0")




    st.markdown("**Evidence**")
    st.markdown("\n".join(lines))


# ✅ NEW: Insight rendering helpers (supports narrative raw_text)
def _insight_section(state: Dict[str, Any]):
    insight = state.get("insight")
    ok = state.get("insight_ok")
    issues = state.get("insight_issues") or []
    parse_mode = state.get("insight_parse_mode")
    raw_text = state.get("insight_raw_text")

    st.markdown('<div class="section-title">✨ Insights (LLM)</div>', unsafe_allow_html=True)

    has_any = (insight is not None) or (isinstance(raw_text, str) and raw_text.strip())
    if not has_any:
        st.info("No insights generated yet. Run **Refine** with LLM enabled to produce insights.")
        return

    if ok is True:
        st.success(f"Insight generated ({parse_mode or 'unknown parse'}).")
    elif ok is False:
        st.warning("Insight generation had issues (showing best-effort output).")
    else:
        st.caption("Insight status unknown.")

    if issues:
        with st.expander("⚠️ Insight issues"):
            for it in issues:
                st.write(f"- {it}")

    # ✅ Prefer narrative text if present
    if isinstance(raw_text, str) and raw_text.strip():
        st.markdown(raw_text)
        return

    # Otherwise render structured JSON insight
    headline = (insight or {}).get("headline")
    if isinstance(headline, str) and headline.strip():
        st.markdown(f"**{headline.strip()}**")
    else:
        st.markdown("**Portfolio insights**")

    story = (insight or {}).get("portfolio_story") or []
    if isinstance(story, list) and story:
        st.markdown("**What changed / what it means**")
        for s in story[:8]:
            if isinstance(s, str) and s.strip():
                st.write(f"- {s.strip()}")

    drivers = (insight or {}).get("risk_drivers") or []
    if isinstance(drivers, list) and drivers:
        st.markdown("**Main risk drivers**")
        for d in drivers[:8]:
            if isinstance(d, str) and d.strip():
                st.write(f"- {d.strip()}")

    bvr = (insight or {}).get("base_vs_refine") or {}
    metric_deltas = (bvr.get("metric_deltas") or {}) if isinstance(bvr, dict) else {}
    key_changes = (bvr.get("key_changes") or []) if isinstance(bvr, dict) else []

    if key_changes:
        st.markdown("**Key changes**")
        for k in key_changes[:8]:
            if isinstance(k, str) and k.strip():
                st.write(f"- {k.strip()}")

    if metric_deltas:
        st.markdown("**Metric deltas (Base → Refine)**")
        try:
            st.json(metric_deltas)
        except Exception:
            st.write(metric_deltas)

    news_overlay = (insight or {}).get("news_overlay") or []
    if isinstance(news_overlay, list) and news_overlay:
        st.markdown("**News overlay**")
        for n in news_overlay[:8]:
            if isinstance(n, str) and n.strip():
                st.write(f"- {n.strip()}")

    actions = (insight or {}).get("action_suggestions_optional") or []
    if isinstance(actions, list) and actions:
        st.markdown("**Optional actions**")
        for a in actions[:8]:
            if isinstance(a, str) and a.strip():
                st.write(f"- {a.strip()}")


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
if "news_actions_state" not in st.session_state:
    st.session_state["news_actions_state"] = None
if "selected_news_actions" not in st.session_state:
    st.session_state["selected_news_actions"] = []
if "news_overview_state" not in st.session_state:
    st.session_state["news_overview_state"] = None



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
            use_container_width=True,
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

    run_base = st.button("🧱 Run Base Portfolio", use_container_width=True)
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
        use_llm=True,
        use_news=False,  # ✅ explicit
    )

    st.session_state["base_state"] = base_state
    st.session_state["refined_state"] = None
    st.session_state["pain_points"] = []
    st.rerun()

# ---------------- HEADER ----------------
is_refined_active = st.session_state["refined_state"] is not None
active_label = "Refined Portfolio" if is_refined_active else "Base Portfolio"

st.markdown(
    f"""
    <div class="card" style="margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="header-title">📈 Financial Risk & Portfolio Optimizer</div>
          <div class="header-sub">Two-step UX: Run Base → then Refine with candidate selection + insights</div>
          <div class="header-sub" style="margin-top:0.35rem;">Currently showing: <b>{active_label}</b></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _get_compare_state_for_charts():
    """
    Chart comparison should follow Insight story:
    - If refined exists: compare Base Portfolio -> Refined Portfolio
    - Else if user provided Current portfolio: compare Current -> Active (base)
    - Else: fallback to baseline (equal weight) only when nothing else exists
    """
    base_state = st.session_state.get("base_state")
    refined_state = st.session_state.get("refined_state")
    active_state = refined_state or base_state

    if refined_state is not None and base_state is not None:
        prev_metrics = (base_state.get("optimized_metrics") or {})
        prev_label = "Base Portfolio"
        return prev_metrics, prev_label

    if active_state is not None:
        cm = active_state.get("current_metrics")
        if cm:
            return cm, "Current"
        bm = active_state.get("baseline_metrics")
        if bm:
            return bm, "Baseline (Equal Weight)"

    return None, None


# ---------------- ACTIVE STATE ----------------
graph_state = st.session_state["refined_state"] or st.session_state["base_state"]
is_refined_active = st.session_state["refined_state"] is not None
active_label = "Refined Portfolio" if is_refined_active else "Base Portfolio"

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
        st.plotly_chart(fig, use_container_width=True)

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

        ret_str = (
            _fmt_pct_from_pct_field(opt.get("return_pct"))
            if opt.get("return_pct") is not None
            else _fmt_pct_from_decimal(_safe_float(opt.get("return")))
        )
        vol_str = (
            _fmt_pct_from_pct_field(opt.get("vol_pct"))
            if opt.get("vol_pct") is not None
            else _fmt_pct_from_decimal(_safe_float(opt.get("vol")))
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{_fmt_num(opt.get("sharpe"))}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Risk-adjusted return</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Return</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{ret_str}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{vol_str}</div>', unsafe_allow_html=True)
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
    st.info("Run **Base Portfolio** first. Then you can refine using candidate selection + insights.")
    st.markdown("</div>", unsafe_allow_html=True)  # close card
else:
    happy_ui = st.radio(
        "Are you happy with this portfolio?",
        ["✅ Yes, this looks good", "❌ No, I’d like to adjust it"],
        index=0,
        horizontal=True,
    )
    is_happy = happy_ui.startswith("✅")

    use_llm_refine = st.checkbox(
        "🤖 Use LLM (choose candidate + generate insights)",
        value=True,
        disabled=is_happy,
        help="When enabled, the model selects the best candidate (Max-Sharpe vs Min-Variance) AND generates portfolio insights.",
    )
    st.caption("News snapshot & risk check is enabled by default.")

    
    st.markdown("---")
    st.markdown('<div class="section-title">📰 News Overview (snapshot + risk)</div>', unsafe_allow_html=True)

    run_news_overview = st.button(
        "📰 Run News Overview (does not change portfolio)",
        use_container_width=True,
        disabled=(st.session_state["base_state"] is None),
    )

    if run_news_overview and selected_tickers:
        # Hangi portfolio “aktif” ise (refined varsa refined, yoksa base) onu referans al
        active_state = st.session_state.get("refined_state") or st.session_state.get("base_state") or {}
        base_state = st.session_state.get("base_state") or {}

        news_overview_state = run_graph(
            selected_tickers=selected_tickers,
            rf=float(rf),
            w_max=float(w_max),
            preferences={},
            current_weights=current_weights_dict,

            mode="refine",
            stage="news_overview",

            # backend zaten stage=news_overview’da force ediyor ama açık yazmak iyi
            use_llm=True,
            use_news=True,

            # ✅ kritik: portfolio seçimi / refine akışı değişmesin diye satisfaction=yes
            clarification_answers={"satisfaction": "yes", "use_news": "yes"},

            # insight için base/ref ilişkisi bozulmasın
            base_portfolio_metrics=(base_state.get("optimized_metrics")),
            base_portfolio_weights=(base_state.get("optimized_weights")),
            base_portfolio_objective=(base_state.get("objective_key")),
        )

        st.session_state["news_overview_state"] = news_overview_state
        st.rerun()
    

    st.markdown("---")
    st.markdown('<div class="section-title">🧠 News → LLM Action List</div>', unsafe_allow_html=True)

    generate_news_actions = st.button(
        "📰 Generate Actions from News",
        use_container_width=True,
        disabled=(st.session_state["base_state"] is None),
    )

    if generate_news_actions and selected_tickers:
        base_state = st.session_state.get("base_state") or {}
        news_actions_state = run_graph(
            selected_tickers=selected_tickers,
            rf=float(rf),
            w_max=float(w_max),
            preferences={},
            current_weights=current_weights_dict,

            # ✅ IMPORTANT: news_actions is a STAGE, not a mode
            mode="refine",
            stage="news_actions",

            use_llm=True,
            use_news=True,

            # (opsiyonel ama iyi) UI stop olmasın diye
            clarification_answers={"satisfaction": "yes"},

            base_portfolio_metrics=base_state.get("optimized_metrics"),
            base_portfolio_weights=base_state.get("optimized_weights"),
            base_portfolio_objective=base_state.get("objective_key"),
        )
        st.session_state["news_actions_state"] = news_actions_state
        st.session_state["selected_news_actions"] = []
        st.rerun()


    nas = st.session_state.get("news_actions_state")
    if nas is not None:
        actions = nas.get("news_actions") or []
        verifier = nas.get("news_actions_verifier")

        # ✅ DEBUG: show what the news_actions run actually produced
        with st.expander("🔍 Debug notes (News Actions run)"):
            for n in (nas.get("debug_notes") or []):
                st.write(f"- {n}")

        with st.expander("🧾 News snapshot text (debug)"):
            st.write(nas.get("news_snapshot_text"))

        with st.expander("🧾 News risk json (debug)"):
            st.json(nas.get("news_risk_json") or {})
        
        if verifier:
            with st.expander("✅ Verifier output"):
                st.json(verifier)

        if not actions:
            st.info("No news actions found. (Backend should return state['news_actions'] list.)")
        else:
            news_items_by_id = _get_news_items_by_id(nas)

            action_labels = []
            for i, a in enumerate(actions):
                if isinstance(a, dict):
                    t = str(a.get("type") or a.get("action") or "").strip()
                    ticker = str(a.get("ticker") or "").strip()
                    value = a.get("value", None)
                    intensity = str(a.get("intensity") or "").strip()
                    reason = str(a.get("reason") or "").strip()

                    parts = []
                    if t:
                        parts.append(t)
                    if ticker:
                        parts.append(ticker)
                    if value is not None:
                        parts.append(f"value={value}")
                    if intensity:
                        parts.append(f"intensity={intensity}")
                    label = " | ".join(parts) if parts else f"Action #{i+1}"

                    if reason:
                        label += f" — {reason}"
                else:
                    label = str(a)

                action_labels.append(label)

            # ✅ 1) Selection UI (aynı kalsın)
            picked = st.multiselect(
                "Select actions to apply in Refine",
                options=action_labels,
                default=st.session_state.get("selected_news_actions", []),
            )
            st.session_state["selected_news_actions"] = picked

            # ✅ 2) Evidence rendering (seçimden bağımsız olarak her action altında göster)
            st.markdown("---")
            st.markdown("#### Proposed actions (with evidence)")

            for i, a in enumerate(actions):
                label = action_labels[i] if i < len(action_labels) else f"Action #{i+1}"
                with st.expander(label, expanded=False):
                    if isinstance(a, dict):
                        st.write(a.get("reason", ""))
                        _render_action_evidence(
                            action=a,
                            news_items_by_id=news_items_by_id,
                            universe_tickers=selected_tickers,  # veya all_tickers
                        )

                    else:
                        st.write(a)
                        
            # ✅ 3) Evidence Snapshot (only evidence referenced by actions)
            st.markdown("---")
            st.markdown("#### 🧾 Evidence Snapshot (linked to actions)")

            ev_text = nas.get("news_evidence_snapshot_text")
            ev_ok = nas.get("news_evidence_snapshot_ok")
            ev_issues = nas.get("news_evidence_snapshot_issues") or []

            if isinstance(ev_ok, bool):
                if ev_ok:
                    st.success("Evidence snapshot generated.")
                else:
                    st.warning("Evidence snapshot had issues (showing best-effort output).")

            if isinstance(ev_text, str) and ev_text.strip():
                # text is already formatted by backend (usually multi-line)
                st.markdown(ev_text)
            else:
                st.caption("No evidence snapshot available (missing `news_evidence_snapshot_text`).")

            if ev_issues:
                with st.expander("⚠️ Evidence snapshot issues"):
                    for it in ev_issues:
                        st.write(f"- {it}")

            with st.expander("Raw proposed actions (debug)"):
                st.json(actions)

            with st.expander("News items by id (debug)"):
                st.write(f"items: {len(news_items_by_id)}")
                st.json(dict(list(news_items_by_id.items())[:5]))


    st.markdown("---")

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

        apply_refine = st.button("✅ Run Candidate Selection (Refine)", use_container_width=True)

        if apply_refine and selected_tickers:
            refined_answers = {
                "satisfaction": "no",
                "pain_points": pain_points,
                "excluded_assets": excluded_assets,
                "use_news": "yes",
                "extra_notes": extra_notes,
                "notes_tickers": notes_tickers,
                "selected_news_actions": st.session_state.get("selected_news_actions", []),
            }

            base_state = st.session_state.get("base_state") or {}
            refined_state = run_graph(
                selected_tickers=selected_tickers,
                rf=float(rf),
                w_max=float(w_max),
                preferences={},
                current_weights=current_weights_dict,
                clarification_answers=refined_answers,
                mode="refine",
                use_llm=bool(use_llm_refine),
                use_news=True,
                base_portfolio_metrics=base_state.get("optimized_metrics"),
                base_portfolio_weights=base_state.get("optimized_weights"),
                base_portfolio_objective=base_state.get("objective_key"),
            )

            st.session_state["refined_state"] = refined_state
            st.success("Refinement applied. Scroll up to see the selected candidate portfolio.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)  # close card

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

                st.write("Insight status:")
                st.write("Base portfolio objective passed to refine:", rs.get("base_portfolio_objective"))
                st.write("Base portfolio metrics present:", rs.get("base_portfolio_metrics") is not None)
                st.write(
                    {
                        "insight_ok": rs.get("insight_ok"),
                        "insight_parse_mode": rs.get("insight_parse_mode"),
                        "insight_issues_n": len(rs.get("insight_issues") or []),
                    }
                )

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
        st.write(f"- Return: {_fmt_pct_from_decimal(_safe_float(base_sum['return']))}")
        st.write(f"- Vol: {_fmt_pct_from_decimal(_safe_float(base_sum['vol']))}")
        st.write(f"- Sharpe: {_fmt_num(base_sum['sharpe'])}")
        st.write(f"- Active assets: {base_sum['active_assets']}")
        st.write(f"- Max weight: {_fmt_pct_from_decimal(_safe_float(base_sum['max_weight']))}")
        st.write(f"- Effective N: {base_sum['effective_n']:.1f}")

    with col2:
        st.markdown("**Refined**")
        if ref_sum is None:
            st.write("Not computed yet.")
        else:
            st.write(f"- Candidate: `{ref_sum['candidate']}`")
            st.write(f"- Return: {_fmt_pct_from_decimal(_safe_float(ref_sum['return']))}")
            st.write(f"- Vol: {_fmt_pct_from_decimal(_safe_float(ref_sum['vol']))}")
            st.write(f"- Sharpe: {_fmt_num(ref_sum['sharpe'])}")
            st.write(f"- Active assets: {ref_sum['active_assets']}")
            st.write(f"- Max weight: {_fmt_pct_from_decimal(_safe_float(ref_sum['max_weight']))}")
            st.write(f"- Effective N: {ref_sum['effective_n']:.1f}")

    with col3:
        st.markdown("**Delta (Refined − Base)**")
        if ref_sum is None:
            st.write("Run refinement to see deltas.")
        else:
            d_ret = _safe_diff(ref_sum.get("return"), base_sum.get("return"))
            d_vol = _safe_diff(ref_sum.get("vol"), base_sum.get("vol"))

            d_sh = (
                (float(ref_sum["sharpe"]) - float(base_sum["sharpe"]))
                if (_safe_float(ref_sum.get("sharpe")) is not None and _safe_float(base_sum.get("sharpe")) is not None)
                else None
            )
            d_eff = float(ref_sum["effective_n"] - base_sum["effective_n"])
            d_mx = _safe_diff(ref_sum.get("max_weight"), base_sum.get("max_weight"))
            d_act = int(ref_sum["active_assets"] - base_sum["active_assets"])

            st.write(f"- Δ Return: {_fmt_pct_from_decimal(d_ret)}")
            st.write(f"- Δ Vol: {_fmt_pct_from_decimal(d_vol)}")
            st.write(f"- Δ Sharpe: {_fmt_num(d_sh)}")
            st.write(f"- Δ Max weight: {_fmt_pct_from_decimal(d_mx)}")
            st.write(f"- Δ Effective N: {d_eff:+.1f}")
            st.write(f"- Δ Active assets: {d_act:+d}")

with st.expander("📦 Export run logs (JSON)"):
    if st.session_state.get("base_state") is not None:
        st.download_button(
            "Download BASE state (JSON)",
            data=json.dumps(st.session_state["base_state"], indent=2, default=str),
            file_name="base_state.json",
            mime="application/json",
            use_container_width=True,
        )
    if st.session_state.get("refined_state") is not None:
        st.download_button(
            "Download REFINED state (JSON)",
            data=json.dumps(st.session_state["refined_state"], indent=2, default=str),
            file_name="refined_state.json",
            mime="application/json",
            use_container_width=True,
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

    chosen = portfolio_metrics["candidate"]
    port = optimization_result.get(chosen, {}) if isinstance(optimization_result, dict) else {}

    x_vol = _safe_float(port.get("vol"))
    y_ret = _safe_float(port.get("return"))

    if optimized_metrics is not None and optimized_metrics:
        x_final = _safe_float(optimized_metrics.get("vol"))
        y_final = _safe_float(optimized_metrics.get("return"))
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

    compare_metrics, compare_label = _get_compare_state_for_charts()
    if compare_metrics is not None and compare_label:
        x_cmp = _safe_float(compare_metrics.get("vol"))
        y_cmp = _safe_float(compare_metrics.get("return"))
        if x_cmp is not None and y_cmp is not None:
            fig_frontier.add_trace(
                go.Scatter(
                    x=[x_cmp],
                    y=[y_cmp],
                    mode="markers+text",
                    name=compare_label,
                    text=[compare_label],
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
    st.plotly_chart(fig_frontier, use_container_width=True)

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
    tickers_rc = list(map(str, optimized_metrics.get("tickers", [])))
    active_rc = np.array(optimized_metrics.get("rc_pct", []), dtype=float)

    if len(tickers_rc) == 0 or len(active_rc) == 0 or len(tickers_rc) != len(active_rc):
        st.info("Risk contribution data is missing or malformed (tickers/rc_pct mismatch).")
    else:
        df_rc = pd.DataFrame({"Ticker": tickers_rc, "Active": active_rc})

        compare_metrics, compare_label = _get_compare_state_for_charts()
        compare_label_used = None
        if compare_metrics is not None and compare_label:
            aligned = _rc_series_aligned_to(tickers_rc, compare_metrics)
            if aligned is not None:
                df_rc[compare_label] = aligned
                compare_label_used = compare_label
            else:
                st.info(f"{compare_label} RC cannot be aligned (needs 'tickers' + 'rc_pct').")

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
        st.plotly_chart(fig_rc, use_container_width=True)

        if compare_label_used:
            st.caption(
                f"Bars show each asset’s share of total portfolio risk (σ). "
                f"Comparison: **{compare_label_used}** vs **{active_label}**."
            )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Bottom: Weights + News + Insight + Explanation ----------------
st.markdown("")
bottom_left, bottom_right = st.columns([1.3, 1.0])

with bottom_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📉 Weights — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("No portfolio yet.")
    else:
        df_weights = portfolio_weights.to_frame("Weight")
        st.dataframe(df_weights.style.format("{:.3f}"), use_container_width=True, height=360)

    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if graph_state is None:
        st.info("News, insight & explanation will appear after base/refine.")
    else:
        # ✅ News panel
        news_state = st.session_state.get("news_overview_state") or graph_state
        _news_section(news_state)

        st.markdown("---")

        # ✅ Insight panel
        _insight_section(graph_state)

        st.markdown("---")
        st.markdown(f'<div class="section-title">💬 Explanation — {active_label}</div>', unsafe_allow_html=True)
        st.write(graph_state.get("explanation", "No explanation generated."))

        with st.expander("🧠 LLM decision (candidate selection)"):
            st.json(graph_state.get("llm_decision", {}))

        with st.expander("📰 News risk (raw JSON)"):
            st.json(graph_state.get("news_risk_json") or graph_state.get("news_signals") or {})


        with st.expander("🔍 Debug notes (graph trace)"):
            notes = graph_state.get("debug_notes", [])
            if not notes:
                st.write("No debug notes.")
            else:
                for n in notes:
                    st.write(f"- {n}")

    st.markdown("</div>", unsafe_allow_html=True)