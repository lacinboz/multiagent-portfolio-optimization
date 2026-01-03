# portfolio_langgraph.py
from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional, Literal

import pandas as pd
from langgraph.graph import StateGraph, END

from agents_langgraph import (
    data_agent_get_mu_cov,
    optimization_agent_from_mu_cov,
    risk_agent,
    recommendation_agent,
)

# ✅ Insight Generator helpers
try:
    from agents_langgraph import insight_agent_prepare  # type: ignore
except Exception:  # pragma: no cover
    insight_agent_prepare = None  # type: ignore

try:
    from llm_client import LLMClient
except Exception:  # pragma: no cover
    LLMClient = None  # type: ignore


PP_TOO_RISKY = "It feels too risky"
PP_TOO_CONSERVATIVE = "It feels too conservative"
PP_TOO_CONCENTRATED = "It’s too concentrated in a few assets"
PP_DISLIKE_ASSETS = "I don’t like some of the assets"
PP_NOT_SURE = "I’m not sure — I just want something safer/smoother"

Mode = Literal["base", "refine"]
ObjectiveKey = Literal["maxsharpe", "minvar"]


class PortfolioState(TypedDict, total=False):
    mode: Mode
    selected_tickers: List[str]
    rf: float
    w_max: float
    lambda_l2: float
    preferences: Dict[str, Any]
    use_llm: bool

    clarification_questions: List[Dict[str, Any]]
    clarification_answers: Optional[Dict[str, Any]]
    needs_user_input: bool

    objective_key: ObjectiveKey
    chosen_candidate: Optional[ObjectiveKey]
    llm_decision: Optional[Dict[str, Any]]

    mu: Optional[pd.Series]
    cov: Optional[pd.DataFrame]
    optimization_result: Dict[str, Any]

    current_weights: Optional[Dict[str, float]]
    baseline_metrics: Optional[Dict[str, Any]]
    current_metrics: Optional[Dict[str, Any]]

    candidates: Dict[str, Dict[str, Any]]

    optimized_weights: Dict[str, float]
    optimized_metrics: Dict[str, Any]

    news_raw: Optional[List[Dict[str, Any]]]
    news_signals: Optional[Dict[str, Any]]

    debug_notes: List[str]
    explanation: str

    # ✅ Insight Generator outputs
    # - insight_raw_text: UI should render this as the narrative "report"
    # - insight: optional JSON (kept for backward compatibility)
    insight: Optional[Dict[str, Any]]
    insight_ok: Optional[bool]
    insight_issues: List[str]
    insight_raw_text: Optional[str]
    insight_parse_mode: Optional[str]

    # ✅ carry the user's previous portfolio (Run Base output) into refine run
    base_portfolio_weights: Optional[Dict[str, float]]
    base_portfolio_metrics: Optional[Dict[str, Any]]
    base_portfolio_objective: Optional[str]


# =========================================================
# Defaults / prefs
# =========================================================
def _init_defaults(state: PortfolioState) -> PortfolioState:
    state.setdefault("mode", "refine")
    state.setdefault("rf", 0.02)
    state.setdefault("w_max", 0.30)
    state.setdefault("lambda_l2", 1e-3)
    state.setdefault("objective_key", "maxsharpe")
    state.setdefault("preferences", {})
    state.setdefault("use_llm", False)
    state.setdefault("current_weights", None)

    state.setdefault("clarification_questions", [])
    state.setdefault("clarification_answers", None)
    state.setdefault("needs_user_input", False)

    state.setdefault("mu", None)
    state.setdefault("cov", None)
    state.setdefault("optimization_result", {})

    state.setdefault("baseline_metrics", None)
    state.setdefault("current_metrics", None)

    state.setdefault("candidates", {})
    state.setdefault("chosen_candidate", None)
    state.setdefault("llm_decision", None)

    state.setdefault("optimized_weights", {})
    state.setdefault("optimized_metrics", {})

    state.setdefault("news_raw", None)
    state.setdefault("news_signals", None)

    state.setdefault("debug_notes", [])
    state.setdefault("explanation", "")

    # ✅ Insight outputs
    state.setdefault("insight", None)
    state.setdefault("insight_ok", None)
    state.setdefault("insight_issues", [])
    state.setdefault("insight_raw_text", None)
    state.setdefault("insight_parse_mode", None)

    # ✅ Base portfolio from previous run (optional)
    state.setdefault("base_portfolio_weights", None)
    state.setdefault("base_portfolio_metrics", None)
    state.setdefault("base_portfolio_objective", None)

    return state


def _merged_prefs(state: PortfolioState) -> Dict[str, Any]:
    return (state.get("clarification_answers") or state.get("preferences") or {}) or {}


# =========================================================
# UI questions
# =========================================================
def _build_default_questions(state: PortfolioState) -> List[Dict[str, Any]]:
    n = len(state.get("selected_tickers", []))
    return [
        {
            "id": "satisfaction",
            "type": "select",
            "label": "Are you happy with this portfolio?",
            "options": ["yes", "no"],
            "option_labels": ["Yes, looks good", "No, adjust it"],
            "default": "yes",
        },
        {
            "id": "pain_points",
            "type": "multiselect",
            "label": "What doesn’t feel right? (optional)",
            "options": [PP_TOO_RISKY, PP_TOO_CONSERVATIVE, PP_TOO_CONCENTRATED, PP_DISLIKE_ASSETS, PP_NOT_SURE],
            "default": [],
        },
        {
            "id": "excluded_assets",
            "type": "multiselect",
            "label": "Exclude specific assets (optional)",
            "options": state.get("selected_tickers", []),
            "default": [],
            "help": f"Universe size: {n}. Excluding assets removes them from optimization.",
        },
        {
            "id": "extra_notes",
            "type": "text",
            "label": "Extra notes (optional)",
            "default": "",
        },
    ]


# =========================================================
# Metrics helpers
# =========================================================
def _extract_active_weights(weights_all: Dict[str, Any]) -> Dict[str, float]:
    return {t: float(v) for t, v in (weights_all or {}).items() if abs(float(v)) > 1e-6}


def _safe_max_weight(weights: Dict[str, float]) -> float:
    return max(weights.values()) if weights else 0.0


def _effective_n(weights: Dict[str, float]) -> float:
    if not weights:
        return 0.0
    s = sum(float(w) ** 2 for w in weights.values())
    return float(1.0 / s) if s > 0 else 0.0


def _attach_concentration_metrics(metrics: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    out = dict(metrics or {})
    out["max_weight"] = _safe_max_weight(weights)
    out["effective_n"] = _effective_n(weights)
    out["active_assets"] = int(out.get("active_assets") or len([w for w in weights.values() if abs(w) > 1e-6]))
    return out


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def _normalize_metrics(m: Dict[str, Any], *, rf: float) -> Dict[str, Any]:
    out = dict(m or {})
    r = _as_float(out.get("return"))
    v = _as_float(out.get("vol"))
    s = _as_float(out.get("sharpe"))

    if s is None and (r is not None) and (v is not None) and v > 0:
        s = (r - float(rf)) / v
        out["sharpe"] = s

    out["return_pct"] = (r * 100.0) if r is not None else None
    out["vol_pct"] = (v * 100.0) if v is not None else None

    mw = _as_float(out.get("max_weight"))
    out["max_weight_pct"] = (mw * 100.0) if mw is not None else None

    out["rf"] = float(rf)
    return out


# =========================================================
# Nodes
# =========================================================
def node_ask_clarifications(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    if state.get("mode") == "base":
        state["needs_user_input"] = False
        state["debug_notes"].append("Clarifications(BASE): skipped (base run is non-interactive).")
        return state

    if state.get("clarification_answers") is not None:
        state["needs_user_input"] = False
        state["debug_notes"].append("Clarifications(REFINE): answers present → continue.")
        return state

    state["clarification_questions"] = _build_default_questions(state)
    state["needs_user_input"] = True
    state["debug_notes"].append(
        f"Clarifications(REFINE): generated {len(state['clarification_questions'])} questions → stop for user input."
    )
    return state


def route_after_clarifications(state: PortfolioState) -> str:
    return "end" if state.get("needs_user_input") else "perception"


def node_perception(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)
    prefs = _merged_prefs(state)

    if state.get("mode") == "base":
        state["debug_notes"].append(
            f"Perception(BASE): objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f}, "
            f"lambda_l2={float(state['lambda_l2']):.4g}"
        )
        return state

    satisfaction = str(prefs.get("satisfaction") or "").lower().strip()

    excluded_assets = prefs.get("excluded_assets") or []
    if excluded_assets:
        excluded = set(map(str, excluded_assets))
        state["selected_tickers"] = [t for t in list(state.get("selected_tickers", [])) if t not in excluded]
        state["debug_notes"].append(f"Perception: excluded={sorted(excluded)}")

    extra_notes = str(prefs.get("extra_notes") or "").strip()
    pain_points = prefs.get("pain_points") or []
    pain_points_n = len(pain_points) if isinstance(pain_points, list) else 1

    state["debug_notes"].append(
        f"Perception(REFINE): satisfaction={satisfaction or '∅'}, pain_points={pain_points_n}, "
        f"extra_notes={'yes' if extra_notes else 'no'}, n={len(state.get('selected_tickers', []))}"
    )
    return state


def node_compute_baselines(state: PortfolioState) -> PortfolioState:
    tickers = state.get("selected_tickers", [])

    state["baseline_metrics"] = None
    state["current_metrics"] = None

    if tickers:
        ew = {t: 1.0 / len(tickers) for t in tickers}
        try:
            bm = risk_agent(ew, tickers, rf=float(state["rf"]))
            bm = _attach_concentration_metrics(bm, _extract_active_weights(ew))
            state["baseline_metrics"] = _normalize_metrics(bm, rf=float(state["rf"]))
        except Exception as e:
            state["baseline_metrics"] = None
            state["debug_notes"].append(f"Baseline metrics failed: {e}")

    if state.get("current_weights") is not None and tickers:
        try:
            cm = risk_agent(state["current_weights"], tickers, rf=float(state["rf"]))
            cm = _attach_concentration_metrics(cm, _extract_active_weights(state["current_weights"]))
            state["current_metrics"] = _normalize_metrics(cm, rf=float(state["rf"]))
        except Exception as e:
            state["current_metrics"] = None
            state["debug_notes"].append(f"Current metrics failed: {e}")

    return state


def node_data(state: PortfolioState) -> PortfolioState:
    state["mu"], state["cov"] = None, None

    tickers = state.get("selected_tickers", [])
    if not tickers:
        state["debug_notes"].append("Data: skipped (no tickers).")
        return state

    try:
        mu, cov = data_agent_get_mu_cov(tickers)
        state["mu"], state["cov"] = mu, cov
        state["debug_notes"].append(f"Data: loaded mu/cov for n={len(mu)}")
    except Exception as e:
        state["mu"], state["cov"] = None, None
        state["debug_notes"].append(f"Data: failed → {e}")

    return state


def node_optimize(state: PortfolioState) -> PortfolioState:
    if state.get("mu") is None or state.get("cov") is None:
        state["optimization_result"] = {}
        state["debug_notes"].append("Optimization: skipped (missing mu/cov).")
        return state

    res = optimization_agent_from_mu_cov(
        mu=state["mu"],
        cov=state["cov"],
        rf=float(state["rf"]),
        w_max=float(state["w_max"]),
        lambda_l2=float(state["lambda_l2"]),
    )
    state["optimization_result"] = res
    state["debug_notes"].append("Optimization: done (mu/cov).")
    return state


def node_extract_candidates(state: PortfolioState) -> PortfolioState:
    state["candidates"] = {}

    res = state.get("optimization_result") or {}
    if not res:
        state["debug_notes"].append("ExtractCandidates: skipped (missing optimization_result).")
        return state

    mode = state.get("mode", "refine")

    if mode == "base":
        obj = state.get("objective_key", "maxsharpe")
        if obj in res:
            w = _extract_active_weights(res[obj].get("weights", {}))
            state["candidates"][obj] = {"weights": w, "metrics": None}
            state["chosen_candidate"] = obj
            state["debug_notes"].append(
                f"Extract(BASE): objective={obj} active={len(w)} max_w={_safe_max_weight(w):.4f}"
            )
        else:
            state["debug_notes"].append(f"Extract(BASE): objective '{obj}' not found.")
        return state

    for obj in ("maxsharpe", "minvar"):
        if obj in res:
            w = _extract_active_weights(res[obj].get("weights", {}))
            state["candidates"][obj] = {"weights": w, "metrics": None}
            state["debug_notes"].append(f"ExtractCandidates: {obj} active={len(w)} max_w={_safe_max_weight(w):.4f}")

    if not state["candidates"]:
        state["debug_notes"].append("ExtractCandidates: none available.")
    return state


def node_risk_candidates(state: PortfolioState) -> PortfolioState:
    tickers = state.get("selected_tickers", [])
    cands = state.get("candidates") or {}
    if not tickers or not cands:
        state["debug_notes"].append("RiskCandidates: skipped (missing tickers or candidates).")
        return state

    for k, item in cands.items():
        w = item.get("weights") or {}
        if not w:
            item["metrics"] = {}
            continue
        try:
            m = risk_agent(w, tickers, rf=float(state["rf"]))
            m = _attach_concentration_metrics(m, w)
            item["metrics"] = _normalize_metrics(m, rf=float(state["rf"]))
        except Exception as e:
            item["metrics"] = {}
            state["debug_notes"].append(f"RiskCandidates: failed for {k}: {e}")

    state["debug_notes"].append("RiskCandidates: computed metrics for candidates.")
    return state


def node_news_fetch(state: PortfolioState) -> PortfolioState:
    tickers = state.get("selected_tickers", [])
    state["news_raw"] = [{"ticker": t, "headline": None, "source": None, "ts": None} for t in tickers]
    state["debug_notes"].append(f"NewsFetch: collected raw items for n={len(tickers)} (stub).")
    return state


def node_news_signals_placeholder(state: PortfolioState) -> PortfolioState:
    raw = state.get("news_raw") or []
    signals: Dict[str, Any] = {"by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}

    keywords = ("lawsuit", "fraud", "bankrupt", "guidance cut", "downgrade", "shock")
    for item in raw:
        t = item.get("ticker")
        h = (item.get("headline") or "").lower()
        if t:
            risk_flag = "none"
            conf = 0.0
            if any(k in h for k in keywords):
                risk_flag = "event_risk"
                conf = 0.6
                signals["global"]["risk_flags"].append({"ticker": t, "flag": "event_risk"})
                signals["global"]["vol_regime"] = "high"
            signals["by_ticker"][t] = {"risk_flag": risk_flag, "confidence": conf}

    state["news_signals"] = signals
    state["debug_notes"].append("NewsSignals(placeholder): produced signals.")
    return state


def node_llm_select_candidate(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    if state.get("mode") == "base":
        chosen = state.get("chosen_candidate") or state.get("objective_key", "maxsharpe")
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "Base run: candidate selection disabled. Portfolio generated for comparison.",
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append("LLM_Select(BASE): accept (no selection).")
        return state

    prefs = _merged_prefs(state)
    satisfaction = str(prefs.get("satisfaction") or "").lower().strip()

    candidates = state.get("candidates") or {}
    if not candidates:
        chosen = state.get("objective_key", "maxsharpe")
        state["chosen_candidate"] = chosen  # type: ignore
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "No candidates available; cannot select.",
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append("LLM_Select: no candidates -> accept fallback.")
        return state

    if satisfaction == "yes":
        chosen = state.get("objective_key", "maxsharpe")
        if chosen not in candidates:
            chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
        state["chosen_candidate"] = chosen  # type: ignore
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "User indicated satisfaction=yes; skipping candidate selection.",
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append("LLM_Select: satisfaction=yes -> accept.")
        return state

    if satisfaction != "no":
        chosen = state.get("objective_key", "maxsharpe")
        if chosen not in candidates:
            chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
        state["chosen_candidate"] = chosen  # type: ignore
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "No explicit dissatisfaction; skipping candidate selection.",
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append("LLM_Select: satisfaction not 'no' -> accept.")
        return state

    use_llm = bool(state.get("use_llm", False))

    if use_llm and LLMClient is not None:
        try:
            client = LLMClient()
            llm_payload = client.select_candidate(
                mode=str(state.get("mode")),
                objective_key=str(state.get("objective_key")),
                rf=float(state.get("rf")),
                w_max=float(state.get("w_max")),
                lambda_l2=float(state.get("lambda_l2")),
                selected_tickers=list(state.get("selected_tickers", [])),
                candidates=candidates,
                baseline_metrics=state.get("baseline_metrics"),
                current_metrics=state.get("current_metrics"),
                preferences=prefs,
                news_signals=state.get("news_signals"),
            )

            chosen = str(llm_payload.get("chosen_candidate", "")).lower().strip()
            if chosen not in candidates:
                chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))

            rationale = str(llm_payload.get("rationale", "")).strip() or (
                "LLM selected the most preference-aligned candidate."
            )
            state["chosen_candidate"] = chosen  # type: ignore
            state["llm_decision"] = {"decision": "accept", "rationale": rationale, "chosen_candidate": chosen}
            state["debug_notes"].append(f"LLM_Select(LLM): chosen={chosen}")
            return state

        except Exception as e:
            state["debug_notes"].append(f"LLM_Select(LLM): failed → fallback: {e}")

    chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
    state["chosen_candidate"] = chosen  # type: ignore
    state["llm_decision"] = {
        "decision": "accept",
        "rationale": "LLM disabled/unavailable; defaulting to a deterministic candidate.",
        "chosen_candidate": chosen,
    }
    state["debug_notes"].append(f"LLM_Select(Fallback): chosen={chosen}")
    return state


def node_finalize_selection(state: PortfolioState) -> PortfolioState:
    chosen = state.get("chosen_candidate") or state.get("objective_key", "maxsharpe")
    candidates = state.get("candidates") or {}

    if chosen in candidates:
        state["optimized_weights"] = candidates[chosen].get("weights") or {}
        state["optimized_metrics"] = candidates[chosen].get("metrics") or {}
        state["objective_key"] = chosen
        state["debug_notes"].append(
            f"FinalizeSelection: chosen={chosen}, active={len(state['optimized_weights'])}, "
            f"max_w={_safe_max_weight(state['optimized_weights']):.4f}"
        )
    else:
        state["optimized_weights"] = {}
        state["optimized_metrics"] = {}
        state["debug_notes"].append(f"FinalizeSelection: chosen candidate '{chosen}' missing -> empty result.")
    return state


# =========================================================
# ✅ Insight Generator Node (Narrative mode, base-vs-refine semantics)
# =========================================================
def node_insight_generator(state: PortfolioState) -> PortfolioState:
    """
    Insight MUST compare:
      Base Portfolio (user's previous portfolio from Run Base)  ->  Refined Portfolio (chosen candidate)
    NOT equal-weight baseline, unless base portfolio wasn't provided.

    ✅ FIX: generate narrative text (not strict JSON) so UI can display a "report".
    """
    state = _init_defaults(state)

    use_llm = bool(state.get("use_llm", False))
    if (not use_llm) or (LLMClient is None) or (insight_agent_prepare is None):
        state["debug_notes"].append("Insight: skipped (use_llm disabled or LLMClient/insight_agent_prepare unavailable).")
        return state

    refine_metrics = state.get("optimized_metrics") or {}
    if not refine_metrics:
        state["debug_notes"].append("Insight: skipped (missing optimized_metrics).")
        return state

    # 1) Prefer REAL previous portfolio (Run Base output)
    base_metrics = state.get("base_portfolio_metrics")
    base_obj = state.get("base_portfolio_objective")

    # 2) If user entered current portfolio
    if not base_metrics:
        base_metrics = state.get("current_metrics")
        if base_metrics and not base_obj:
            base_obj = "user_current"

    # 3) Last resort: equal-weight baseline
    if not base_metrics:
        base_metrics = state.get("baseline_metrics") or {}
        if not base_obj:
            base_obj = "equal_weight"

    prefs = _merged_prefs(state)
    news_signals = state.get("news_signals")

    chosen = str(state.get("objective_key") or "maxsharpe").lower().strip()
    refine_obj = chosen

    try:
        # Optional tightening: keep base_constraints minimal if base is "real previous"
        base_constraints = {"rf": float(state.get("rf", 0.02))}
        refine_constraints = {
            "rf": float(state.get("rf", 0.02)),
            "w_max": float(state.get("w_max", 0.30)),
            "lambda_l2": float(state.get("lambda_l2", 1e-3)),
        }

        prep = insight_agent_prepare(
            base_metrics=base_metrics,
            refine_metrics=refine_metrics,
            preferences=prefs,
            news_signals=news_signals,
            base_objective=base_obj,
            refine_objective=refine_obj,
            base_constraints=base_constraints,
            refine_constraints=refine_constraints,
        )

        prompts = prep.get("prompts") or {}
        payload = prep.get("payload") or {}

        client = LLMClient()

        # ✅ IMPORTANT: narrative mode (LLM returns a long text report)
        # This requires llm_client.generate_portfolio_insights to accept mode="narrative"
        out = client.generate_portfolio_insights(
            prompts=prompts,
            payload=payload,
            mode="narrative",
        )

        state["insight_ok"] = bool(out.get("ok"))
        state["insight_issues"] = list(out.get("issues") or [])
        state["insight_parse_mode"] = out.get("parse_mode") or "narrative"

        # ✅ UI should render this
        state["insight_raw_text"] = (out.get("text") or out.get("raw_text") or "").strip() or None

        # Keep JSON insight optional (not used in narrative mode)
        state["insight"] = out.get("insight") if isinstance(out.get("insight"), dict) else None

        state["debug_notes"].append(
            f"Insight: generated ok={state['insight_ok']} issues={len(state['insight_issues'])} mode={state['insight_parse_mode']}"
        )
        return state

    except Exception as e:
        state["insight_ok"] = False
        state["insight_raw_text"] = None
        state["insight"] = None
        state["insight_issues"] = [f"insight_exception: {e}"]
        state["insight_parse_mode"] = "error"
        state["debug_notes"].append(f"Insight: failed → {e}")
        return state


def node_explain(state: PortfolioState) -> PortfolioState:
    """
    - Always narrate the chosen objective
    - Use final_metrics=optimized_metrics so numbers match UI (single source of truth)
    """
    if not state.get("optimization_result"):
        state["explanation"] = "No optimization result available (empty universe)."
        state["debug_notes"].append("Explain: skipped (no optimization_result).")
        return state

    chosen = state.get("objective_key") or "maxsharpe"
    obj = "max_sharpe" if chosen == "maxsharpe" else "min_var"

    text = recommendation_agent(
        state["optimization_result"],
        objective=obj,
        current_metrics=state.get("current_metrics"),
        rf=float(state["rf"]),
        preferences=_merged_prefs(state),
        final_metrics=state.get("optimized_metrics") or None,  # ✅ IMPORTANT
    )

    om = state.get("optimized_metrics") or {}
    r_pct = om.get("return_pct")
    v_pct = om.get("vol_pct")
    s = om.get("sharpe")

    if isinstance(r_pct, (int, float)) and isinstance(v_pct, (int, float)):
        extra = f"\n\n(Selected candidate metrics: return {float(r_pct):.1f}%, vol {float(v_pct):.1f}%"
        if isinstance(s, (int, float)):
            extra += f", Sharpe {float(s):.2f})"
        else:
            extra += ")"
        text += extra

    state["explanation"] = text
    state["debug_notes"].append(f"Explain: generated (chosen_candidate={chosen}, objective_str={obj}).")
    return state


# =========================================================
# Graph wiring
# =========================================================
def build_portfolio_graph():
    g = StateGraph(PortfolioState)

    g.add_node("ask_clarifications", node_ask_clarifications)
    g.add_node("perception", node_perception)
    g.add_node("baselines", node_compute_baselines)

    g.add_node("data", node_data)
    g.add_node("optimize", node_optimize)

    g.add_node("extract_candidates", node_extract_candidates)
    g.add_node("risk_candidates", node_risk_candidates)

    g.add_node("news_fetch", node_news_fetch)
    g.add_node("news_signals", node_news_signals_placeholder)

    g.add_node("llm_select", node_llm_select_candidate)
    g.add_node("finalize", node_finalize_selection)

    g.add_node("insight", node_insight_generator)
    g.add_node("explain", node_explain)

    g.set_entry_point("ask_clarifications")

    g.add_conditional_edges(
        "ask_clarifications",
        route_after_clarifications,
        {"end": END, "perception": "perception"},
    )

    g.add_edge("perception", "baselines")
    g.add_edge("baselines", "data")
    g.add_edge("data", "optimize")

    g.add_edge("optimize", "extract_candidates")
    g.add_edge("extract_candidates", "risk_candidates")

    g.add_edge("risk_candidates", "news_fetch")
    g.add_edge("news_fetch", "news_signals")
    g.add_edge("news_signals", "llm_select")
    g.add_edge("llm_select", "finalize")

    g.add_edge("finalize", "insight")
    g.add_edge("insight", "explain")
    g.add_edge("explain", END)

    return g.compile()


def run_graph(
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    preferences: Optional[Dict[str, Any]] = None,
    current_weights: Optional[Dict[str, float]] = None,
    max_iterations: int = 0,
    clarification_answers: Optional[Dict[str, Any]] = None,
    mode: Mode = "refine",
    use_llm: bool = False,
    # ✅ pass base portfolio from the previous Run Base
    base_portfolio_metrics: Optional[Dict[str, Any]] = None,
    base_portfolio_weights: Optional[Dict[str, float]] = None,
    base_portfolio_objective: Optional[str] = None,
) -> PortfolioState:
    app = build_portfolio_graph()

    init: PortfolioState = {
        "mode": mode,
        "selected_tickers": selected_tickers,
        "rf": float(rf),
        "w_max": float(w_max),
        "lambda_l2": 1e-3,
        "preferences": preferences or {},
        "use_llm": bool(use_llm),
        "current_weights": current_weights,
        "debug_notes": [],
        "clarification_answers": clarification_answers,
        "objective_key": "maxsharpe",
        "chosen_candidate": None,
        "candidates": {},
        "llm_decision": None,
        "optimized_weights": {},
        "optimized_metrics": {},
        "insight": None,
        "insight_ok": None,
        "insight_issues": [],
        "insight_raw_text": None,
        "insight_parse_mode": None,
        # ✅ base portfolio injection
        "base_portfolio_metrics": base_portfolio_metrics,
        "base_portfolio_weights": base_portfolio_weights,
        "base_portfolio_objective": base_portfolio_objective,
    }

    return app.invoke(init, config={"recursion_limit": 200})
