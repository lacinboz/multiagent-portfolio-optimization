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

# ----------------------------
# State enums
# ----------------------------
Stability = Literal["stable", "balanced", "swingy"]
Concentration = Literal["low", "high"]
Goal = Literal["best_tradeoff", "lowest_risk"]
RiskFeedback = Literal["too_risky", "ok", "too_conservative"]
Mode = Literal["base", "refine"]  # ✅ new


# ----------------------------
# State
# ----------------------------
class PortfolioState(TypedDict, total=False):
    # inputs
    mode: Mode
    selected_tickers: List[str]
    rf: float
    w_max: float
    lambda_l2: float
    preferences: Dict[str, Any]

    # interactive loop
    clarification_questions: List[Dict[str, Any]]
    clarification_answers: Optional[Dict[str, Any]]
    needs_user_input: bool

    # derived decisions
    objective_key: Literal["maxsharpe", "minvar"]

    # computed / intermediate
    mu: Optional[pd.Series]
    cov: Optional[pd.DataFrame]

    current_weights: Optional[Dict[str, float]]
    current_metrics: Optional[Dict[str, Any]]
    baseline_metrics: Optional[Dict[str, Any]]

    optimization_result: Dict[str, Any]
    optimized_weights: Dict[str, float]
    optimized_metrics: Dict[str, Any]

    # news / context
    news_raw: Optional[List[Dict[str, Any]]]
    news_signals: Optional[Dict[str, Any]]

    # loop control (refine loop)
    iteration: int
    max_iterations: int
    needs_refine: bool
    refine_actions: List[Dict[str, Any]]
    llm_decision: Optional[Dict[str, Any]]
    changes_applied: List[Dict[str, Any]]
    changes_rejected: List[Dict[str, Any]]
    debug_notes: List[str]

    # output
    explanation: str


def _init_defaults(state: PortfolioState) -> PortfolioState:
    state.setdefault("mode", "refine")

    state.setdefault("rf", 0.02)
    state.setdefault("w_max", 0.30)
    state.setdefault("lambda_l2", 1e-3)
    state.setdefault("objective_key", "maxsharpe")
    state.setdefault("preferences", {})

    state.setdefault("clarification_questions", [])
    state.setdefault("clarification_answers", None)
    state.setdefault("needs_user_input", False)

    state.setdefault("mu", None)
    state.setdefault("cov", None)

    state.setdefault("news_raw", None)
    state.setdefault("news_signals", None)

    state.setdefault("iteration", 0)
    state.setdefault("max_iterations", 2)
    state.setdefault("needs_refine", False)
    state.setdefault("refine_actions", [])
    state.setdefault("llm_decision", None)
    state.setdefault("changes_applied", [])
    state.setdefault("changes_rejected", [])
    state.setdefault("debug_notes", [])
    state.setdefault("optimization_result", {})
    state.setdefault("optimized_weights", {})
    state.setdefault("optimized_metrics", {})
    state.setdefault("baseline_metrics", None)
    state.setdefault("current_metrics", None)
    state.setdefault("explanation", "")
    return state


# ----------------------------
# Helper: build questions
# ----------------------------
def _build_default_questions(state: PortfolioState) -> List[Dict[str, Any]]:
    n = len(state.get("selected_tickers", []))
    return [
        {
            "id": "goal",
            "type": "select",
            "label": "What’s your main goal?",
            "options": ["best_tradeoff", "lowest_risk"],
            "option_labels": ["Best balance of risk & return", "Lowest risk possible"],
            "default": "best_tradeoff",
        },
        {
            "id": "stability",
            "type": "select",
            "label": "How stable should the portfolio feel?",
            "options": ["balanced", "stable", "swingy"],
            "option_labels": ["Balanced", "Very stable", "More swingy"],
            "default": "balanced",
        },
        {
            "id": "concentration",
            "type": "select",
            "label": "Do you want the portfolio to be diversified?",
            "options": ["low", "high"],
            "option_labels": ["Yes, keep it diversified", "I’m OK with a few big positions"],
            "default": "low",
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
            "id": "risk_feedback",
            "type": "select",
            "label": "Optional feedback for a refine loop",
            "options": ["", "too_risky", "ok", "too_conservative"],
            "option_labels": ["No feedback yet", "This feels too risky", "Looks OK", "This feels too conservative"],
            "default": "",
            "help": "If you pick too_risky/too_conservative, the graph may refine before finalizing.",
        },
    ]


# ----------------------------
# Nodes: ask clarifications
# ----------------------------
def node_ask_clarifications(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    # ✅ BASE MODE: never stop. (We also don't need to inject answers.)
    if state.get("mode") == "base":
        state["needs_user_input"] = False
        state["debug_notes"].append("Clarifications(BASE): skipped (base run is non-interactive).")
        return state

    # ✅ REFINE MODE: interactive behavior
    if state.get("clarification_answers") is not None:
        state["needs_user_input"] = False
        state["debug_notes"].append("Clarifications(REFINE): answers present -> continue.")
        return state

    state["clarification_questions"] = _build_default_questions(state)
    state["needs_user_input"] = True
    state["debug_notes"].append(
        f"Clarifications(REFINE): generated {len(state['clarification_questions'])} questions -> stop for user input."
    )
    return state


def route_after_clarifications(state: PortfolioState) -> str:
    return "end" if state.get("needs_user_input") else "perception"


# ----------------------------
# Node: perception (prefs -> constraints)
# ----------------------------
def node_perception(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    # ✅ KEY FIX:
    # Base run should NOT apply any preference-to-constraint nudges.
    # It should respect the slider inputs (rf/w_max) and default objective_key.
    if state.get("mode") == "base":
        state["debug_notes"].append(
            f"Perception(BASE): skipping nudges (objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f})."
        )
        return state

    answers = state.get("clarification_answers") or {}
    prefs = state.get("preferences", {}) or {}

    goal: Optional[Goal] = answers.get("goal") or prefs.get("goal")
    stability: Optional[Stability] = answers.get("stability") or prefs.get("stability")
    concentration: Optional[Concentration] = answers.get("concentration") or prefs.get("concentration")
    excluded_assets = answers.get("excluded_assets") or prefs.get("excluded_assets") or []

    selected = list(state.get("selected_tickers", []))

    excluded = set(excluded_assets)
    if excluded:
        selected = [t for t in selected if t not in excluded]
        state["selected_tickers"] = selected
        state["debug_notes"].append(f"Perception: excluded={sorted(excluded)}")

    # Goal -> objective
    if goal == "lowest_risk":
        state["objective_key"] = "minvar"
    elif goal == "best_tradeoff":
        state["objective_key"] = "maxsharpe"

    # Stability -> objective nudge
    if stability == "stable":
        state["objective_key"] = "minvar"

    # Concentration -> w_max mapping
    if concentration == "low":
        state["w_max"] = min(float(state["w_max"]), 0.20)
    elif concentration == "high":
        state["w_max"] = max(float(state["w_max"]), 0.35)

    state["debug_notes"].append(
        f"Perception: objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f}, n={len(state.get('selected_tickers', []))}"
    )
    return state


# ----------------------------
# Node: baselines
# ----------------------------
def node_compute_baselines(state: PortfolioState) -> PortfolioState:
    tickers = state.get("selected_tickers", [])

    state["baseline_metrics"] = None
    state["current_metrics"] = None

    if tickers:
        ew = {t: 1.0 / len(tickers) for t in tickers}
        try:
            state["baseline_metrics"] = risk_agent(ew, tickers)
        except Exception as e:
            state["baseline_metrics"] = None
            state["debug_notes"].append(f"Baseline metrics failed: {e}")

    if state.get("current_weights") is not None and tickers:
        try:
            state["current_metrics"] = risk_agent(state["current_weights"], tickers)
        except Exception as e:
            state["current_metrics"] = None
            state["debug_notes"].append(f"Current metrics failed: {e}")

    return state


# ----------------------------
# Node: data
# ----------------------------
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
        state["debug_notes"].append(f"Data: failed -> {e}")

    return state


# ----------------------------
# Node: optimization
# ----------------------------
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


def node_extract_portfolio(state: PortfolioState) -> PortfolioState:
    if not state.get("optimization_result"):
        state["optimized_weights"] = {}
        state["debug_notes"].append("Extract: skipped (missing optimization_result).")
        return state

    obj = state.get("objective_key", "maxsharpe")
    if obj not in state["optimization_result"]:
        state["optimized_weights"] = {}
        state["debug_notes"].append(f"Extract: skipped (objective '{obj}' not found).")
        return state

    port = state["optimization_result"][obj]
    w_all = port.get("weights", {})
    weights = {t: float(w) for t, w in w_all.items() if abs(float(w)) > 1e-6}

    state["optimized_weights"] = weights
    state["debug_notes"].append(f"Extract: objective={obj}, active={len(weights)}")
    return state


def node_risk(state: PortfolioState) -> PortfolioState:
    if not state.get("optimized_weights") or not state.get("selected_tickers"):
        state["optimized_metrics"] = {}
        state["debug_notes"].append("Risk: skipped (missing weights or tickers).")
        return state

    state["optimized_metrics"] = risk_agent(state["optimized_weights"], state["selected_tickers"])
    state["debug_notes"].append("Risk: computed optimized metrics.")
    return state


# ----------------------------
# Node: news fetcher (stub)
# ----------------------------
def node_news_fetch(state: PortfolioState) -> PortfolioState:
    tickers = state.get("selected_tickers", [])
    state["news_raw"] = [{"ticker": t, "headline": None, "source": None, "ts": None} for t in tickers]
    state["debug_notes"].append(f"NewsFetch: collected raw items for n={len(tickers)} (stub).")
    return state


# ----------------------------
# Node: news -> signals (LLM-visible; placeholder)
# ----------------------------
def node_news_signals_llm(state: PortfolioState) -> PortfolioState:
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
    state["debug_notes"].append("NewsSignals(LLM): produced signals (placeholder).")
    return state


# ----------------------------
# Node: LLM evaluation / refinement controller
# ----------------------------
def node_llm_evaluate(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    # ✅ BASE MODE: always accept, no refine loop
    if state.get("mode") == "base":
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "Base run: refinement disabled. Portfolio generated for comparison.",
            "proposed_actions": [],
        }
        state["debug_notes"].append("LLM_Evaluate(BASE): accept (no refine).")
        state["debug_notes"].append(
            f"LLM_Evaluate: iteration={state.get('iteration')} max_iterations={state.get('max_iterations')} mode={state.get('mode')}"
        )

        return state

    answers = state.get("clarification_answers") or {}
    prefs = state.get("preferences", {}) or {}

    risk_feedback: Optional[RiskFeedback] = (answers.get("risk_feedback") or prefs.get("risk_feedback") or None) or None
    concentration: Optional[Concentration] = answers.get("concentration") or prefs.get("concentration")

    actions: List[Dict[str, Any]] = []
    needs_refine = False
    rationale_parts: List[str] = []

    if int(state["iteration"]) >= int(state["max_iterations"]):
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {"decision": "accept", "rationale": "Max iterations reached.", "proposed_actions": []}
        state["debug_notes"].append("LLM_Evaluate: max iterations reached -> accept.")
        return state

    if not state.get("optimized_weights"):
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {"decision": "accept", "rationale": "No optimized weights; stopping.", "proposed_actions": []}
        state["debug_notes"].append("LLM_Evaluate: no weights -> accept.")
        return state

    # News-aware trigger
    news_signals = state.get("news_signals") or {}
    vol_regime = ((news_signals.get("global") or {}).get("vol_regime")) or "normal"
    if vol_regime == "high":
        needs_refine = True
        rationale_parts.append("News signals suggest elevated event/volatility regime.")
        actions.append({"type": "set_objective_key", "value": "minvar"})
        actions.append({"type": "set_w_max", "value": max(0.10, float(state["w_max"]) - 0.05)})

    # Preference-aware triggers
    if risk_feedback == "too_risky":
        needs_refine = True
        rationale_parts.append("User feedback: too risky.")
        actions.append({"type": "set_objective_key", "value": "minvar"})
        actions.append({"type": "set_w_max", "value": max(0.10, float(state["w_max"]) - 0.05)})

    if risk_feedback == "too_conservative":
        needs_refine = True
        rationale_parts.append("User feedback: too conservative.")
        actions.append({"type": "set_objective_key", "value": "maxsharpe"})
        actions.append({"type": "set_w_max", "value": min(0.50, float(state["w_max"]) + 0.05)})

    if concentration == "low":
        max_w = max(state["optimized_weights"].values()) if state["optimized_weights"] else 0.0
        if max_w > 0.22:
            needs_refine = True
            rationale_parts.append(f"Concentration too high (max weight {max_w:.2f}).")
            actions.append({"type": "set_w_max", "value": 0.20})

    # de-dup by type (keep last)
    last_by_type: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        if a.get("type"):
            last_by_type[a["type"]] = a
    actions = list(last_by_type.values())

    decision = "refine" if needs_refine else "accept"
    rationale = " ".join(rationale_parts) if rationale_parts else "Portfolio matches current preferences and context."

    state["needs_refine"] = needs_refine
    state["refine_actions"] = actions
    state["llm_decision"] = {"decision": decision, "rationale": rationale, "proposed_actions": actions}

    state["debug_notes"].append(f"LLM_Evaluate: decision={decision}, actions={actions}")
    return state


# ----------------------------
# Node: apply refinement
# ----------------------------
def node_apply_refine(state: PortfolioState) -> PortfolioState:
    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for a in state.get("refine_actions", []):
        t = a.get("type")

        if t == "set_objective_key":
            v = a.get("value")
            if v in ("maxsharpe", "minvar"):
                state["objective_key"] = v
                applied.append(a)
            else:
                rejected.append({**a, "reason": "invalid objective_key"})

        elif t == "set_w_max":
            try:
                v = float(a.get("value"))
                v = min(max(v, 0.05), 1.0)
                state["w_max"] = v
                applied.append(a)
            except Exception:
                rejected.append({**a, "reason": "invalid w_max"})

        elif t == "exclude_assets":
            ex = set(a.get("tickers", []))
            if ex:
                state["selected_tickers"] = [x for x in state["selected_tickers"] if x not in ex]
                applied.append(a)
            else:
                rejected.append({**a, "reason": "empty exclude list"})

        else:
            rejected.append({**a, "reason": "unknown action type"})

    state["changes_applied"] = applied
    state["changes_rejected"] = rejected

    state["iteration"] = int(state["iteration"]) + 1
    state["debug_notes"].append(
        f"ApplyRefine: applied={len(applied)}, rejected={len(rejected)}, iteration={state['iteration']}"
    )

    if not state.get("selected_tickers"):
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["debug_notes"].append("ApplyRefine: no tickers left after exclusions -> stop loop.")

    return state


def node_explain(state: PortfolioState) -> PortfolioState:
    if not state.get("optimization_result"):
        state["explanation"] = "No optimization result available (empty universe)."
        state["debug_notes"].append("Explain: skipped (no optimization_result).")
        return state

    obj = "max_sharpe" if state.get("objective_key") == "maxsharpe" else "min_var"
    text = recommendation_agent(
        state["optimization_result"],
        objective=obj,
        current_metrics=state.get("current_metrics"),
        rf=float(state["rf"]),
        preferences=state.get("clarification_answers") or state.get("preferences") or {},
    )
    state["explanation"] = text
    state["debug_notes"].append(f"Explain: generated text (objective_str={obj}).")
    return state


def route_after_llm_evaluate(state: PortfolioState) -> str:
    return "apply_refine" if state.get("needs_refine") else "explain"


# ----------------------------
# Build graph
# ----------------------------
def build_portfolio_graph():
    g = StateGraph(PortfolioState)

    g.add_node("ask_clarifications", node_ask_clarifications)
    g.add_node("perception", node_perception)
    g.add_node("baselines", node_compute_baselines)

    g.add_node("data", node_data)
    g.add_node("optimize", node_optimize)
    g.add_node("extract", node_extract_portfolio)
    g.add_node("risk", node_risk)

    g.add_node("news_fetch", node_news_fetch)
    g.add_node("news_signals", node_news_signals_llm)
    g.add_node("llm_evaluate", node_llm_evaluate)

    g.add_node("apply_refine", node_apply_refine)
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
    g.add_edge("optimize", "extract")
    g.add_edge("extract", "risk")

    g.add_edge("risk", "news_fetch")
    g.add_edge("news_fetch", "news_signals")
    g.add_edge("news_signals", "llm_evaluate")

    g.add_conditional_edges(
        "llm_evaluate",
        route_after_llm_evaluate,
        {"apply_refine": "apply_refine", "explain": "explain"},
    )

    # loop back after refine
    g.add_edge("apply_refine", "baselines")

    g.add_edge("explain", END)

    return g.compile()


def run_graph(
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    preferences: Optional[Dict[str, Any]] = None,
    current_weights: Optional[Dict[str, float]] = None,
    max_iterations: int = 2,
    clarification_answers: Optional[Dict[str, Any]] = None,
    mode: Mode = "refine",
) -> PortfolioState:
    app = build_portfolio_graph()

    init: PortfolioState = {
        "mode": mode,
        "selected_tickers": selected_tickers,
        "rf": rf,
        "w_max": w_max,
        "lambda_l2": 1e-3,
        "preferences": preferences or {},
        "current_weights": current_weights,
        "iteration": 0,
        "max_iterations": max_iterations,
        "debug_notes": [],
        "clarification_answers": clarification_answers,
        "changes_applied": [],
        "changes_rejected": [],
    }
    return app.invoke(init,config={"recursion_limit": 200})
