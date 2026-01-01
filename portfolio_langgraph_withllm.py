 # portfolio_langgraph.py
# ✅ Updated for REAL “LLM-in-the-loop” value:
# - Keep numerical core deterministic (mu/cov → optimizer → metrics)
# - Move ambiguous intent (“extra_notes”) interpretation to the LLM
# - Allow LLM to propose *meaningful* safe changes, including switching objective:
#     * set_objective_key  ✅ (maxsharpe <-> minvar)
#     * set_w_max
#     * set_lambda_l2
#     * exclude_assets  (only if user explicitly indicates dislike-assets)
# - IMPORTANT: When extra_notes is present and pain_points are empty, we treat UI-inferred
#   goal/stability/concentration as “soft” and do NOT deterministically override objective.
#   This is the main change that gives the LLM real leverage without hardcoding rules.

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

# Optional LLM (safe: only used if use_llm=True and llm_client exists)
try:
    from llm_client import LLMClient
except Exception:  # pragma: no cover
    LLMClient = None  # type: ignore


# ----------------------------
# UI label constants (IMPORTANT: must match dashboard strings)
# ----------------------------
PP_TOO_RISKY = "It feels too risky"
PP_TOO_CONSERVATIVE = "It feels too conservative"
PP_TOO_CONCENTRATED = "It’s too concentrated in a few assets"
PP_DISLIKE_ASSETS = "I don’t like some of the assets"
PP_NOT_SURE = "I’m not sure — I just want something safer/smoother"


# ----------------------------
# State enums
# ----------------------------
Mode = Literal["base", "refine"]
ObjectiveKey = Literal["maxsharpe", "minvar"]


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

    # UI-driven LLM toggle
    use_llm: bool

    # interactive loop
    clarification_questions: List[Dict[str, Any]]
    clarification_answers: Optional[Dict[str, Any]]
    needs_user_input: bool

    # derived decisions
    objective_key: ObjectiveKey

    # computed / intermediate
    mu: Optional[pd.Series]
    cov: Optional[pd.DataFrame]

    # current portfolio
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


# ----------------------------
# Utils
# ----------------------------
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


def _merged_prefs(state: PortfolioState) -> Dict[str, Any]:
    # dashboard passes answers via clarification_answers
    return (state.get("clarification_answers") or state.get("preferences") or {}) or {}


def _pref_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


def _wants_diversification(prefs: Dict[str, Any]) -> bool:
    pain_points = _pref_str_list(prefs.get("pain_points"))
    concentration = prefs.get("concentration")
    # diversification desire can come from explicit concentration OR explicit pain point
    return (concentration == "low") or (PP_TOO_CONCENTRATED in pain_points)


def _user_dislikes_assets(prefs: Dict[str, Any]) -> bool:
    pain_points = _pref_str_list(prefs.get("pain_points"))
    return (PP_DISLIKE_ASSETS in pain_points) or bool(prefs.get("excluded_assets"))


def _conflicting_risk_signals(prefs: Dict[str, Any]) -> bool:
    pain_points = _pref_str_list(prefs.get("pain_points"))
    return (PP_TOO_RISKY in pain_points) and (PP_TOO_CONSERVATIVE in pain_points)


def _has_meaningful_extra_notes(prefs: Dict[str, Any]) -> bool:
    txt = str(prefs.get("extra_notes") or "").strip()
    return len(txt) >= 8  # small threshold to ignore accidental whitespace


def _soft_inferred_ui_targets_should_be_ignored(prefs: Dict[str, Any]) -> bool:
    """
    Key idea:
    - Dashboard currently infers (goal/stability/concentration) even when user only wrote extra_notes.
    - If pain_points is empty BUT extra_notes exists, we treat these inferred targets as "soft"
      and do NOT deterministically override objective_key / caps in perception.
    The LLM should decide how to convert that text into actions.
    """
    pain_points = _pref_str_list(prefs.get("pain_points"))
    if pain_points:
        return False
    return _has_meaningful_extra_notes(prefs)


# ----------------------------
# Helper: build questions (rarely used now; dashboard usually provides answers)
# ----------------------------
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


# ----------------------------
# Nodes: ask clarifications
# ----------------------------
def node_ask_clarifications(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    if state.get("mode") == "base":
        state["needs_user_input"] = False
        state["debug_notes"].append("Clarifications(BASE): skipped (base run is non-interactive).")
        return state

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
# Node: perception (ONLY applies explicit, non-ambiguous inputs)
# ----------------------------
def node_perception(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    if state.get("mode") == "base":
        state["debug_notes"].append(
            f"Perception(BASE): objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f}, lambda_l2={float(state['lambda_l2']):.4g}"
        )
        return state

    prefs = _merged_prefs(state)
    satisfaction = str(prefs.get("satisfaction") or "").lower().strip()  # "yes" / "no"
    pain_points = _pref_str_list(prefs.get("pain_points"))
    excluded_assets = prefs.get("excluded_assets") or []

    # apply exclusions deterministically (safe + explicit)
    if excluded_assets:
        excluded = set(map(str, excluded_assets))
        selected = [t for t in list(state.get("selected_tickers", [])) if t not in excluded]
        state["selected_tickers"] = selected
        state["debug_notes"].append(f"Perception: excluded={sorted(excluded)}")

    # ⭐ MAIN CHANGE:
    # If user only wrote extra_notes (pain_points empty), do NOT enforce inferred targets deterministically.
    # Let LLM propose meaningful actions (e.g., switch to minvar) in node_llm_evaluate.
    ignore_soft_targets = _soft_inferred_ui_targets_should_be_ignored(prefs) and (satisfaction == "no")
    if ignore_soft_targets:
        state["debug_notes"].append(
            "Perception: extra_notes present + pain_points empty -> ignoring soft inferred targets (LLM will interpret)."
        )
        state["debug_notes"].append(
            f"Perception: objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f}, lambda_l2={float(state['lambda_l2']):.4g}, n={len(state.get('selected_tickers', []))}"
        )
        return state

    # If explicit structured targets exist (from UI), we can still apply deterministically.
    goal = prefs.get("goal")         # "best_tradeoff" | "lowest_risk" | None
    stability = prefs.get("stability")  # "stable" | "balanced" | "swingy" | None
    concentration = prefs.get("concentration")  # "low" | "high" | None

    # objective from explicit goal/stability
    if goal == "lowest_risk":
        state["objective_key"] = "minvar"
    elif goal == "best_tradeoff":
        state["objective_key"] = "maxsharpe"

    if stability == "stable":
        state["objective_key"] = "minvar"

    # caps from explicit concentration
    if concentration == "low":
        state["w_max"] = min(float(state["w_max"]), 0.20)
    elif concentration == "high":
        state["w_max"] = max(float(state["w_max"]), 0.35)

    # Keep tiny safe nudges ONLY when pain_points explicitly selected
    if satisfaction == "no" and pain_points:
        conflict = _conflicting_risk_signals(prefs)
        if conflict:
            state["lambda_l2"] = max(float(state["lambda_l2"]), 2e-3)
            state["debug_notes"].append("Perception: conflicting pain points -> mild lambda_l2 only.")
        else:
            if PP_TOO_CONCENTRATED in pain_points:
                state["w_max"] = min(float(state["w_max"]), 0.20)
                state["lambda_l2"] = max(float(state["lambda_l2"]), 5e-3)
            if (PP_TOO_RISKY in pain_points) or (PP_NOT_SURE in pain_points):
                state["lambda_l2"] = max(float(state["lambda_l2"]), 5e-3)
            if PP_TOO_CONSERVATIVE in pain_points:
                state["lambda_l2"] = min(float(state["lambda_l2"]), 1e-3)

    state["debug_notes"].append(
        f"Perception: objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f}, "
        f"lambda_l2={float(state['lambda_l2']):.4g}, n={len(state.get('selected_tickers', []))}, "
        f"goal={goal}, stability={stability}, concentration={concentration}"
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
    max_w = max(weights.values()) if weights else 0.0
    state["debug_notes"].append(f"Extract: max_weight={max_w:.4f} vs w_max={float(state['w_max']):.2f}")
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
# Node: news -> signals (placeholder)
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
    state["debug_notes"].append("NewsSignals(placeholder): produced signals.")
    return state


# ----------------------------
# Node: LLM evaluation / refinement controller
# ----------------------------
def node_llm_evaluate(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    if state.get("mode") == "base":
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "Base run: refinement disabled. Portfolio generated for comparison.",
            "proposed_actions": [],
        }
        state["debug_notes"].append("LLM_Evaluate(BASE): accept (no refine).")
        return state

    prefs = _merged_prefs(state)
    satisfaction = str(prefs.get("satisfaction") or "").lower().strip()  # "yes" / "no"
    pain_points = _pref_str_list(prefs.get("pain_points"))

    # ✅ If user is happy -> ACCEPT, always
    if satisfaction == "yes":
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "User indicated the portfolio looks good (satisfaction=yes).",
            "proposed_actions": [],
        }
        state["debug_notes"].append("LLM_Evaluate: satisfaction=yes -> accept.")
        return state

    # If max iters reached -> accept
    if int(state["iteration"]) >= int(state["max_iterations"]):
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {"decision": "accept", "rationale": "Max iterations reached.", "proposed_actions": []}
        state["debug_notes"].append("LLM_Evaluate: max iterations reached -> accept.")
        return state

    if not state.get("optimized_weights"):
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "No optimized weights; stopping.",
            "proposed_actions": [],
        }
        state["debug_notes"].append("LLM_Evaluate: no weights -> accept.")
        return state

    # ✅ Only refine if user explicitly said "no"
    if satisfaction != "no":
        state["needs_refine"] = False
        state["refine_actions"] = []
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "No explicit dissatisfaction signal. Skipping refinement.",
            "proposed_actions": [],
        }
        state["debug_notes"].append("LLM_Evaluate: satisfaction not provided -> accept.")
        return state

    wants_div = _wants_diversification(prefs)
    allow_exclude = _user_dislikes_assets(prefs)
    conflict = _conflicting_risk_signals(prefs)

    # ---- LLM path ----
    use_llm = bool(state.get("use_llm", False))
    if use_llm and LLMClient is not None:
        try:
            client = LLMClient()

            # NOTE:
            # Your llm_client.py currently validates allowed action types.
            # To support real value, you MUST allow "set_objective_key" there too.
            # We already enforce guards here + in apply_refine.
            llm_payload = client.decide_refine_actions(
                mode=str(state.get("mode")),
                iteration=int(state.get("iteration", 0)),
                max_iterations=int(state.get("max_iterations", 0)),
                objective_key=str(state.get("objective_key")),
                rf=float(state.get("rf")),
                w_max=float(state.get("w_max")),
                lambda_l2=float(state.get("lambda_l2")),
                selected_tickers=list(state.get("selected_tickers", [])),
                optimized_metrics=state.get("optimized_metrics") or {},
                optimized_weights=state.get("optimized_weights") or {},
                baseline_metrics=state.get("baseline_metrics"),
                current_metrics=state.get("current_metrics"),
                preferences=prefs,
                news_signals=state.get("news_signals"),
            )

            decision = str(llm_payload.get("decision", "accept")).lower().strip()
            actions = llm_payload.get("proposed_actions", []) or []
            rationale = str(llm_payload.get("rationale", "")).strip()

            # Guard: exclude_assets only if user explicitly signaled it
            if not allow_exclude:
                actions = [a for a in actions if a.get("type") != "exclude_assets"]

            # Guard: if wants diversification, do not increase w_max
            filtered: List[Dict[str, Any]] = []
            for a in actions:
                if a.get("type") == "set_w_max" and wants_div:
                    try:
                        if float(a.get("value")) > float(state["w_max"]):
                            continue
                    except Exception:
                        continue
                filtered.append(a)
            actions = filtered

            # Guard: conflict case -> only mild changes (prefer lambda_l2 + objective switch at most)
            if conflict:
                keep = {"set_lambda_l2", "set_objective_key"}
                actions = [a for a in actions if a.get("type") in keep]

            needs_refine = (decision == "refine") and (len(actions) > 0)

            state["needs_refine"] = needs_refine
            state["refine_actions"] = actions if needs_refine else []
            state["llm_decision"] = {
                "decision": "refine" if needs_refine else "accept",
                "rationale": rationale or ("LLM proposed changes." if needs_refine else "LLM: accept."),
                "proposed_actions": state["refine_actions"],
            }
            state["debug_notes"].append(f"LLM_Evaluate(LLM): decision={decision}, actions={actions}")
            return state

        except Exception as e:
            state["debug_notes"].append(f"LLM_Evaluate(LLM): failed -> fallback to rules: {e}")

    # ---- Minimal fallback rules (only for safety) ----
    actions: List[Dict[str, Any]] = []
    rationale_parts: List[str] = []

    # If news regime high: reduce concentration + diversify
    news_signals = state.get("news_signals") or {}
    vol_regime = ((news_signals.get("global") or {}).get("vol_regime")) or "normal"
    if vol_regime == "high":
        actions.append({"type": "set_w_max", "value": _clamp(float(state["w_max"]) - 0.05, 0.10, 1.0)})
        actions.append({"type": "set_lambda_l2", "value": max(float(state["lambda_l2"]), 5e-3)})
        rationale_parts.append("Elevated volatility regime → reduce concentration + encourage diversification.")

    # If user provided explicit pain_points, honor them minimally
    if conflict:
        actions.append({"type": "set_lambda_l2", "value": max(float(state["lambda_l2"]), 2e-3)})
        rationale_parts.append("Conflicting feedback → mild diversification pressure.")
    else:
        if pain_points:
            if (PP_TOO_RISKY in pain_points) or (PP_NOT_SURE in pain_points):
                actions.append({"type": "set_objective_key", "value": "minvar"})
                actions.append({"type": "set_lambda_l2", "value": max(float(state["lambda_l2"]), 5e-3)})
                rationale_parts.append("Risk/smoother preference → switch to min-variance + diversify.")
            if PP_TOO_CONCENTRATED in pain_points:
                actions.append({"type": "set_w_max", "value": min(float(state["w_max"]), 0.20)})
                actions.append({"type": "set_lambda_l2", "value": max(float(state["lambda_l2"]), 5e-3)})
                rationale_parts.append("Concentration complaint → lower cap + add L2.")
        else:
            # If no pain_points but dissatisfaction exists, do one safe move:
            # keep fallback conservative: small lambda_l2 increase
            actions.append({"type": "set_lambda_l2", "value": max(float(state["lambda_l2"]), 2e-3)})
            rationale_parts.append("No structured feedback; applying small safe diversification pressure.")

    # de-dup by type (keep last)
    last_by_type: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        if a.get("type"):
            last_by_type[a["type"]] = a
    actions = list(last_by_type.values())

    # enforce exclude_assets only if user wants it
    if not allow_exclude:
        actions = [a for a in actions if a.get("type") != "exclude_assets"]

    # enforce no w_max increase when wants_div
    if wants_div:
        filtered2: List[Dict[str, Any]] = []
        for a in actions:
            if a.get("type") == "set_w_max":
                try:
                    if float(a.get("value")) > float(state["w_max"]):
                        continue
                except Exception:
                    continue
            filtered2.append(a)
        actions = filtered2

    needs_refine = len(actions) > 0
    state["needs_refine"] = needs_refine
    state["refine_actions"] = actions if needs_refine else []
    state["llm_decision"] = {
        "decision": "refine" if needs_refine else "accept",
        "rationale": " ".join(rationale_parts) if rationale_parts else "No actionable adjustments inferred.",
        "proposed_actions": state["refine_actions"],
    }
    state["debug_notes"].append(f"LLM_Evaluate(FallbackRules): actions={state['refine_actions']}")
    return state


# ----------------------------
# Node: apply refinement
# ----------------------------
def node_apply_refine(state: PortfolioState) -> PortfolioState:
    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    prefs = _merged_prefs(state)
    wants_div = _wants_diversification(prefs)
    allow_exclude = _user_dislikes_assets(prefs)

    for a in state.get("refine_actions", []):
        t = a.get("type")

        if t == "set_objective_key":
            v = str(a.get("value") or "").strip().lower()
            if v not in ("maxsharpe", "minvar"):
                rejected.append({**a, "reason": "invalid objective_key"})
                continue
            state["objective_key"] = "minvar" if v == "minvar" else "maxsharpe"
            applied.append(a)
            continue

        if t == "set_w_max":
            try:
                v = float(a.get("value"))
                v = _clamp(v, 0.05, 1.0)
                if wants_div and v > float(state["w_max"]):
                    rejected.append({**a, "reason": "cannot increase w_max when user wants diversification"})
                    continue
                state["w_max"] = v
                applied.append(a)
            except Exception:
                rejected.append({**a, "reason": "invalid w_max"})
            continue

        if t == "set_lambda_l2":
            try:
                v = float(a.get("value"))
                v = _clamp(v, 0.0, 1.0)
                state["lambda_l2"] = v
                applied.append(a)
            except Exception:
                rejected.append({**a, "reason": "invalid lambda_l2"})
            continue

        if t == "exclude_assets":
            if not allow_exclude:
                rejected.append({**a, "reason": "exclude_assets only allowed when user dislikes assets"})
                continue
            ex = set(map(str, a.get("tickers", []) or []))
            if not ex:
                rejected.append({**a, "reason": "empty exclude list"})
                continue
            state["selected_tickers"] = [x for x in state["selected_tickers"] if x not in ex]
            applied.append(a)
            continue

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
        preferences=_merged_prefs(state),
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
    use_llm: bool = False,
) -> PortfolioState:
    app = build_portfolio_graph()

    init: PortfolioState = {
        "mode": mode,
        "selected_tickers": selected_tickers,
        "rf": rf,
        "w_max": w_max,
        "lambda_l2": 1e-3,
        "preferences": preferences or {},
        "use_llm": bool(use_llm),
        "current_weights": current_weights,
        "iteration": 0,
        "max_iterations": int(max_iterations),
        "debug_notes": [],
        "clarification_answers": clarification_answers,
        "changes_applied": [],
        "changes_rejected": [],
    }

    return app.invoke(init, config={"recursion_limit": 200})
 