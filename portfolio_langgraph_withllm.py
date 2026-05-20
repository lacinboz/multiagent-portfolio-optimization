
# portfolio_langgraph_withllm.py
from __future__ import annotations
from collections import defaultdict, Counter
import hashlib

from typing import Optional, Set
import re
from typing import Set, Optional

from typing import TypedDict, List, Dict, Any, Optional, Literal
import re
from collections import Counter
import pandas as pd
from langgraph.graph import StateGraph, END
from news_return_predictor import load_prediction_model
from probabilistic_news_integration import (
    build_adjusted_inputs_from_existing_news_state,
    evaluate_news_adjustment_effect,  
)
from agents_langgraph import (
    data_agent_get_mu_cov,
    optimization_agent_from_mu_cov,
    prediction_constrained_optimization_agent,
    risk_agent,
    recommendation_agent,
    apply_news_actions_to_params,
)

# ✅ News fetch (real)
try:
    from agents_langgraph import news_agent_fetch_for_tickers  # type: ignore
except Exception:  # pragma: no cover
    news_agent_fetch_for_tickers = None  # type: ignore

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
Stage = Literal["main", "news_actions", "news_overview"]
ObjectiveKey = Literal["maxsharpe", "minvar"]


class PortfolioState(TypedDict, total=False):
    mode: Mode
    stage: Stage  # ✅ NEW: stage controls which sub-flow runs

    selected_tickers: List[str]
    rf: float
    w_max: float
    lambda_l2: float
    preferences: Dict[str, Any]
    use_llm: bool

    # ✅ news checkbox / toggle
    use_news: bool

    clarification_questions: List[Dict[str, Any]]
    clarification_answers: Optional[Dict[str, Any]]
    needs_user_input: bool

    objective_key: ObjectiveKey
    chosen_candidate: Optional[ObjectiveKey]
    llm_decision: Optional[Dict[str, Any]]
    news_evidence_snapshot_text: Optional[str]
    news_evidence_snapshot_ok: Optional[bool]
    news_evidence_snapshot_issues: List[str]
    news_snapshot_text_raw: Optional[str]   # ✅ UI-friendly, evidence tag'siz


    mu: Optional[pd.Series]
    cov: Optional[pd.DataFrame]
    optimization_result: Dict[str, Any]

    current_weights: Optional[Dict[str, float]]
    baseline_metrics: Optional[Dict[str, Any]]
    current_metrics: Optional[Dict[str, Any]]

    candidates: Dict[str, Dict[str, Any]]

    optimized_weights: Dict[str, float]
    optimized_metrics: Dict[str, Any]

    # ✅ news
    news_raw: Optional[List[Dict[str, Any]]]
    news_signals: Optional[Dict[str, Any]]  # keep legacy name: will store risk_json
    news_snapshot_text: Optional[str]
    news_risk_json: Optional[Dict[str, Any]]
    news_items_llm: Optional[List[Dict[str, Any]]]
    evidence_map: Optional[Dict[str, Dict[str, Any]]]

    # ✅ NEW: News action stage outputs
    news_actions: Optional[List[Dict[str, Any]]]
    news_actions_verifier: Optional[Dict[str, Any]]
    news_adjustment_evaluation: Optional[Dict[str, Any]]
    prob_prediction_evaluation: Optional[Dict[str, Any]]

    prediction_probs: Optional[Dict[str, float]]
    prediction_adjusted_caps: Optional[Dict[str, float]]
    prediction_model_metrics: Optional[Dict[str, Any]]
    prediction_model_used: Optional[bool]

    constraint_debug: Optional[Dict[str, Any]]
    baseline_candidate_weights: Optional[Dict[str, float]]



    debug_notes: List[str]
    explanation: str

    # ✅ Insight Generator outputs
    insight: Optional[Dict[str, Any]]
    insight_ok: Optional[bool]
    insight_issues: List[str]
    insight_raw_text: Optional[str]
    insight_parse_mode: Optional[str]

    prob_news_signals: Optional[Dict[str, Any]]
    prob_adjusted_mu: Optional[pd.Series]
    prob_adjusted_cov: Optional[pd.DataFrame]
    prob_alpha: Optional[float]
    prob_beta: Optional[float]

    prob_news_trace: Optional[Dict[str, Any]]
    historical_prediction_evaluation: Optional[Dict[str, Any]]

    # ✅ carry the user's previous portfolio (Run Base output) into refine run
    base_portfolio_weights: Optional[Dict[str, float]]
    base_portfolio_metrics: Optional[Dict[str, Any]]
    base_portfolio_objective: Optional[str]


# =========================================================
# Defaults / prefs
# =========================================================
def _init_defaults(state: PortfolioState) -> PortfolioState:
    state.setdefault("mode", "refine")
    state.setdefault("stage", "main")  # ✅ NEW

    state.setdefault("rf", 0.02)
    state.setdefault("w_max", 0.30)
    state.setdefault("lambda_l2", 1e-3)
    state.setdefault("objective_key", "maxsharpe")
    state.setdefault("preferences", {})
    state.setdefault("use_llm", False)

    # ✅ default: news OFF (checkbox)
    state.setdefault("use_news", False)

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
    state.setdefault("news_snapshot_text", None)
    state.setdefault("news_snapshot_text_raw", None)
    state.setdefault("news_risk_json", None)
    state.setdefault("news_items_llm", None)
    state.setdefault("evidence_map", None)
    state.setdefault("news_evidence_snapshot_text", None)
    state.setdefault("news_evidence_snapshot_ok", None)
    state.setdefault("news_evidence_snapshot_issues", [])

    state.setdefault("news_adjustment_evaluation", None)
    state.setdefault("prob_prediction_evaluation", None)

    state.setdefault("prob_news_trace", None)
    state.setdefault("historical_prediction_evaluation", None)

    
    


    # ✅ NEW: news action outputs
    state.setdefault("news_actions", None)
    state.setdefault("news_actions_verifier", None)

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

    state.setdefault("prob_news_signals", None)
    state.setdefault("prob_adjusted_mu", None)
    state.setdefault("prob_adjusted_cov", None)
    state.setdefault("prob_alpha", 0.08)
    state.setdefault("prob_beta", 0.35)

    state.setdefault("historical_prediction_evaluation", None)

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
            "id": "use_news",
            "type": "select",
            "label": "Include news snapshot & risk check? (optional)",
            "options": ["no", "yes"],
            "option_labels": ["No (faster)", "Yes (fetch latest news)"],
            "default": "no",
        },
        {
            "id": "extra_notes",
            "type": "text",
            "label": "Extra notes (optional)",
            "default": "",
        },
    ]

# =========================================================
# ✅ NEW: News selection + deterministic cleaning helpers
# =========================================================
from datetime import datetime
from collections import defaultdict

_ALLOWED_RISK_FLAGS = {
    "none",
    "event_risk",
    "earnings_uncertainty",
    "regulatory",
    "litigation",
    "product_issue",
    "macro",
}

_RISK_FLAG_ALIASES = {
    # common sloppy outputs
    "low": "event_risk",
    "medium": "event_risk",
    "high": "event_risk",
    "med": "event_risk",
    "moderate": "event_risk",
    "": "none",
    "n/a": "none",
    "na": "none",
    "unknown": "none",
}

_ALLOWED_ACTION_TYPES = {
    "exclude_ticker",
    "set_w_max",
    "shift_objective",
    "reduce_exposure",
    "hedge",
}


from datetime import datetime, timezone
def _epoch_to_ymd(x: Any) -> Optional[str]:
    try:
        if x is None:
            return None
        # Finnhub datetime genelde epoch seconds
        if isinstance(x, (int, float)):
            if float(x) <= 0:
                return None
            return datetime.fromtimestamp(float(x), tz=timezone.utc).strftime("%Y-%m-%d")
        # eğer string iso gelirse
        s = str(x).strip()
        if not s:
            return None
        # "2026-01-12T..." gibi
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z","")).date().isoformat()
        # "2026-01-12" gibi
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return s
        return None
    except Exception:
        return None

def _parse_dt(x: Any) -> Optional[datetime]:
    if not x:
        return None
    if isinstance(x, datetime):
        return x

    # ✅ NEW: epoch seconds
    if isinstance(x, (int, float)):
        try:
            if x > 0:
                return datetime.fromtimestamp(float(x), tz=timezone.utc)
        except Exception:
            pass

    s = str(x).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1]
        return datetime.fromisoformat(s)
    except Exception:
        return None

from collections import defaultdict
from typing import Any, Dict, List

def _select_news_items_for_llm(
    news_raw: List[Dict[str, Any]],
    tickers: List[str],
    *,
    per_ticker: int = 2,
    max_total: int = 140,
) -> List[Dict[str, Any]]:
    """
    Prevent one ticker from dominating the prompt.
    Picks up to `per_ticker` items per ticker (newest first if datetime is parseable).
    """
    if not news_raw or not tickers:
        return []

    allowed = {str(t).upper().strip() for t in tickers if str(t).strip()}
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for it in news_raw:
        t = str(it.get("ticker") or "").upper().strip()
        if t in allowed:
            buckets[t].append(it)

    # sort each bucket by datetime desc:
    # 1) dt parse edilebilenler önce
    # 2) sonra dt'ye göre (en yeni önce)
    for t, items in buckets.items():
        decorated = []
        for it in items:
            dt = _parse_dt(it.get("datetime"))
            decorated.append((dt is not None, dt, it))  # (has_dt, dt, item)

        decorated.sort(key=lambda x: (x[0], x[1]), reverse=True)
        buckets[t] = [it for _, _, it in decorated]

    selected: List[Dict[str, Any]] = []

    tickers_u_raw = [str(t).upper().strip() for t in tickers if str(t).strip()]
    tickers_u_raw = [t for t in tickers_u_raw if t in buckets and buckets[t]]

    # ✅ unique (order-preserving)
    seen = set()
    tickers_u = []
    for t in tickers_u_raw:
        if t in seen:
            continue
        seen.add(t)
        tickers_u.append(t)


    if not tickers_u:
        return []

    # Each ticker contributes up to per_ticker (round-robin)
    for _ in range(per_ticker):
        for t in tickers_u:
            if buckets[t]:
                selected.append(buckets[t].pop(0))
            if len(selected) >= max_total:
                return selected

    # Fill remaining space with leftovers globally (newest first)
    if len(selected) < max_total:
        leftovers: List[Dict[str, Any]] = []
        for t in tickers_u:
            leftovers.extend(buckets[t])

        decorated = []
        for it in leftovers:
            dt = _parse_dt(it.get("datetime"))
            decorated.append((dt is not None, dt, it))

        decorated.sort(key=lambda x: (x[0], x[1]), reverse=True)

        space = max_total - len(selected)
        selected.extend([it for _, _, it in decorated[:space]])

    return selected


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
        if v != v:
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return float(v)
    except Exception:
        return 0.0


def _clean_risk_json_fill_universe(risk_json: Dict[str, Any], tickers: List[str]) -> Dict[str, Any]:
    """
    Deterministic post-cleaner:
    - normalizes risk_flag enums
    - ensures {summary, by_ticker, global{risk_flags, vol_regime}} exists
    - fills every ticker with {none, 0.0} if missing
    """
    tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
    allowed = set(tickers_u)

    out: Dict[str, Any] = {
        "summary": "",
        "by_ticker": {},
        "global": {"risk_flags": [], "vol_regime": "normal"},
    }

    if not isinstance(risk_json, dict):
        risk_json = {}

    if isinstance(risk_json.get("summary"), str):
        out["summary"] = risk_json["summary"].strip()

    # global
    glob = risk_json.get("global")
    if isinstance(glob, dict):
        vr = str(glob.get("vol_regime") or "normal").strip().lower()
        if vr not in ("normal", "high"):
            vr = "normal"
        out["global"]["vol_regime"] = vr

        rf = glob.get("risk_flags")
        if isinstance(rf, list):
            kept = []
            for item in rf:
                if not isinstance(item, dict):
                    continue
                t = str(item.get("ticker") or "").upper().strip()

                flag0 = str(item.get("flag") or "").strip().lower()
                flag0 = _RISK_FLAG_ALIASES.get(flag0, flag0)
                if flag0 not in _ALLOWED_RISK_FLAGS:
                    flag0 = "none"

                if t in allowed and flag0 and flag0 != "none":
                    kept.append({"ticker": t, "flag": flag0})
            out["global"]["risk_flags"] = kept


    # by_ticker
    bt = risk_json.get("by_ticker")
    if isinstance(bt, dict):
        for k, v in bt.items():
            t = str(k).upper().strip()
            if t not in allowed:
                continue
            if not isinstance(v, dict):
                continue
            rf0 = str(v.get("risk_flag") or "none").strip().lower()
            rf0 = _RISK_FLAG_ALIASES.get(rf0, rf0)
            if rf0 not in _ALLOWED_RISK_FLAGS:
                rf0 = "none"
            conf = _clamp01(v.get("confidence"))
            out["by_ticker"][t] = {"risk_flag": rf0, "confidence": conf}

    # fill missing tickers
    for t in tickers_u:
        if t not in out["by_ticker"]:
            out["by_ticker"][t] = {"risk_flag": "none", "confidence": 0.0}
  
    rebuilt = []
    for t, v in out["by_ticker"].items():
        flag = str(v.get("risk_flag") or "none")
        if flag and flag != "none":
            rebuilt.append({"ticker": t, "flag": flag})
    out["global"]["risk_flags"] = rebuilt


    return out



def _extract_evidence_ids_from_snapshot(snapshot_text: str) -> Set[str]:
    """
    Extract IDs from canonical snapshot bullets:
      - ([NVDA_03] 2026-01-18 | Reuters) ...
    Returns a set of evidence_ids.
    """
    text = snapshot_text or ""
    ids = re.findall(r"\(\[([A-Z0-9_]+)\]", text)
    return set(i.strip() for i in ids if i and i.strip())

def _clean_news_actions(
    actions: List[Dict[str, Any]],
    tickers: List[str],
    *,
    allowed_eids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extra deterministic guard (even if llm_client already cleaned):
    - drops unknown action types
    - enforces ticker for exclude_ticker
    - enforces ticker for reduce_exposure (THIS fixes ticker-less ambiguity)
    - validates tighten_w_max bounds
    - validates shift_objective
    """
    allowed = {str(t).upper().strip() for t in (tickers or []) if str(t).strip()}
    allowed_eids = allowed_eids or set()
    cleaned: List[Dict[str, Any]] = []

    for a in actions or []:
        if not isinstance(a, dict):
            continue
        t = str(a.get("type") or "").strip()
        if t not in _ALLOWED_ACTION_TYPES:
            continue

        base = {"type": t, "reason": str(a.get("reason") or "").strip()}
        # ✅ NEW: evidence_ids (Yol A)
        eids = a.get("evidence_ids")
        if isinstance(eids, list):
            eids_out = []
            for x in eids:
                s = str(x).strip()
                if not s:
                    continue
                if allowed_eids and s not in allowed_eids:
                    continue
                eids_out.append(s)
            base["evidence_ids"] = eids_out[:2]

            if not base["evidence_ids"]:
                continue
        else:
            base["evidence_ids"] = []
        if not base["reason"]:
            base["reason"] = "Derived from recent news risk signals."
              
        ev = a.get("evidence")
        if isinstance(ev, list):
            ev_out = []
            for x in ev[:3]:
                if not isinstance(x, dict):
                    continue
                headline = str(x.get("headline") or "").strip()
                if not headline:
                    continue
                item = {"headline": headline}

                date = x.get("date")
                source = x.get("source")
                url = x.get("url")

                if isinstance(date, str) and date.strip():
                    item["date"] = date.strip()
                if isinstance(source, str) and source.strip():
                    item["source"] = source.strip()
                if isinstance(url, str) and url.strip():
                    item["url"] = url.strip()

                ev_out.append(item)

            if ev_out:
                base["evidence"] = ev_out


        if t == "exclude_ticker":
            ticker = str(a.get("ticker") or "").upper().strip()
            if ticker in allowed:
                base["ticker"] = ticker
                cleaned.append(base)
            continue

        if t == "set_w_max":
            try:
                v = float(a.get("value"))
            except Exception:
                continue
            if 0.05 <= v <= 0.50:
                base["value"] = v
                cleaned.append(base)
            continue

        if t == "shift_objective":
            to = str(a.get("to") or "").lower().strip()
            if to in ("minvar", "maxsharpe"):
                base["to"] = to
                cleaned.append(base)
            continue

        if t == "reduce_exposure":
            # ✅ enforce ticker, otherwise it's too vague
            ticker = str(a.get("ticker") or "").upper().strip()
            if ticker not in allowed:
                continue
            intensity = str(a.get("intensity") or "medium").lower().strip()
            if intensity not in ("low", "medium", "high"):
                intensity = "medium"
            base["ticker"] = ticker
            base["intensity"] = intensity
            cleaned.append(base)
            continue

        if t == "hedge":
            base["hedge_hint"] = str(a.get("hedge_hint") or "").strip()
            if not base["hedge_hint"]:
                continue
            cleaned.append(base)
            continue

    # de-dup simple (type+ticker+to/value/intensity)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for a in cleaned:
        key = (
            a.get("type"),
            a.get("ticker"),
            a.get("to"),
            a.get("value"),
            a.get("intensity"),
            a.get("hedge_hint"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq

def _debug_explain_action_drop_reasons(
    raw_actions: List[Dict[str, Any]],
    cleaned_actions: List[Dict[str, Any]],
    tickers: List[str],
    *,
    allowed_eids: Optional[Set[str]] = None,
    max_logs: int = 60,
) -> List[str]:
    """
    Returns debug lines explaining why raw actions were dropped by _clean_news_actions.
    Keeps logs bounded to avoid blowing up UI / recursion.
    """
    allowed_eids = allowed_eids or set()
    allowed = {str(t).upper().strip() for t in (tickers or []) if str(t).strip()}
    allowed_types = set(_ALLOWED_ACTION_TYPES)

    # index cleaned by a stable-ish key so we can detect which raw survived
    def _key(a: Dict[str, Any]) -> tuple:
        return (
            str(a.get("type") or "").strip(),
            str(a.get("ticker") or "").upper().strip() or None,
            str(a.get("to") or "").strip() or None,
            a.get("value"),
            str(a.get("intensity") or "").strip() or None,
            str(a.get("hedge_hint") or "").strip() or None,
        )

    cleaned_keys = {_key(a) for a in (cleaned_actions or []) if isinstance(a, dict)}

    logs: List[str] = []
    dropped = 0

    for i, a in enumerate(raw_actions or []):
        if len(logs) >= max_logs:
            break

        if not isinstance(a, dict):
            dropped += 1
            logs.append(f"NewsActionsDrop[{i}]: not a dict -> dropped: {repr(a)[:200]}")
            continue

        t = str(a.get("type") or "").strip()
        ticker = str(a.get("ticker") or "").upper().strip()
        reason = ""

        # If it survived, don't log as dropped.
        # We approximate survival by recomputing the post-clean key.
        approx_clean = {"type": t}
        if t == "exclude_ticker":
            approx_clean["ticker"] = ticker
        elif t == "set_w_max":
            approx_clean["value"] = a.get("value")
        elif t == "shift_objective":
            approx_clean["to"] = str(a.get("to") or "").lower().strip()
        elif t == "reduce_exposure":
            approx_clean["ticker"] = ticker
            approx_clean["intensity"] = str(a.get("intensity") or "medium").lower().strip()
        elif t == "hedge":
            approx_clean["hedge_hint"] = str(a.get("hedge_hint") or "").strip()

        if _key(approx_clean) in cleaned_keys:
            continue

        # Determine drop reason (mirror _clean_news_actions rules)
        if not t:
            reason = "missing type"
        elif t not in allowed_types:
            reason = f"type not allowed ({t})"
        elif t in ("exclude_ticker", "reduce_exposure") and ticker not in allowed:
            reason = f"ticker missing/invalid ({ticker or '∅'})"
        elif t == "set_w_max":
            try:
                v = float(a.get("value"))
                if not (0.05 <= v <= 0.50):
                    reason = f"value out of bounds for set_w_max ({v})"
                else:
                    reason = "unknown (possibly duplicate dropped)"
            except Exception:
                reason = "value not parseable for set_w_max"
        elif t == "shift_objective":
            to = str(a.get("to") or "").lower().strip()
            if to not in ("minvar", "maxsharpe"):
                reason = f"invalid 'to' for shift_objective ({to or '∅'})"
            else:
                reason = "unknown (possibly duplicate dropped)"
        elif t == "reduce_exposure":
            intensity = str(a.get("intensity") or "medium").lower().strip()
            if intensity not in ("low", "medium", "high"):
                reason = f"invalid intensity ({intensity}) -> would be normalized, but dropped/duplicate"
            else:
                reason = "unknown (possibly duplicate dropped)"
        elif t == "hedge":
            hint = str(a.get("hedge_hint") or "").strip()
            if not hint:
                reason = "missing hedge_hint"
            else:
                reason = "unknown (possibly duplicate dropped)"
        else:
            reason = "unknown (not matched)"

        dropped += 1
        eids = a.get("evidence_ids")
        if allowed_eids and isinstance(eids, list) and eids:
            bad = [str(x).strip() for x in eids if str(x).strip() and str(x).strip() not in allowed_eids]
            if bad:
                reason = f"invalid evidence_ids (not in evidence_map): {bad[:3]}"
        logs.append(f"NewsActionsDrop[{i}]: {reason} | raw={a}")


    # summary line (helps a lot)
    if raw_actions is not None:
        logs.insert(
            0,
            f"NewsActionsDropSummary: raw={len(raw_actions)} cleaned={len(cleaned_actions)} dropped={dropped} (logs_cap={max_logs})",
        )

    return logs


# =========================================================
# Metrics helpers

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

    # ✅ For news_actions stage, we do NOT stop for user input.
    if state.get("stage") in ("news_actions", "news_overview"):
        state["needs_user_input"] = False
        state["debug_notes"].append(f"Clarifications: skipped (stage={state.get('stage')}).")
        return state

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

    # ✅ stage=news_actions should force news on (unless base)
    # ✅ stage=news_actions OR news_overview should force news on (unless base)
    if state.get("stage") in ("news_actions", "news_overview") and state.get("mode") != "base":
        state["use_news"] = True
        state["use_llm"] = True
        state["debug_notes"].append(f"Perception: stage={state.get('stage')} -> forcing use_llm=True.")
        state["debug_notes"].append(f"Perception: stage={state.get('stage')} -> forcing use_news=True.")

    if state.get("mode") == "base":
        # ✅ hard rule: base never fetches news
        state["use_news"] = False
        state["debug_notes"].append(
            f"Perception(BASE): objective_key={state['objective_key']}, w_max={float(state['w_max']):.2f}, "
            f"lambda_l2={float(state['lambda_l2']):.4g}, use_news={state['use_news']}"
        )
        return state

    # refine:
    satisfaction = str(prefs.get("satisfaction") or "").lower().strip()

    excluded_assets = prefs.get("excluded_assets") or []
    selected_news_actions = prefs.get("selected_news_actions") or []
    if excluded_assets:
        excluded = set(map(str, excluded_assets))
        state["selected_tickers"] = [t for t in list(state.get("selected_tickers", [])) if t not in excluded]
        state["debug_notes"].append(f"Perception: excluded={sorted(excluded)}")



    if isinstance(selected_news_actions, list) and selected_news_actions:
        try:
            applied = apply_news_actions_to_params(
                selected_tickers=list(state.get("selected_tickers", []) or []),
                w_max=float(state.get("w_max", 0.30)),
                lambda_l2=float(state.get("lambda_l2", 1e-3)),
                objective_key=str(state.get("objective_key", "maxsharpe")),
                actions=selected_news_actions,
            )

            state["selected_tickers"] = applied.get("selected_tickers", state.get("selected_tickers", []))
            state["w_max"] = float(applied.get("w_max", state.get("w_max", 0.30)))
            state["lambda_l2"] = float(applied.get("lambda_l2", state.get("lambda_l2", 1e-3)))
            state["objective_key"] = str(applied.get("objective_key", state.get("objective_key", "maxsharpe")))

            state["debug_notes"].append(
                f"Perception: applied selected_news_actions n={len(selected_news_actions)} "
                f"-> selected_tickers={state['selected_tickers']} "
                f"w_max={state['w_max']:.4f} "
                f"lambda_l2={state['lambda_l2']:.6f} "
                f"objective={state['objective_key']}"
            )

        except Exception as e:
            state["debug_notes"].append(f"Perception: failed to apply selected_news_actions -> {e}")

    if state.get("stage") not in ("news_actions", "news_overview"):
        if "use_news" in prefs:
            use_news_ui = str(prefs.get("use_news") or "no").lower().strip()
            state["use_news"] = (use_news_ui == "yes")

    extra_notes = str(prefs.get("extra_notes") or "").strip()
    pain_points = prefs.get("pain_points") or []
    pain_points_n = len(pain_points) if isinstance(pain_points, list) else 1

    state["debug_notes"].append(
        f"Perception(REFINE): stage={state.get('stage')}, satisfaction={satisfaction or '∅'}, pain_points={pain_points_n}, "
        f"extra_notes={'yes' if extra_notes else 'no'}, use_news={state['use_news']}, "
        f"n={len(state.get('selected_tickers', []))}"
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

    tickers = list(state.get("selected_tickers", []) or [])
    n = len(tickers)
    requested_w_max = float(state["w_max"])
    effective_w_max = max(requested_w_max, 1.0 / n) if n > 0 else requested_w_max

    state["debug_notes"].append(
        f"[OPT DEBUG BEFORE] tickers={tickers}, n={n}, "
        f"requested_w_max={requested_w_max:.4f}, "
        f"effective_w_max_needed={effective_w_max:.4f}, "
        f"sum_capacity={n * requested_w_max:.4f}"
    )

    res = optimization_agent_from_mu_cov(
        mu=state["mu"],
        cov=state["cov"],
        rf=float(state["rf"]),
        w_max=float(state["w_max"]),
        lambda_l2=float(state["lambda_l2"]),
    )

    state["optimization_result"] = res

    frontier = res.get("frontier") if isinstance(res, dict) else None
    state["debug_notes"].append(
        f"[OPT DEBUG AFTER] result_keys={list(res.keys()) if isinstance(res, dict) else None}, "
        f"frontier_type={type(frontier).__name__}, "
        f"frontier_len={len(frontier) if isinstance(frontier, list) else 'not_list'}, "
        f"has_maxsharpe={'maxsharpe' in res if isinstance(res, dict) else False}, "
        f"has_minvar={'minvar' in res if isinstance(res, dict) else False}"
    )

    for k in ["maxsharpe", "minvar"]:
        if isinstance(res, dict) and k in res:
            w = (res.get(k) or {}).get("weights", {})
            state["debug_notes"].append(
                f"[OPT DEBUG {k}] weights={w}, "
                f"sum_weights={sum(float(v) for v in w.values()) if isinstance(w, dict) else 'NA'}"
            )

    return state

def node_optimize_prob_news(state: PortfolioState) -> PortfolioState:
    if state.get("mu") is None or state.get("cov") is None:
        state["optimization_result"] = {}
        state["debug_notes"].append("OptimizationProbNews: skipped (missing mu/cov).")
        return state

    mu_to_use = state["mu"]
    cov_to_use = state["cov"]

    if (
        state.get("mode") != "base"
        and bool(state.get("use_news", False))
        and state.get("news_raw")
    ):
        try:
            news_out = build_adjusted_inputs_from_existing_news_state(
                mu=state["mu"],
                cov=state["cov"],
                tickers=state.get("selected_tickers", []),
                news_raw=state.get("news_raw", []),
                alpha=float(state.get("prob_alpha", 0.08)),
                beta=float(state.get("prob_beta", 0.35)),
            )

            mu_to_use = news_out["adjusted_mu"]
            cov_to_use = news_out["adjusted_cov"]

            state["prob_news_signals"] = news_out.get("ticker_signals", {})
            state["prob_adjusted_mu"] = mu_to_use
            state["prob_adjusted_cov"] = cov_to_use

            state["prob_prediction_evaluation"] = news_out.get("prediction_evaluation")
            state["historical_prediction_evaluation"] = news_out.get("historical_prediction_evaluation")

            mu_before = state["mu"].copy()
            cov_before = state["cov"].copy()

            
            state["prob_news_trace"] = {
                "parameters": news_out.get("parameters", {}),
                "ticker_signals": news_out.get("ticker_signals", {}),
                "prediction_signals": news_out.get("prediction_signals", {}),
                "article_signals": news_out.get("article_signals", []),
                "historical_prediction_evaluation": news_out.get("historical_prediction_evaluation"),
                "mu_before": mu_before.to_dict(),
                "mu_after": mu_to_use.to_dict(),
                "mu_delta": (mu_to_use - mu_before).to_dict(),
                "variance_before": {
                    t: float(cov_before.loc[t, t]) for t in cov_before.index
                },
                "variance_after": {
                    t: float(cov_to_use.loc[t, t]) for t in cov_to_use.index
                },
                "variance_delta": {
                    t: float(cov_to_use.loc[t, t] - cov_before.loc[t, t])
                    for t in cov_before.index
                },

            }
            state["debug_notes"].append(
                f"[DEBUG MU BEFORE] {mu_before.to_dict()}"
            )
            state["debug_notes"].append(
                f"[DEBUG MU AFTER] {mu_to_use.to_dict()}"
            )
            state["debug_notes"].append(
                f"[DEBUG MU DELTA] {(mu_to_use - mu_before).to_dict()}"
            )
            state["debug_notes"].append(
                f"[DEBUG VAR DELTA] {state['prob_news_trace'].get('variance_delta')}"
            )

            state["debug_notes"].append(
                f"OptimizationProbNews: prediction_signals={list((news_out.get('prediction_signals') or {}).keys())}"
            )

            state["debug_notes"].append(
                "OptimizationProbNews: probabilistic news adjustment applied before MVO."
            )

        except Exception as e:
            state["debug_notes"].append(
                f"OptimizationProbNews: news adjustment failed -> fallback to original mu/cov: {e}"
            )

    base_res = optimization_agent_from_mu_cov(
    mu=state["mu"],
    cov=state["cov"],
    rf=float(state["rf"]),
    w_max=float(state["w_max"]),
    lambda_l2=float(state["lambda_l2"]),
)

    res = optimization_agent_from_mu_cov(
        mu=mu_to_use,
        cov=cov_to_use,
        rf=float(state["rf"]),
        w_max=float(state["w_max"]),
        lambda_l2=float(state["lambda_l2"]),
    )

    try:
        chosen = str(state.get("objective_key") or "maxsharpe").lower().strip()
        if chosen not in base_res:
            chosen = "maxsharpe"
        if chosen not in res:
            chosen = "maxsharpe"

        evaluation = evaluate_news_adjustment_effect(
            base_weights=base_res.get(chosen, {}).get("weights", {}),
            base_metrics=base_res.get(chosen, {}),
            news_weights=res.get(chosen, {}).get("weights", {}),
            news_metrics=res.get(chosen, {}),
        )

        state["news_adjustment_evaluation"] = evaluation
        state["debug_notes"].append(
            "NewsEvaluation: computed base vs news-adjusted comparison."
        )

    except Exception as e:
        state["news_adjustment_evaluation"] = None
        state["debug_notes"].append(f"NewsEvaluation failed: {e}")

    state["optimization_result"] = res
    state["debug_notes"].append("OptimizationProbNews: done.")
    return state

def node_optimize_prediction_constraint(state: PortfolioState) -> PortfolioState:

    if state.get("mu") is None or state.get("cov") is None:
        state["optimization_result"] = {}
        

        state["debug_notes"].append(
            "OptimizationPredictionConstraint: skipped (missing mu/cov)."
        )

        return state

    mu_to_use = state["mu"]
    cov_to_use = state["cov"]
    prediction_probs = {}
    news_constraints = {}
    baseline_weights = {}

    try:
        state["debug_notes"].append(
            "PredictionConstraint: ENTER TRY BLOCK"
        )
        state["debug_notes"].append(
            "PredictionConstraint: importing integration module"
        )


        from news_constraint_integration import (
            build_news_probability_constraints
        )
        state["debug_notes"].append(
            "PredictionConstraint: import success"
        )

        # =====================================================
        # Load latest prediction signals
        # =====================================================

        from pathlib import Path

        csv_path = Path(
            "data/news_prediction/latest_news_prediction_signals.csv"
        )

        state["debug_notes"].append(
            f"PredictionConstraint: csv_path={csv_path}"
        )

        state["debug_notes"].append(
            f"PredictionConstraint: csv_exists={csv_path.exists()}"
        )
        latest_signals = pd.read_csv(csv_path)

        prediction_probs = {
        str(row["ticker"]).upper(): float(row["predicted_positive_probability"])
        for _, row in latest_signals.iterrows()
        }

        state["prediction_probs"] = prediction_probs

        # =====================================================
        # FIRST:
        # Build baseline portfolio
        # (without news constraints)
        # =====================================================

        baseline_res = optimization_agent_from_mu_cov(
            mu=mu_to_use,
            cov=cov_to_use,
            rf=float(state["rf"]),
            w_max=float(state["w_max"]),
            lambda_l2=float(state["lambda_l2"]),
        )
        state["debug_notes"].append(
            f"PredictionConstraint: baseline_res_keys={list(baseline_res.keys())}"
        )

        state["debug_notes"].append(
            f"PredictionConstraint: baseline_maxsharpe="
            f"{baseline_res.get('maxsharpe')}"
        )

        objective_key = str(
            state.get("objective_key", "maxsharpe")
        ).lower().strip()

        state["debug_notes"].append(
            f"PredictionConstraint: objective_key={objective_key}"
        )
        baseline_weights = (
            baseline_res
            .get(objective_key, {})
            .get("weights", {})
        )
        state["debug_notes"].append(
            f"PredictionConstraint: baseline_weights={baseline_weights}"
        )

        # =====================================================
        # Build threshold-based constraints
        # =====================================================

        news_constraints = build_news_probability_constraints(
            latest_signals=latest_signals,
            baseline_weights=baseline_weights,
            bullish_threshold=0.60,
            bearish_threshold=0.40,
            delta=0.02,
            w_max=float(state["w_max"]), 
        )

        state["prediction_model_used"] = True

        state["debug_notes"].append(
            f"PredictionConstraint: constraints={news_constraints}"
        )


    except Exception as e:

        state["prediction_model_used"] = False

        state["debug_notes"].append(
            f"PredictionConstraint failed -> fallback normal optimization: {e}"
        )

        news_constraints = {}

    # =====================================================
    # FINAL constrained optimization
    # =====================================================

    res = prediction_constrained_optimization_agent(
        mu=mu_to_use,
        cov=cov_to_use,
        rf=float(state["rf"]),
        w_max=float(state["w_max"]),
        lambda_l2=float(state["lambda_l2"]),
        news_constraints=news_constraints,
    )

    state["optimization_result"] = res

    state["debug_notes"].append(
        "OptimizationPredictionConstraint: done."
    )

    # =====================================================
    # Store dashboard visualization fields
    # =====================================================

    # baseline weights (before prediction constraints)
    state["baseline_candidate_weights"] = baseline_weights

    # final optimized weights
    state["optimized_weights"] = (
        res.get(objective_key, {})
        .get("weights", {})
    )
    raw_m = res.get(objective_key, {})
    state["optimized_metrics"] = {
        "return": raw_m.get("return"),
        "vol": raw_m.get("vol"),
        "sharpe": raw_m.get("sharpe"),
        "return_pct": (raw_m.get("return") or 0) * 100,
        "vol_pct": (raw_m.get("vol") or 0) * 100,
        "active_assets": len([
            v for v in state["optimized_weights"].values()
            if abs(v) > 1e-6
        ]),
        "max_weight": max(state["optimized_weights"].values())
            if state["optimized_weights"] else 0.0,
    }
    # Pie chart ve header için chosen_candidate set et
    state["chosen_candidate"] = objective_key
    state["debug_notes"].append(
        f"PredictionConstraint: chosen_candidate set to {objective_key}"
    )

    # ✅ FIX 2: prediction_constraint_summary — UI elementleri için
    state["prediction_constraint_summary"] = {
        "constraints_applied": list(news_constraints.keys()),
        "bullish": [t for t, c in news_constraints.items() if c.get("type") == "bullish"],
        "bearish": [t for t, c in news_constraints.items() if c.get("type") == "bearish"],
        "delta": 0.02,
        "model": "LogisticRegression",
        "constraint_type": "side_constraints",
    }


    constraint_debug_enriched = {}

    for row in res.get("constraint_debug", []):

        ticker = row["ticker"]

        prob = prediction_probs.get(ticker, 0.5)

        enriched_row = {
            **row,

            "prediction_probability": float(prob),

            "is_bullish": bool(prob >= 0.60),

            "is_bearish": bool(prob <= 0.40),
        }

        constraint_debug_enriched[ticker] = enriched_row

    state["constraint_debug"] = constraint_debug_enriched
    state["debug_notes"].append(
    f"constraint_debug_keys={list(state['constraint_debug'].keys())}"
    )

    # prediction-adjusted caps
    prediction_caps = {}

    for ticker in state.get("selected_tickers", []):

        adjusted_cap = float(state["w_max"])

        if ticker in news_constraints:

            cdict = news_constraints[ticker]

            if "max_weight" in cdict:
                adjusted_cap = float(cdict["max_weight"])

            elif "min_weight" in cdict:
                adjusted_cap = float(cdict["min_weight"])

        prediction_caps[ticker] = adjusted_cap

    state["prediction_adjusted_caps"] = prediction_caps

    return state
def route_after_data_prob_news(state: PortfolioState) -> str:
    if state.get("mode") == "base":
        return "skip_news"
    return "do_news" if bool(state.get("use_news", False)) else "skip_news"

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

    use_prob_news_metrics = (
        state.get("mode") != "base"
        and bool(state.get("use_news", False))
        and state.get("prob_adjusted_mu") is not None
        and state.get("prob_adjusted_cov") is not None
    )

    for k, item in cands.items():
        w_dict = item.get("weights") or {}

        if not w_dict:
            item["metrics"] = {}
            continue

        try:
            if use_prob_news_metrics:
                # ✅ Important:
                # The portfolio was optimized with news-adjusted mu/cov,
                # so its metrics must also be evaluated with the same adjusted inputs.
                mu_eval = state["prob_adjusted_mu"]
                cov_eval = state["prob_adjusted_cov"]

                aligned_tickers = list(mu_eval.index)
                w = pd.Series(w_dict, dtype=float).reindex(aligned_tickers).fillna(0.0)

                s = float(w.sum())
                if s <= 0:
                    raise ValueError("Candidate weights sum to zero after alignment.")
                w = w / s

                ret = float(w.values @ mu_eval.values)
                vol = float((w.values @ cov_eval.values @ w.values) ** 0.5)
                sharpe = float((ret - float(state["rf"])) / vol) if vol > 0 else None

                weights_full = {t: float(w.loc[t]) for t in aligned_tickers}
                active_weights = _extract_active_weights(weights_full)

                m = {
                    "tickers": aligned_tickers,
                    "weights": weights_full,
                    "return": ret,
                    "vol": vol,
                    "sharpe": sharpe,
                    "max_weight": _safe_max_weight(active_weights),
                    "effective_n": _effective_n(active_weights),
                    "active_assets": len(active_weights),
                    "return_pct": ret * 100.0,
                    "vol_pct": vol * 100.0,
                    "max_weight_pct": _safe_max_weight(active_weights) * 100.0,
                    "rc_abs": [],
                    "rc_pct": [],
                }

                state["debug_notes"].append(
                    f"RiskCandidates(PROB_NEWS): {k} evaluated with adjusted mu/cov."
                )

            else:
                # ✅ Normal/base/LLM-action flow stays unchanged
                m = risk_agent(w_dict, tickers, rf=float(state["rf"]))
                m = _attach_concentration_metrics(m, w_dict)

            item["metrics"] = _normalize_metrics(m, rf=float(state["rf"]))

        except Exception as e:
            item["metrics"] = {}
            state["debug_notes"].append(f"RiskCandidates: failed for {k}: {e}")

    state["debug_notes"].append("RiskCandidates: computed metrics for candidates.")
    return state


# =========================================================
# ✅ REAL news fetch (gated)
# =========================================================
def node_news_fetch(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)
    tickers = state.get("selected_tickers", []) or []

    # ✅ hard rule: base never does news
    if state.get("mode") == "base":
        state["news_raw"] = []
        state["news_snapshot_text"] = None
        empty = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        state["news_risk_json"] = empty
        state["news_signals"] = empty
        state["debug_notes"].append("NewsFetch(BASE): skipped (base run).")
        return state

    # ✅ checkbox off => skip
    if not bool(state.get("use_news", False)):
        state["news_raw"] = []
        state["news_snapshot_text"] = None
        empty = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        state["news_risk_json"] = empty
        state["news_signals"] = empty
        state["debug_notes"].append("NewsFetch: skipped (use_news=False).")
        return state

    if not tickers:
        state["news_raw"] = []
        state["debug_notes"].append("NewsFetch: skipped (no tickers).")
        return state

    if news_agent_fetch_for_tickers is None:
        state["news_raw"] = [{"ticker": t, "headline": None, "source": None, "ts": None} for t in tickers]
        state["debug_notes"].append(
            f"NewsFetch: fallback stub (news_agent_fetch_for_tickers unavailable), n={len(tickers)}."
        )
        return state

    lookback_days = int((_merged_prefs(state).get("lookback_days") or 7))
    max_items_per_ticker = int((_merged_prefs(state).get("news_max_items_per_ticker") or 12))

    try:
        fetched = news_agent_fetch_for_tickers(
        tickers=tickers,
        include_news=True,
        lookback_days=lookback_days,
        min_company_items=1,
        max_items_per_ticker=max_items_per_ticker,
        include_market_fallback=True,
        market_category="general",
        cache_ttl_s=600,
        sleep_s=0.25,
    )

        # ✅ Agents artık kanıt ID'lerini üretiyor: flat_items + evidence_map
        flat_items = (fetched or {}).get("flat_items") or []
        evidence_map = (fetched or {}).get("evidence_map") or {}

        state["news_raw"] = flat_items
        state["evidence_map"] = evidence_map
        

        state["debug_notes"].append(
            f"NewsFetch: agents ok tickers={len(tickers)} items={len(flat_items)} evidence_map={len(evidence_map)} "
            f"lookback_days={lookback_days} max_per_ticker={max_items_per_ticker}"
        )

        stats = (fetched or {}).get("stats") or {}
        if stats:
            state["debug_notes"].append(
                f"NewsFetchStats: company_used={stats.get('company_used')} fallback_used={stats.get('fallback_used')} "
                f"errors={len((stats.get('errors') or {}))}"
            )

        return state


    except Exception as e:
        state["news_raw"] = [{"ticker": t, "headline": None, "source": None, "ts": None} for t in tickers]
        state["debug_notes"].append(f"NewsFetch: failed → stub fallback: {e}")
        return state


# =========================================================
# ✅ News Snapshot + Risk Check (LLM, with placeholder fallback)
# =========================================================
def node_news_snapshot_and_risk(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    state["debug_notes"].append(
        f"[DEBUG] ENTER node_news_snapshot_and_risk "
        f"stage={state.get('stage')} "
        f"use_news={state.get('use_news')} "
        f"use_llm={state.get('use_llm')}"
    )

    # base never does news
    if state.get("mode") == "base":
        state["news_snapshot_text"] = None
        empty = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        empty = _clean_risk_json_fill_universe(empty, state.get("selected_tickers", []) or [])
        state["news_risk_json"] = empty
        state["news_signals"] = empty
        state["debug_notes"].append("NewsSnapshot(BASE): skipped.")
        return state

    if not bool(state.get("use_news", False)):
        state["news_snapshot_text"] = None
        empty = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        empty = _clean_risk_json_fill_universe(empty, state.get("selected_tickers", []) or [])
        state["news_risk_json"] = empty
        state["news_signals"] = empty
        state["debug_notes"].append("NewsSnapshot: skipped (use_news=False).")
        return state

    tickers = state.get("selected_tickers", []) or []
    raw = state.get("news_raw") or []
    # ✅ BYPASS: stage=news_actions => snapshot üretme, sadece evidence_map hazırla
    
    if state.get("stage") == "news_actions":
        state["debug_notes"].append(
            "[DEBUG] BYPASS TRIGGERED (stage=news_actions) "
            "=> snapshot emptied + risk emptied"
        )
        per_ticker = int((_merged_prefs(state).get("news_items_per_ticker") or 3))
        max_total = int((_merged_prefs(state).get("news_max_items_total") or 140))

        selected_raw = _select_news_items_for_llm(
            news_raw=raw,
            tickers=tickers,
            per_ticker=per_ticker,
            max_total=max_total,
        )

        # ✅ IMPORTANT: NewsActionsGenerate node'u bunu kullanıyor
        state["news_items_llm"] = selected_raw

        # ✅ IMPORTANT: evidence_map zaten node_news_fetch içinde state'e yazılıyor
        evidence_map = state.get("evidence_map") or {}

        # snapshot intentionally empty
        state["news_snapshot_text"] = ""
        empty = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        empty = _clean_risk_json_fill_universe(empty, tickers)
        state["news_risk_json"] = empty
        state["news_signals"] = empty

        state["debug_notes"].append(
            f"NewsSnapshot(BYPASS stage=news_actions): items={len(selected_raw)} evidence_map={len(evidence_map)}"
        )
        return state


    # LLM disabled/unavailable -> placeholder fallback
    if (not bool(state.get("use_llm", False))) or (LLMClient is None):
        signals: Dict[str, Any] = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        keywords = (
            "lawsuit",
            "fraud",
            "bankrupt",
            "guidance cut",
            "downgrade",
            "shock",
            "sec",
            "probe",
            "investigation",
        )

        for item in raw:
            t = str(item.get("ticker") or "").upper().strip()
            h = (item.get("headline") or "").lower()
            s = (item.get("summary") or "").lower()
            text = f"{h} {s}"
            if not t:
                continue

            risk_flag = signals["by_ticker"].get(t, {}).get("risk_flag", "none")
            conf = float(signals["by_ticker"].get(t, {}).get("confidence", 0.0) or 0.0)

            if any(k in text for k in keywords):
                risk_flag = "event_risk"
                conf = max(conf, 0.6)
                signals["global"]["risk_flags"].append({"ticker": t, "flag": "event_risk"})
                signals["global"]["vol_regime"] = "high"

            if t not in signals["by_ticker"]:
                signals["by_ticker"][t] = {"risk_flag": risk_flag, "confidence": conf}
            else:
                signals["by_ticker"][t]["risk_flag"] = risk_flag
                signals["by_ticker"][t]["confidence"] = conf

        # ✅ NEW: fill universe + normalize schema
        signals = _clean_risk_json_fill_universe(signals, tickers)

        state["news_snapshot_text"] = "News snapshot (fallback) generated."
        state["news_risk_json"] = signals
        state["news_signals"] = signals
        state["debug_notes"].append(f"NewsSnapshot(Fallback): used keyword placeholder, raw_items={len(raw)}.")
        return state

    # LLM path (balanced prompt input + deterministic post-clean)
    try:
        # ✅ NEW: prevent 1-2 tickers from dominating the prompt
        per_ticker = int((_merged_prefs(state).get("news_items_per_ticker") or 3))
        max_total = int((_merged_prefs(state).get("news_max_items_total") or 140))

        selected_raw = _select_news_items_for_llm(
            news_raw=raw,
            tickers=tickers,
            per_ticker=per_ticker,
            max_total=max_total,
        )
        
        state["news_items_llm"] = selected_raw

       
        evidence_map = state.get("evidence_map") or {}
        state["debug_notes"].append(f"EvidenceMap(from_agents): items={len(evidence_map)}")

       
        try:
            cnt = Counter([str(it.get("ticker") or "").upper().strip() for it in (selected_raw or [])])
           
            if "" in cnt:
                del cnt[""]
            state["debug_notes"].append("NewsItemsPassedToLLM(counts): " + str(dict(cnt)))

            dts = [_parse_dt(it.get("datetime")) for it in (selected_raw or [])]
            dts = [d for d in dts if d is not None]
            if dts:
                state["debug_notes"].append(
                    f"NewsItemsPassedToLLM(date_range): {min(dts).isoformat()} -> {max(dts).isoformat()}"
                )
        except Exception as e:
            state["debug_notes"].append(f"NewsItemsPassedToLLM(debug_failed): {e}")



        state["debug_notes"].append(
            f"NewsSnapshotInput: raw_items={len(raw)} selected_items={len(selected_raw)} "
            f"per_ticker={per_ticker} max_total={max_total}"
        )

        client = LLMClient()
        out = client.generate_news_snapshot(
            tickers=tickers,
            news_raw=selected_raw, 
            lookback_days=int((_merged_prefs(state).get("lookback_days") or 7)),
            max_items_total=len(selected_raw), 
        )
       
        try:
            state["debug_notes"].append(f"NewsSnapshotOut: ok={out.get('ok')} parse={out.get('parse_mode')}")
            st_preview = (str(out.get("snapshot_text") or "")[:220]).replace("\n", " ")
            state["debug_notes"].append(f"NewsSnapshotOut(snapshot_preview_220): {st_preview}")
            rj = out.get("risk_json") if isinstance(out.get("risk_json"), dict) else {}
            bt = rj.get("by_ticker") if isinstance(rj.get("by_ticker"), dict) else {}
            state["debug_notes"].append(f"NewsSnapshotOut(risk_by_ticker_keys): {sorted(list(bt.keys()))}")

            
            import re
            ids = re.findall(r"\(\[([^\]]+)\]", str(out.get("snapshot_text") or ""))
            state["debug_notes"].append(f"NewsSnapshotOut(evidence_ids_in_snapshot): {len(ids)}")
        except Exception as e:
            state["debug_notes"].append(f"NewsSnapshotOut(debug_failed): {e}")

        snapshot_text_canonical = str(out.get("snapshot_text") or "").strip()
        snapshot_text_ui = str(out.get("snapshot_text_raw") or "").strip()

        state["news_snapshot_text"] = snapshot_text_canonical or None
        state["news_snapshot_text_raw"] = snapshot_text_ui or snapshot_text_canonical or None
        state["debug_notes"].append(
            f"NewsSnapshotDebug: snapshot_len={len(state.get('news_snapshot_text') or '')} "
            f"raw_len={len(state.get('news_snapshot_text_raw') or '')}"
        )
        state["debug_notes"].append(
            "NewsSnapshotDebug(raw_preview_200): " + (state.get("news_snapshot_text_raw") or "")[:200].replace("\n", " ")
        )
        # ✅ NEW: UI-friendly raw snapshot (strip evidence tags)
        snap = state.get("news_snapshot_text") or ""
        # örn: "([APP_abc] 2026-02-22 | Yahoo)" gibi tag'leri kaldır
        snap_raw = re.sub(r"\(\[[A-Z0-9_]+\]\s+[0-9]{4}-[0-9]{2}-[0-9]{2}\s+\|\s+[^)]+\)\s*", "", snap)
        state["news_snapshot_text_raw"] = snap_raw.strip() or None
        state["debug_notes"].append(
            f"[DEBUG] AFTER LLM snapshot_len="
            f"{len(state.get('news_snapshot_text') or '')}"
        )
        risk_json = out.get("risk_json") if isinstance(out.get("risk_json"), dict) else {}

        # ✅ NEW: deterministic post-clean + fill universe + normalize enums
        risk_json_clean = _clean_risk_json_fill_universe(risk_json, tickers)


        def _validate_snapshot_format(snapshot_text: str, evidence_map: dict, max_per_ticker: int = 6) -> list[str]:
            logs = []
            text = snapshot_text or ""
            # 1) formatta evidence_id yakala
            ids = re.findall(r"\(\[([A-Z0-9_]+)\]\s+([0-9]{4}-[0-9]{2}-[0-9]{2}|unknown)\s+\|\s+([^)]+)\)", text)

            logs.append(f"SnapshotCheck: evidence_tags_found={len(ids)}")
            if not ids:
                logs.append("SnapshotCheck:  No '([EVIDENCE_ID] DATE | SOURCE)' tags found.")
                return logs

            # 2) evidence_map’te var mı?
            missing = []
            per_ticker = Counter()
            for (eid, date, source) in ids:
                if eid not in (evidence_map or {}):
                    missing.append(eid)
                t = eid.split("_")[0] if "_" in eid else "UNK"
                per_ticker[t] += 1

            if missing:
                logs.append(f"SnapshotCheck: ❌ evidence_ids missing in evidence_map: {sorted(set(missing))[:20]}")
            else:
                logs.append("SnapshotCheck: ✅ all evidence_ids exist in evidence_map.")

            # 3) ticker başına limit
            over = {t:c for t,c in per_ticker.items() if c > max_per_ticker}
            if over:
                logs.append(f"SnapshotCheck: ❌ per-ticker bullet overflow: {over}")
            else:
                logs.append(f"SnapshotCheck: ✅ per-ticker bullets <= {max_per_ticker}. counts={dict(per_ticker)}")

            return logs

        state["news_risk_json"] = risk_json_clean
        state["news_signals"] = risk_json_clean

        state["debug_notes"].append(
            f"NewsSnapshot(LLM): ok={bool(out.get('ok'))} issues={len(out.get('issues') or [])} parse={out.get('parse_mode')}"
        )
        state["debug_notes"].append(
            f"NewsRiskCoverage: by_ticker={len((risk_json_clean.get('by_ticker') or {}))} "
            f"global_flags={len(((risk_json_clean.get('global') or {}).get('risk_flags') or []))}"
        )
        state["debug_notes"].extend(_validate_snapshot_format(state.get("news_snapshot_text") or "", evidence_map, max_per_ticker=6))

        return state
    # snapshot_text ve evidence_map hazır olduktan sonra:

    except Exception as e:
        state["news_snapshot_text"] = None
        empty = {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}}
        empty = _clean_risk_json_fill_universe(empty, tickers)
        state["news_risk_json"] = empty
        state["news_signals"] = empty
        state["debug_notes"].append(f"NewsSnapshot(LLM): failed -> empty risk_json: {e}")
        return state


# =========================================================
# ✅ News Actions stage nodes
# =========================================================
def node_news_actions_generate(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)
    state["debug_notes"].append(
        f"ENTER NewsActionsGenerate: stage={state.get('stage')} mode={state.get('mode')} "
        f"use_news={state.get('use_news')} use_llm={state.get('use_llm')} LLMClient_is_None={LLMClient is None}"
    )

    if state.get("stage") != "news_actions":
        state["debug_notes"].append("NewsActions: skipped (stage!=news_actions).")
        return state

    if state.get("mode") == "base":
        state["news_actions"] = []
        state["news_actions_verifier"] = {"ok": True, "notes": ["Base mode: no actions."]}
        state["debug_notes"].append("NewsActions(BASE): skipped.")
        return state

    if not bool(state.get("use_news", False)):
        state["news_actions"] = []
        state["news_actions_verifier"] = {"ok": True, "notes": ["use_news=False: no actions."]}
        state["debug_notes"].append("NewsActions: skipped (use_news=False).")
        return state

    if (not bool(state.get("use_llm", False))) or (LLMClient is None):
        state["news_actions"] = []
        state["news_actions_verifier"] = {"ok": True, "notes": ["LLM disabled/unavailable."]}
        state["debug_notes"].append("NewsActions: skipped (LLM disabled/unavailable).")
        return state

    tickers = list(state.get("selected_tickers", []) or [])
    snapshot_text = state.get("news_snapshot_text") or ""
    risk_json = state.get("news_risk_json") or state.get("news_signals") or {}

    snapshot_obj = {"snapshot_text": snapshot_text}

    try:
        client = LLMClient()
        news_items = state.get("news_items_llm") or []  # ✅ LLM snapshot’ta gördüğü aynı balanced liste
        state["debug_notes"].append(f"NewsActionsSAMPLE_first_item: {str(news_items[0])[:180] if news_items else 'EMPTY'}")
        state["debug_notes"].append(f"NewsActionsEvidenceMapSize: {len(state.get('evidence_map') or {})}")


        state["debug_notes"].append(f"NewsActionsInput: news_items_llm={len(news_items)}")

        out = client.generate_news_actions_lines(
        tickers=tickers,
        news_items=news_items,
        max_actions=8,
    )
        state["debug_notes"].append(
            "NewsActionsLinesMeta: "
            f"parse_mode={out.get('parse_mode')} "
            f"used_fixer={out.get('used_fixer', False)} "
            f"issues={len(out.get('issues') or [])} "
            f"text_len={len(str(out.get('raw_text') or out.get('text') or ''))}"
        )



        # 🔴 1️⃣ HAM LLM ÇIKTISI
        raw_actions = out.get("actions") if isinstance(out.get("actions"), list) else []
                # ✅ DEBUG: see what LLM actually returned (text + parser metadata)
        try:
            raw_text = out.get("raw_text") or out.get("text") or ""
            raw_text = str(raw_text)
            state["debug_notes"].append(
                "NewsActionsRawTextPreview_800: " + raw_text[:800].replace("\n", " ")
            )
            state["debug_notes"].append(
                f"NewsActionsParseMode: {out.get('parse_mode')}"
            )
            issues = out.get("issues")
            if issues:
                state["debug_notes"].append(
                    "NewsActionsIssues: " + str(issues)[:500]
                )
        except Exception as _e:
            state["debug_notes"].append(f"NewsActionsRawDebugFailed: {_e}")

        state["debug_notes"].append(
            f"NewsActions(LLM): raw_actions={len(raw_actions)} → {raw_actions}"
        )

        # 🟢 2️⃣ DETERMINISTIC CLEAN / NORMALIZATION
        snapshot_text = state.get("news_snapshot_text") or ""
        allowed_eids = set((state.get("evidence_map") or {}).keys())
        state["debug_notes"].append(f"NewsActionsAllowedEIDs(from_evidence_map): n={len(allowed_eids)}")


        actions = _clean_news_actions(
            raw_actions,
            tickers,
            allowed_eids=allowed_eids,
        )

        # ✅ DEBUG: ACTION EID -> MAP CHECK (EvidenceSnapshot'tan önce)
        try:
            evidence_map = state.get("evidence_map") or {}
            state["debug_notes"].append("=== DEBUG: ACTION EID -> MAP CHECK ===")
            for a in actions:
                for eid in a.get("evidence_ids", []):
                    info = evidence_map.get(eid)
                    state["debug_notes"].append(f"{eid} -> {info}")
            state["debug_notes"].append("=== /DEBUG: ACTION EID -> MAP CHECK ===")
            if info and isinstance(info, dict):
                map_ticker = str(info.get("ticker") or "").upper().strip()
                eid_ticker = eid.split("_", 1)[0]
                if map_ticker and map_ticker != eid_ticker:
                    state["debug_notes"].append(f"⚠️ TICKER MISMATCH: eid={eid_ticker} map={map_ticker} for {eid}")

        except Exception as e:
            state["debug_notes"].append(f"DEBUG ACTION EID MAP CHECK failed: {e}")

        state["debug_notes"].append(
            f"NewsActions(CLEAN): final_actions={len(actions)} → {actions}"
        )


        state["debug_notes"].append(
            f"NewsActions(CLEAN): final_actions={len(actions)} → {actions}"
        )
        # 🟡 2.5) DEBUG: why drops happened
        drop_logs = _debug_explain_action_drop_reasons(
            raw_actions=raw_actions,
            cleaned_actions=actions,
            tickers=tickers,
            allowed_eids=allowed_eids,
            max_logs=60,
        )
        state["debug_notes"].extend(drop_logs)


        # 🟢 3️⃣ STATE’E YAZ
        state["news_actions"] = actions
        state["debug_notes"].append(
            f"[CHECK] state.news_actions_len={len(state.get('news_actions') or [])} "
            f"sample={ (state.get('news_actions') or [])[:2] }"
        )

        state["debug_notes"].append(
            f"NewsActions(LLM): generated n={len(actions)} (post-clean)."
        )
        return state

    except Exception as e:
        state["news_actions"] = []
        state["debug_notes"].append(f"NewsActions(LLM): failed -> empty: {e}")
        return state
    

def node_news_evidence_snapshot(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)
    state["debug_notes"].append(
    f"ENTER EvidenceSnapshot: stage={state.get('stage')} "
    f"mode={state.get('mode')} use_news={state.get('use_news')} use_llm={state.get('use_llm')} "
    f"actions={len(state.get('news_actions') or [])} evidence_map={len(state.get('evidence_map') or {})}"
)

    # sadece news_actions stage'de çalışsın
    if state.get("stage") != "news_actions":
        return state

    # base veya LLM/news kapalıysa skip
    if state.get("mode") == "base":
        state["news_evidence_snapshot_text"] = ""
        state["news_evidence_snapshot_ok"] = True
        state["news_evidence_snapshot_issues"] = []
        state["debug_notes"].append("EvidenceSnapshot(BASE): skipped.")
        return state

    if (not bool(state.get("use_news", False))) or (not bool(state.get("use_llm", False))) or (LLMClient is None):
        state["news_evidence_snapshot_text"] = ""
        state["news_evidence_snapshot_ok"] = True
        state["news_evidence_snapshot_issues"] = ["disabled_or_unavailable"]
        state["debug_notes"].append("EvidenceSnapshot: skipped (use_news/use_llm off or LLMClient unavailable).")
        return state

    actions = state.get("news_actions") or []
    evidence_map = state.get("evidence_map") or {}
    tickers = list(state.get("selected_tickers", []) or [])

    # 1) actions -> evidence_ids set
    eids: Set[str] = set()
    for a in actions:
        if not isinstance(a, dict):
            continue
        ev = a.get("evidence_ids")
        if isinstance(ev, list):
            for x in ev:
                sx = str(x).strip()
                if sx:
                    eids.add(sx)

    state["debug_notes"].append(f"EvidenceSnapshot: collected_eids={len(eids)} from actions={len(actions)}")

    if not eids:
        state["news_evidence_snapshot_text"] = ""
        state["news_evidence_snapshot_ok"] = True
        state["news_evidence_snapshot_issues"] = ["no_evidence_ids"]
        state["debug_notes"].append("EvidenceSnapshot: no evidence_ids -> empty.")
        return state

    # 2) evidence_map -> minimal news list (LLMClient.generate_news_snapshot expects news_raw-like dicts)
    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for eid in sorted(eids):
        it = evidence_map.get(eid)
        if not isinstance(it, dict):
            missing.append(eid)
            continue

        t = str(it.get("ticker") or "").upper().strip()
        if not t and "_" in eid:
            t = eid.split("_", 1)[0].upper().strip()

        date = it.get("date") or _epoch_to_ymd(it.get("datetime")) or "unknown"
        src = it.get("source") or it.get("provider") or "unknown"

        items.append(
            {
                "id": it.get("id"),
                "ticker": t,
                "evidence_id": it.get("evidence_id") or eid,
                "date": date,
                "headline": it.get("headline"),
                "summary": it.get("summary"),
                "source": src,
                "url": it.get("url"),
                "datetime": it.get("datetime"),
            }
        )

    if missing:
        state["debug_notes"].append(f"EvidenceSnapshot: missing_in_evidence_map={missing[:10]}")

    # 3) LLM ile snapshot üret (sadece evidence item’ları)
    try:
        client = LLMClient()
        out = client.generate_evidence_snapshot_from_actions(
            actions=actions,
            news_items=items,   # items zaten evidence_id + headline + summary içeriyor
            max_items=12,
        )

        text = str(out.get("snapshot_text") or out.get("text") or out.get("raw_text") or "").strip()
        ok = bool(out.get("ok"))
        issues = list(out.get("issues") or [])
        if ok and not text:
            ok = False
            issues.append("missing_snapshot_text")

        state["news_evidence_snapshot_text"] = text
        state["news_evidence_snapshot_ok"] = ok
        state["news_evidence_snapshot_issues"] = issues

        state["debug_notes"].append(
            f"EvidenceSnapshot(LLM): ok={ok} issues={len(issues)} items={len(items)}"
        )
        prev = text[:220].replace("\n", " ")
        state["debug_notes"].append(f"EvidenceSnapshotPreview_220: {prev}")

        return state

    except Exception as e:
        state["news_evidence_snapshot_text"] = ""
        state["news_evidence_snapshot_ok"] = False
        state["news_evidence_snapshot_issues"] = [f"exception: {e}"]
        state["debug_notes"].append(f"EvidenceSnapshot(LLM): failed -> {e}")
        return state

def node_news_actions_verify(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

    if state.get("stage") != "news_actions":
        return state

    # Line-based + whitelist + _clean_news_actions zaten en sıkı doğrulama
    state["news_actions_verifier"] = {"ok": True, "notes": ["Verifier skipped (line-based + deterministic cleaner)."]}
    state["debug_notes"].append("NewsActionsVerifier: skipped (deterministic line-based flow).")
    return state


def route_after_risk_candidates(state: PortfolioState) -> str:
    if state.get("mode") == "base":
        return "skip_news"
    return "do_news" if bool(state.get("use_news", False)) else "skip_news"


def route_after_news_snapshot(state: PortfolioState) -> str:
    # ✅ news_overview = main flow + overlay, so continue main
    return "news_actions" if state.get("stage") == "news_actions" else "main"


def node_llm_select_candidate(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

 
    if state.get("stage") == "news_actions":
        state["debug_notes"].append("LLM_Select: skipped (stage=news_actions).")
        return state

    if state.get("mode") == "base":
        chosen = state.get("chosen_candidate") or state.get("objective_key", "maxsharpe")
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": "Base run: candidate selection disabled. Portfolio generated for comparison.",
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append("LLM_Select(BASE): accept (no selection).")
        return state
    # ✅ Prediction constraint flow: objective ve chosen_candidate değişmemeli
    if state.get("prediction_model_used"):
        chosen = str(state.get("chosen_candidate") or state.get("objective_key") or "maxsharpe").lower().strip()
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": (
                "Prediction constraint flow: objective locked. "
                "Feasible set constrained by news predictions; objective unchanged."
            ),
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append(
            f"LLM_Select(PREDICTION_CONSTRAINT_LOCK): chosen={chosen}"
        )
        return state
    
        # ✅ Mathematical news integration:
    # News already changed mu/cov before optimization.
    # Do NOT let LLM switch maxsharpe/minvar here.
    # Keep objective fixed for clean thesis evaluation.
    if (
        state.get("mode") != "base"
        and bool(state.get("use_news", False))
        and state.get("prob_news_signals") is not None
    ):
        chosen = str(state.get("objective_key") or "maxsharpe").lower().strip()
        candidates = state.get("candidates") or {}

        if chosen not in candidates:
            chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))

        state["chosen_candidate"] = chosen  # type: ignore
        state["llm_decision"] = {
            "decision": "accept",
            "rationale": (
                "Mathematical news integration: candidate selection is locked to the base objective. "
                "News affects the optimization inputs, not the objective choice."
            ),
            "chosen_candidate": chosen,
        }
        state["debug_notes"].append(
            f"LLM_Select(PROB_NEWS_LOCK): chosen={chosen}, LLM candidate selection skipped."
        )
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

    # ✅ news is EXTRA overlay only; do not influence normal candidate selection.
    if bool(state.get("use_news", False)):
        state["debug_notes"].append(
            "LLM_Select: use_news=True -> ignoring news_signals for candidate selection (extra flow)."
        )

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
                news_signals=None,  # ✅ MUST stay None
            )

            chosen = str(llm_payload.get("chosen_candidate", "")).lower().strip()
            if chosen not in candidates:
                chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))

            rationale = str(llm_payload.get("rationale", "")).strip() or "LLM selected the most preference-aligned candidate."
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
    state = _init_defaults(state)

    # ✅ In news_actions stage we should never finalize portfolio
    if state.get("stage") == "news_actions":
        state["debug_notes"].append("FinalizeSelection: skipped (stage=news_actions).")
        return state
    if state.get("prediction_model_used") and state.get("optimized_weights"):
        state["debug_notes"].append(
            "FinalizeSelection: skipped (prediction_model_used=True, weights already set)."
        )
        return state

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


def node_insight_generator(state: PortfolioState) -> PortfolioState:
        state = _init_defaults(state)
        state["debug_notes"].append(
        f"[INSIGHT DEBUG] objective_key={state.get('objective_key')} "
        f"optimized_metrics={state.get('optimized_metrics')} "
        f"optimized_weights_keys={list((state.get('optimized_weights') or {}).keys())}"
)
        # ✅ In news_actions stage we do not generate portfolio insights
        if state.get("stage") == "news_actions":
            state["debug_notes"].append("Insight: skipped (stage=news_actions).")
            return state

        use_llm = bool(state.get("use_llm", False))
        if (not use_llm) or (LLMClient is None) or (insight_agent_prepare is None):
            state["debug_notes"].append(
                "Insight: skipped (use_llm disabled or LLMClient/insight_agent_prepare unavailable)."
            )
            return state

        refine_metrics = state.get("optimized_metrics") or {}
        if not refine_metrics:
            state["debug_notes"].append("Insight: skipped (missing optimized_metrics).")
            return state

        base_metrics = state.get("base_portfolio_metrics")
        base_obj = state.get("base_portfolio_objective")

        if not base_metrics:
            base_metrics = state.get("current_metrics")
            if base_metrics and not base_obj:
                base_obj = "user_current"

        if not base_metrics:
            base_metrics = state.get("baseline_metrics") or {}
            if not base_obj:
                base_obj = "equal_weight"

        prefs = _merged_prefs(state)
        news_signals = state.get("news_signals")

        chosen = str(state.get("objective_key") or "maxsharpe").lower().strip()
        refine_obj = chosen

        try:
            base_constraints = {"rf": float(state.get("rf", 0.02))}
            refine_constraints = {
                "rf": float(state.get("rf", 0.02)),
                "w_max": float(state.get("w_max", 0.30)),
                "lambda_l2": float(state.get("lambda_l2", 1e-3)),
            }

            # ✅ Keep using agents_langgraph only for payload preparation
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

            payload = prep.get("payload") or {}
            state["debug_notes"].append(
                f"[INSIGHT PAYLOAD CHECK] "
                f"base_obj={base_obj} refine_obj={refine_obj} "
                f"base_return={base_metrics.get('return_pct') if isinstance(base_metrics, dict) else None} "
                f"base_vol={base_metrics.get('vol_pct') if isinstance(base_metrics, dict) else None} "
                f"refine_return={refine_metrics.get('return_pct') if isinstance(refine_metrics, dict) else None} "
                f"refine_vol={refine_metrics.get('vol_pct') if isinstance(refine_metrics, dict) else None}"
            )
            state["debug_notes"].append(
                "[INSIGHT PAYLOAD RAW] " + str(payload)[:3000]
            )

            client = LLMClient()
            out = client.generate_portfolio_insights(
                payload=payload,
                mode="narrative",
            )

            state["insight_ok"] = bool(out.get("ok"))
            state["insight_issues"] = list(out.get("issues") or [])
            state["insight_parse_mode"] = out.get("parse_mode") or "narrative"

            state["insight_raw_text"] = (out.get("text") or out.get("raw_text") or "").strip() or None
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
    state = _init_defaults(state)

    # ✅ In news_actions stage we do not generate explanation
    if state.get("stage") == "news_actions":
        state["explanation"] = ""
        state["debug_notes"].append("Explain: skipped (stage=news_actions).")
        return state

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
        final_metrics=state.get("optimized_metrics") or None,
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
    g.add_node("news_snapshot", node_news_snapshot_and_risk)


    g.add_node("news_actions_generate", node_news_actions_generate)
    g.add_node("news_evidence_snapshot", node_news_evidence_snapshot)
    g.add_node("news_actions_verify", node_news_actions_verify)

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

    g.add_conditional_edges(
        "risk_candidates",
        route_after_risk_candidates,
        {"do_news": "news_fetch", "skip_news": "llm_select"},
    )

    g.add_edge("news_fetch", "news_snapshot")

    # ✅ IMPORTANT: split after news_snapshot by stage
    g.add_conditional_edges(
    "news_snapshot",
    route_after_news_snapshot,
    {
        "news_actions": "news_actions_generate",
        "main": "llm_select",
    },

)

    g.add_edge("news_actions_generate", "news_evidence_snapshot")
    g.add_edge("news_evidence_snapshot", "news_actions_verify")
    g.add_edge("news_actions_verify", END)


    # ✅ normal flow
    g.add_edge("llm_select", "finalize")
    g.add_edge("finalize", "insight")
    g.add_edge("insight", "explain")
    g.add_edge("explain", END)

    return g.compile()

def build_portfolio_graph_prob_news():
    g = StateGraph(PortfolioState)

    g.add_node("ask_clarifications", node_ask_clarifications)
    g.add_node("perception", node_perception)
    g.add_node("baselines", node_compute_baselines)

    g.add_node("data", node_data)
    g.add_node("news_fetch", node_news_fetch)
    g.add_node("optimize_prob_news", node_optimize_prob_news)

    g.add_node("extract_candidates", node_extract_candidates)
    g.add_node("risk_candidates", node_risk_candidates)

    g.add_node("news_snapshot", node_news_snapshot_and_risk)
    g.add_node("news_actions_generate", node_news_actions_generate)
    g.add_node("news_evidence_snapshot", node_news_evidence_snapshot)
    g.add_node("news_actions_verify", node_news_actions_verify)

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

    g.add_conditional_edges(
        "data",
        route_after_data_prob_news,
        {"do_news": "news_fetch", "skip_news": "optimize_prob_news"},
    )

    g.add_edge("news_fetch", "optimize_prob_news")
    g.add_edge("optimize_prob_news", "extract_candidates")
    g.add_edge("extract_candidates", "risk_candidates")

    # optional news overview / actions stays after candidate risk computation
    g.add_conditional_edges(
        "risk_candidates",
        route_after_risk_candidates,
        {"do_news": "news_snapshot", "skip_news": "llm_select"},
    )

    g.add_conditional_edges(
        "news_snapshot",
        route_after_news_snapshot,
        {
            "news_actions": "news_actions_generate",
            "main": "llm_select",
        },
    )

    g.add_edge("news_actions_generate", "news_evidence_snapshot")
    g.add_edge("news_evidence_snapshot", "news_actions_verify")
    g.add_edge("news_actions_verify", END)

    g.add_edge("llm_select", "finalize")
    g.add_edge("finalize", "insight")
    g.add_edge("insight", "explain")
    g.add_edge("explain", END)

    return g.compile()

def build_portfolio_graph_prediction_constraint():
    g = StateGraph(PortfolioState)

    g.add_node("ask_clarifications", node_ask_clarifications)
    g.add_node("perception", node_perception)
    g.add_node("baselines", node_compute_baselines)

    g.add_node("data", node_data)

    g.add_node("news_fetch", node_news_fetch)

    g.add_node(
        "optimize_prediction_constraint",
        node_optimize_prediction_constraint,
    )

    g.add_node("extract_candidates", node_extract_candidates)
    g.add_node("risk_candidates", node_risk_candidates)

    g.add_node("news_snapshot", node_news_snapshot_and_risk)

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

    g.add_conditional_edges(
        "data",
        route_after_data_prob_news,
        {
            "do_news": "news_fetch",
            "skip_news": "optimize_prediction_constraint",
        },
    )

    g.add_edge(
        "news_fetch",
        "optimize_prediction_constraint",
    )

    g.add_edge(
        "optimize_prediction_constraint",
        "extract_candidates",
    )

    g.add_edge("extract_candidates", "risk_candidates")

    g.add_conditional_edges(
        "risk_candidates",
        route_after_risk_candidates,
        {
            "do_news": "news_snapshot",
            "skip_news": "llm_select",
        },
    )

    g.add_edge("news_snapshot", "llm_select")

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
    stage: Stage = "main",  # ✅ NEW
    use_llm: bool = False,
    # ✅ allow passing checkbox programmatically
    use_news: bool = False,
    # ✅ pass base portfolio from the previous Run Base
    base_portfolio_metrics: Optional[Dict[str, Any]] = None,
    base_portfolio_weights: Optional[Dict[str, float]] = None,
    base_portfolio_objective: Optional[str] = None,
) -> PortfolioState:
    app = build_portfolio_graph()

    init: PortfolioState = {
        "mode": mode,
        "stage": stage,  # ✅ NEW

        "selected_tickers": selected_tickers,
        "rf": float(rf),
        "w_max": float(w_max),
        "lambda_l2": 1e-3,
        "preferences": preferences or {},
        "use_llm": bool(use_llm),
        "use_news": bool(use_news) if mode != "base" else False,
        "current_weights": current_weights,
        "debug_notes": [],
        "clarification_answers": clarification_answers,
        "objective_key": "maxsharpe",
        "chosen_candidate": None,
        "candidates": {},
        "llm_decision": None,
        "optimized_weights": {},
        "optimized_metrics": {},
        "news_raw": None,
        "news_signals": None,
        "news_snapshot_text": None,
        "news_risk_json": None,

        # ✅ NEW: action outputs
        "news_actions": None,
        "news_actions_verifier": None,
        "news_items_llm": None,
        "evidence_map": None,

        "prob_news_trace": None,


        "insight": None,
        "insight_ok": None,
        "insight_issues": [],
        "insight_raw_text": None,
        "insight_parse_mode": None,
        "base_portfolio_metrics": base_portfolio_metrics,
        "base_portfolio_weights": base_portfolio_weights,
        "base_portfolio_objective": base_portfolio_objective,
    }

    out = app.invoke(init, config={"recursion_limit": 200})

    out["debug_notes"].append(
        f"[DEBUG FINAL STATE] "
        f"snapshot_len={len(out.get('news_snapshot_text') or '')} "
        f"risk_by_ticker_keys="
        f"{list(((out.get('news_risk_json') or {}).get('by_ticker') or {}).keys())}"
    )

    return out


def run_graph_prob_news(
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    preferences: Optional[Dict[str, Any]] = None,
    current_weights: Optional[Dict[str, float]] = None,
    max_iterations: int = 0,
    clarification_answers: Optional[Dict[str, Any]] = None,
    mode: Mode = "refine",
    stage: Stage = "main",
    use_llm: bool = False,
    use_news: bool = False,
    base_portfolio_metrics: Optional[Dict[str, Any]] = None,
    base_portfolio_weights: Optional[Dict[str, float]] = None,
    base_portfolio_objective: Optional[str] = None,
    prob_alpha: float = 0.08,
    prob_beta: float = 0.35,
) -> PortfolioState:
    app = build_portfolio_graph_prob_news()



    init: PortfolioState = {
        "mode": mode,
        "stage": stage,
        "selected_tickers": selected_tickers,
        "rf": float(rf),
        "w_max": float(w_max),
        "lambda_l2": 1e-3,
        "preferences": preferences or {},
        "use_llm": bool(use_llm),
        "use_news": bool(use_news) if mode != "base" else False,
        "current_weights": current_weights,
        "debug_notes": [],
        "clarification_answers": clarification_answers,
        "objective_key": str(base_portfolio_objective or "maxsharpe").lower().strip(),
        "chosen_candidate": None,
        "candidates": {},
        "llm_decision": None,
        "optimized_weights": {},
        "optimized_metrics": {},
        "news_raw": None,
        "news_signals": None,
        "news_snapshot_text": None,
        "news_risk_json": None,
        "news_actions": None,
        "news_actions_verifier": None,
        "news_items_llm": None,
        "evidence_map": None,
        "prob_news_signals": None,
        "prob_adjusted_mu": None,
        "prob_adjusted_cov": None,
        "insight": None,
        "insight_ok": None,
        "insight_issues": [],
        "insight_raw_text": None,
        "insight_parse_mode": None,
        "base_portfolio_metrics": base_portfolio_metrics,
        "base_portfolio_weights": base_portfolio_weights,
        "base_portfolio_objective": base_portfolio_objective,
        "prob_alpha": float(prob_alpha),
        "prob_beta": float(prob_beta),

        "news_adjustment_evaluation": None,
        "prob_news_trace": None,
        "prob_prediction_evaluation": None,
        "historical_prediction_evaluation": None,
    }

    out = app.invoke(init, config={"recursion_limit": 200})

    out["debug_notes"].append(
        f"[DEBUG FINAL STATE PROB] "
        f"prob_news_signals_keys={list((out.get('prob_news_signals') or {}).keys())}"
    )
    return out

def run_graph_prediction_constraint(
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    preferences: Optional[Dict[str, Any]] = None,
    current_weights: Optional[Dict[str, float]] = None,
    max_iterations: int = 0,
    clarification_answers: Optional[Dict[str, Any]] = None,
    mode: Mode = "refine",
    stage: Stage = "main",
    use_llm: bool = False,
    use_news: bool = False,
    base_portfolio_metrics: Optional[Dict[str, Any]] = None,
    base_portfolio_weights: Optional[Dict[str, float]] = None,
    base_portfolio_objective: Optional[str] = None,
) -> PortfolioState:

    app = build_portfolio_graph_prediction_constraint()

    init: PortfolioState = {
        "mode": mode,
        "stage": stage,

        "selected_tickers": selected_tickers,

        "rf": float(rf),
        "w_max": float(w_max),
        "lambda_l2": 1e-3,

        "preferences": preferences or {},

        "use_llm": bool(use_llm),

        # base mode => no news
        "use_news": bool(use_news) if mode != "base" else False,

        "current_weights": current_weights,

        "debug_notes": [],

        "clarification_answers": clarification_answers,

        # IMPORTANT:
        # Keep same objective as base portfolio
        "objective_key": str(
            base_portfolio_objective or "maxsharpe"
        ).lower().strip(),

        "chosen_candidate": None,

        "candidates": {},

        "llm_decision": None,

        "optimized_weights": {},
        "optimized_metrics": {},

        # news
        "news_raw": None,
        "news_signals": None,
        "news_snapshot_text": None,
        "news_risk_json": None,

        # evidence / actions
        "news_actions": None,
        "news_actions_verifier": None,
        "news_items_llm": None,
        "evidence_map": None,

        # prediction model outputs
        "prediction_probs": None,
        "prediction_adjusted_caps": None,
        "prediction_model_metrics": None,
        "prediction_model_used": None,

        # optional evaluation
        "prob_prediction_evaluation": None,

        # insight
        "insight": None,
        "insight_ok": None,
        "insight_issues": [],
        "insight_raw_text": None,
        "insight_parse_mode": None,

        # base portfolio carry-over
        "base_portfolio_metrics": base_portfolio_metrics,
        "base_portfolio_weights": base_portfolio_weights,
        "base_portfolio_objective": base_portfolio_objective,
    }

    out = app.invoke(
        init,
        config={"recursion_limit": 200},
    )

    out["debug_notes"].append(
        f"[DEBUG FINAL STATE PREDICTION] "
        f"prediction_used={out.get('prediction_model_used')} "
        f"prediction_probs_keys="
        f"{list((out.get('prediction_probs') or {}).keys())}"
    )

    return out