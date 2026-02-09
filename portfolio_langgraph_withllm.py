# portfolio_langgraph.py
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


from agents_langgraph import (
    data_agent_get_mu_cov,
    optimization_agent_from_mu_cov,
    risk_agent,
    recommendation_agent,
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
Stage = Literal["main", "news_actions"]
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

    debug_notes: List[str]
    explanation: str

    # ✅ Insight Generator outputs
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
    state.setdefault("news_risk_json", None)
    state.setdefault("news_items_llm", None)
    state.setdefault("evidence_map", None)
    state.setdefault("news_evidence_snapshot_text", None)
    state.setdefault("news_evidence_snapshot_ok", None)
    state.setdefault("news_evidence_snapshot_issues", [])


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
    if state.get("stage") == "news_actions":
        state["needs_user_input"] = False
        state["debug_notes"].append("Clarifications: skipped (stage=news_actions).")
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
    if state.get("stage") == "news_actions" and state.get("mode") != "base":
        state["use_news"] = True
        state["use_llm"] = True
        state["debug_notes"].append("Perception: stage=news_actions -> forcing use_llm=True.")
        state["debug_notes"].append("Perception: stage=news_actions -> forcing use_news=True.")

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
    if excluded_assets:
        excluded = set(map(str, excluded_assets))
        state["selected_tickers"] = [t for t in list(state.get("selected_tickers", [])) if t not in excluded]
        state["debug_notes"].append(f"Perception: excluded={sorted(excluded)}")

    # ✅ only override programmatic use_news if user explicitly answered it
    # (but NOT during stage=news_actions, where we want it ON)
    if state.get("stage") != "news_actions":
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
        # ✅ LLM snapshot + actions için aynı balanced listeyi state'e koy
        state["news_items_llm"] = selected_raw

        # ✅ evidence_map agents tarafından node_news_fetch'te hazırlandı, burada tekrar üretme
        evidence_map = state.get("evidence_map") or {}
        state["debug_notes"].append(f"EvidenceMap(from_agents): items={len(evidence_map)}")

        # ✅ DEBUG: show what the LLM actually saw (top K items)
        try:
            cnt = Counter([str(it.get("ticker") or "").upper().strip() for it in (selected_raw or [])])
            # remove empty ticker key if any
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
            news_raw=selected_raw,  # ✅ IMPORTANT: balanced subset
            lookback_days=int((_merged_prefs(state).get("lookback_days") or 7)),
            max_items_total=len(selected_raw),  # keep consistent with what we pass
        )
        # ✅ DEBUG: formatter / snapshot quality check (step 1)
        try:
            state["debug_notes"].append(f"NewsSnapshotOut: ok={out.get('ok')} parse={out.get('parse_mode')}")
            st_preview = (str(out.get("snapshot_text") or "")[:220]).replace("\n", " ")
            state["debug_notes"].append(f"NewsSnapshotOut(snapshot_preview_220): {st_preview}")
            rj = out.get("risk_json") if isinstance(out.get("risk_json"), dict) else {}
            bt = rj.get("by_ticker") if isinstance(rj.get("by_ticker"), dict) else {}
            state["debug_notes"].append(f"NewsSnapshotOut(risk_by_ticker_keys): {sorted(list(bt.keys()))}")

            # how many evidence_ids appear in snapshot_text?
            import re
            ids = re.findall(r"\(\[([^\]]+)\]", str(out.get("snapshot_text") or ""))
            state["debug_notes"].append(f"NewsSnapshotOut(evidence_ids_in_snapshot): {len(ids)}")
        except Exception as e:
            state["debug_notes"].append(f"NewsSnapshotOut(debug_failed): {e}")

        state["news_snapshot_text"] = str(out.get("snapshot_text") or "").strip() or None
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
                logs.append("SnapshotCheck: ❌ No '([EVIDENCE_ID] DATE | SOURCE)' tags found.")
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
            f"NewsActions(LLM): generated n={len(actions)} (post-clean)."
        )
        return state

    except Exception as e:
        state["news_actions"] = []
        state["debug_notes"].append(f"NewsActions(LLM): failed -> empty: {e}")
        return state

def node_news_evidence_snapshot(state: PortfolioState) -> PortfolioState:
    state = _init_defaults(state)

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
        out = client.generate_news_snapshot(
            tickers=tickers,
            news_raw=items,
            lookback_days=int((_merged_prefs(state).get("lookback_days") or 7)),
            max_items_total=len(items),
        )

        text = str(out.get("snapshot_text") or "").strip()
        ok = bool(out.get("ok"))
        issues = list(out.get("issues") or [])

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
    # ✅ NEW: split flow based on stage
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

    # ✅ In news_actions stage we do not generate portfolio insights
    if state.get("stage") == "news_actions":
        state["debug_notes"].append("Insight: skipped (stage=news_actions).")
        return state

    use_llm = bool(state.get("use_llm", False))
    if (not use_llm) or (LLMClient is None) or (insight_agent_prepare is None):
        state["debug_notes"].append("Insight: skipped (use_llm disabled or LLMClient/insight_agent_prepare unavailable).")
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
    news_signals = state.get("news_signals")  # ✅ now risk_json

    chosen = str(state.get("objective_key") or "maxsharpe").lower().strip()
    refine_obj = chosen

    try:
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
        out = client.generate_portfolio_insights(
            prompts=prompts,
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
        {"news_actions": "news_actions_generate", "main": "llm_select"},
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

        "insight": None,
        "insight_ok": None,
        "insight_issues": [],
        "insight_raw_text": None,
        "insight_parse_mode": None,
        "base_portfolio_metrics": base_portfolio_metrics,
        "base_portfolio_weights": base_portfolio_weights,
        "base_portfolio_objective": base_portfolio_objective,
    }

    return app.invoke(init, config={"recursion_limit": 200})     