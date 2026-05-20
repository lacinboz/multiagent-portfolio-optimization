
# agents_langgraph.py 
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from evidence_utils import assign_evidence_ids_and_map

import os
import time
import json
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from portfolio_core import run_portfolio_optimization, portfolio_stats, risk_contributions
from portfolio_prediction_core import run_portfolio_optimization_prediction
DATA_DIR = Path("data/processed_yahoo")


# ------------------------------------------------------------
# Small numeric helpers (consistency + safety)
# ------------------------------------------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def validate_portfolio(weights: Dict[str, float], excluded: List[str], *, tag: str = "") -> None:
    tickers = set(map(str, weights.keys()))
    excluded_set = set(map(str, excluded or []))

    present = sorted(excluded_set & tickers)
    print(f"[CHECK{':' + tag if tag else ''}] excluded_present_in_weights={present}")

    w = np.array([float(v) for v in weights.values()], dtype=float)

    s = float(w.sum()) if len(w) else 0.0
    max_w = float(w.max()) if len(w) else 0.0
    active = int((np.abs(w) > 1e-8).sum()) if len(w) else 0

    denom = float(np.sum(w**2)) if len(w) else 0.0
    eff_n = float(1.0 / denom) if denom > 0 else 0.0

    print(
        f"[CHECK{':' + tag if tag else ''}] sum_w={s:.6f} max_w={max_w:.6f} active_assets={active} effective_n={eff_n:.2f}"
    )


def _normalize_return_to_decimal(r: float) -> float:
    """
    Enforces ONE convention across the whole system:
      - returns are decimals (e.g., 0.051 for 5.1%)
    If upstream accidentally provides percent-scale (e.g., 5.1 or 51.0),
    we convert it back to decimal.

    Heuristic:
      - if abs(r) > 1.5 -> treat as percent (e.g., 5.1 means 5.1%, 51 means 51%)
      - else keep as decimal
    """
    r = float(r)
    if abs(r) > 1.5:
        return r / 100.0
    return r


def _normalize_metrics_inplace(port: Dict[str, Any]) -> None:
    """
    Normalizes optimizer output in-place:
      - return -> decimal
      - vol -> decimal (leave as-is unless obviously percent-scale)
      - sharpe -> float if exists
    """
    if not isinstance(port, dict):
        return

    r = _safe_float(port.get("return"))
    if r is not None:
        port["return"] = _normalize_return_to_decimal(r)

    v = _safe_float(port.get("vol"))
    if v is not None:
        port["vol"] = (v / 100.0) if abs(v) > 1.5 else v

    s = _safe_float(port.get("sharpe"))
    if s is not None:
        port["sharpe"] = s


# ------------------------------------------------------------
# Data "agent" helpers
# ------------------------------------------------------------
def data_agent_get_mu_cov(selected_tickers: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    cov = pd.read_csv(DATA_DIR / "cov_annual.csv", index_col=0)

    if "mu_annual" not in summary.columns:
        raise ValueError("summary_per_asset_annual.csv must contain 'mu_annual' column.")

    mu_all = summary["mu_annual"].astype(float)

    cov_index = set(map(str, cov.index))
    cov_cols = set(map(str, cov.columns))
    mu_index = set(map(str, mu_all.index))

    common = [t for t in selected_tickers if (t in mu_index and t in cov_index and t in cov_cols)]
    if len(common) == 0:
        raise ValueError("No common tickers found in mu and cov for the selected universe.")

    mu = mu_all.loc[common].astype(float)
    cov = cov.loc[common, common].astype(float)

    return mu, cov


# ------------------------------------------------------------
# Optimization "agent"
# ------------------------------------------------------------
def optimization_agent(
    selected_tickers: List[str],
    rf: float = 0.02,
    w_max: float = 0.30,
    lambda_l2: float = 1e-3,
) -> Dict[str, Any]:
    mu, cov = data_agent_get_mu_cov(selected_tickers)

    result = run_portfolio_optimization(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
        data_dir=DATA_DIR,
        save_csv=True,
    )

    if isinstance(result, dict):
        for k in ("maxsharpe", "minvar"):
            if k in result and isinstance(result[k], dict):
                _normalize_metrics_inplace(result[k])

    return result


def prediction_constrained_optimization_agent(
    mu: pd.Series,
    cov: pd.DataFrame,
    news_constraints: Dict[str, Dict[str, Any]],
    rf: float = 0.02,
    w_max: float = 0.30,
    lambda_l2: float = 1e-3,
): 
    result = run_portfolio_optimization_prediction(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
        news_constraints=news_constraints,
        data_dir=DATA_DIR,
        save_csv=True,
    )

    if isinstance(result, dict):

        for k in ("maxsharpe", "minvar"):

            if k in result and isinstance(result[k], dict):

                _normalize_metrics_inplace(result[k])

    return result

def optimization_agent_from_mu_cov(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.02,
    w_max: float = 0.30,
    lambda_l2: float = 1e-3,
) -> Dict[str, Any]:
    result = run_portfolio_optimization(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
        data_dir=DATA_DIR,
        save_csv=True,
    )

    if isinstance(result, dict):
        for k in ("maxsharpe", "minvar"):
            if k in result and isinstance(result[k], dict):
                _normalize_metrics_inplace(result[k])

    return result


# ------------------------------------------------------------
# Risk "agent"
# ------------------------------------------------------------
def risk_agent(
    weights: Dict[str, float],
    selected_tickers: List[str],
    rf: float = 0.02,
    *,
    align_to_universe: bool = True,
) -> Dict[str, Any]:
    weights = weights or {}

    if align_to_universe:
        universe = list(map(str, selected_tickers))
        if not universe:
            universe = list(map(str, weights.keys()))
        mu, cov = data_agent_get_mu_cov(universe)
        tickers = list(mu.index)
        w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    else:
        tickers_in_w = [t for t, wv in weights.items() if abs(float(wv)) > 1e-12]
        use_tickers = tickers_in_w or list(selected_tickers)
        mu, cov = data_agent_get_mu_cov(use_tickers)
        tickers = list(mu.index)
        w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)

    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Portfolio has sum=0. Provide at least one positive position.")
    w = w / s

    r, v = portfolio_stats(w, mu, cov)
    rc_abs, rc_pct = risk_contributions(w, cov)

    r = _normalize_return_to_decimal(float(r))
    vol_f = float(v)
    vol_f = (vol_f / 100.0) if abs(vol_f) > 1.5 else vol_f

    sharpe = float((float(r) - float(rf)) / vol_f) if (vol_f is not None and vol_f > 0.0) else None

    weights_out = {t: float(w[i]) for i, t in enumerate(tickers)}
    active_assets = int(sum(1 for x in weights_out.values() if abs(float(x)) > 1e-6))

    max_weight = float(max(weights_out.values())) if weights_out else 0.0
    denom = float(sum((x * x for x in weights_out.values() if x > 0.0)))
    effective_n = float(1.0 / denom) if denom > 0.0 else 0.0

    # Convenience percent fields (for LLM/UI wording)
    return_pct = float(r) * 100.0
    vol_pct = float(vol_f) * 100.0
    max_weight_pct = float(max_weight) * 100.0

    return {
        "tickers": tickers,
        "weights": weights_out,
        "return": float(r),
        "vol": vol_f,
        "sharpe": sharpe,
        "max_weight": max_weight,
        "effective_n": effective_n,
        "active_assets": active_assets,
        "rc_abs": [float(x) for x in rc_abs],
        "rc_pct": [float(x) for x in rc_pct],
        "return_pct": return_pct,
        "vol_pct": vol_pct,
        "max_weight_pct": max_weight_pct,
    }


# ------------------------------------------------------------
# Explanation "agent" (LLM-ready signature)
# ------------------------------------------------------------
def recommendation_agent(
    result: Dict[str, Any],
    objective: str = "max_sharpe",
    current_metrics: Optional[Dict[str, Any]] = None,
    rf: float = 0.02,
    preferences: Optional[Dict[str, Any]] = None,
    *,
    final_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    preferences = preferences or {}

    obj_norm = objective.strip().lower()
    if obj_norm in ("max_sharpe", "maxsharpe", "sharpe"):
        key = "maxsharpe"
        obj_name = "Max Sharpe"
    else:
        key = "minvar"
        obj_name = "Min Variance"

    port = (result or {}).get(key) or {}

    weights_all = port.get("weights", {}) or {}
    weights = {t: float(w) for t, w in weights_all.items() if abs(float(w)) > 1e-6}

    sorted_tickers = sorted(weights.keys(), key=lambda t: weights[t], reverse=True)
    top = sorted_tickers[:3]

    if final_metrics:
        ret = float(final_metrics.get("return", np.nan))
        vol = float(final_metrics.get("vol", np.nan))
        sharpe = final_metrics.get("sharpe", None)
        sharpe = float(sharpe) if sharpe is not None and np.isfinite(float(sharpe)) else None
    else:
        ret = float(port.get("return", np.nan))
        vol = float(port.get("vol", np.nan))
        sharpe = port.get("sharpe", None)
        sharpe = float(sharpe) if sharpe is not None else None

    if np.isfinite(ret):
        ret = _normalize_return_to_decimal(ret)
    if np.isfinite(vol):
        vol = (vol / 100.0) if abs(vol) > 1.5 else vol

    if sharpe is None and np.isfinite(ret) and np.isfinite(vol) and vol > 0:
        sharpe = (ret - rf) / vol

    max_w = max(weights.values()) if len(weights) else 0.0
    eff_n = (1.0 / sum((w**2 for w in weights.values()))) if len(weights) else 0.0

    goal = preferences.get("goal")
    stability = preferences.get("stability")
    concentration = preferences.get("concentration")

    text: List[str] = []
    text.append(f"Objective: **{obj_name}**.")
    if goal:
        text.append(f"Preference (goal): **{goal}**.")
    if stability:
        text.append(f"Preference (stability): **{stability}**.")
    if concentration:
        text.append(f"Preference (concentration): **{concentration}**.")

    text.append(f"The optimized portfolio invests in **{len(weights)}** active assets (non-zero weights).")

    if top:
        top_str = ", ".join([f"{t}: {weights[t]*100:.1f}%" for t in top])
        text.append(f"Top holdings: {top_str}.")

    if np.isfinite(ret) and np.isfinite(vol):
        text.append(f"Optimized expected return: **{ret*100:.1f}%**, volatility: **{vol*100:.1f}%**.")
    elif np.isfinite(ret):
        text.append(f"Optimized expected return: **{ret*100:.1f}%**.")
    elif np.isfinite(vol):
        text.append(f"Volatility: **{vol*100:.1f}%**.")

    if sharpe is not None and np.isfinite(sharpe):
        text.append(f"Optimized Sharpe: **{sharpe:.2f}** (rf={rf:.2%}).")

    text.append(f"Concentration: max weight **{max_w*100:.1f}%**, effective holdings ≈ **{eff_n:.1f}**.")

    if current_metrics is not None:
        r_c = _safe_float(current_metrics.get("return"))
        v_c = _safe_float(current_metrics.get("vol"))
        if r_c is not None:
            r_c = _normalize_return_to_decimal(r_c)
        if v_c is not None:
            v_c = (v_c / 100.0) if abs(v_c) > 1.5 else v_c

        sharpe_c = ((r_c - rf) / v_c) if (r_c is not None and v_c is not None and v_c > 0) else None

        text.append("---")
        if r_c is not None and v_c is not None and np.isfinite(ret) and np.isfinite(vol):
            if sharpe is not None and sharpe_c is not None and np.isfinite(sharpe_c):
                text.append(
                    f"Compared to your current portfolio: return **{r_c*100:.1f}% → {ret*100:.1f}%**, "
                    f"volatility **{v_c*100:.1f}% → {vol*100:.1f}%**, "
                    f"Sharpe **{sharpe_c:.2f} → {sharpe:.2f}**."
                )
            else:
                text.append(
                    f"Compared to your current portfolio: return **{r_c*100:.1f}% → {ret*100:.1f}%**, "
                    f"volatility **{v_c*100:.1f}% → {vol*100:.1f}%**."
                )

            if vol < v_c and ret >= r_c:
                text.append("✅ The optimized portfolio improves **both** risk and return.")
            elif vol < v_c and ret < r_c:
                text.append(
                    "✅ The optimized portfolio reduces risk significantly, trading off some return to improve risk-adjusted performance."
                )
            elif vol >= v_c and ret > r_c:
                text.append("⚠️ The optimized portfolio increases risk to chase higher return (check if this matches your risk tolerance).")
            else:
                text.append("ℹ️ The optimized portfolio is a different trade-off; review the risk contribution chart to understand what changed.")

    return "\n\n".join(text)


# ============================================================
# Insight Generator (LLM agent) — your existing code unchanged
# ============================================================

def _top_k_from_weights(weights: Dict[str, float], k: int = 10) -> List[Dict[str, Any]]:
    items = [(str(t), float(w)) for t, w in (weights or {}).items()]
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[: max(0, int(k))]
    return [{"ticker": t, "weight": w} for t, w in top]


def _top_k_from_rc(metrics: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
    if not metrics:
        return []
    tickers = list(map(str, metrics.get("tickers") or []))
    rc_pct = metrics.get("rc_pct") or []
    if len(tickers) == 0 or len(rc_pct) == 0:
        return []

    pairs = []
    n = min(len(tickers), len(rc_pct))
    for i in range(n):
        t = tickers[i]
        v = _safe_float(rc_pct[i])
        if v is None:
            continue
        pairs.append((t, float(v)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[: max(0, int(k))]
    return [{"ticker": t, "rc_pct": v} for t, v in top]


def _compute_delta(base: Dict[str, Any], refine: Dict[str, Any]) -> Dict[str, Any]:
    def f(d, k):
        return _safe_float((d or {}).get(k))

    delta: Dict[str, Any] = {}
    for k in ("return", "vol", "sharpe", "max_weight", "effective_n"):
        b = f(base, k)
        r = f(refine, k)
        delta[k] = (r - b) if (b is not None and r is not None) else None

    try:
        b_a = int((base or {}).get("active_assets")) if (base or {}).get("active_assets") is not None else None
        r_a = int((refine or {}).get("active_assets")) if (refine or {}).get("active_assets") is not None else None
        delta["active_assets"] = (r_a - b_a) if (b_a is not None and r_a is not None) else None
    except Exception:
        delta["active_assets"] = None

    return delta


def _holdings_change(base_w: Dict[str, float], refine_w: Dict[str, float], threshold: float = 1e-6) -> Dict[str, Any]:
    base_w = base_w or {}
    refine_w = refine_w or {}

    base_active = {t for t, w in base_w.items() if abs(float(w)) > threshold}
    ref_active = {t for t, w in refine_w.items() if abs(float(w)) > threshold}

    entered = sorted(ref_active - base_active)
    exited = sorted(base_active - ref_active)

    common = sorted(base_active & ref_active)
    inc, dec = [], []
    for t in common:
        bw = float(base_w.get(t, 0.0))
        rw = float(refine_w.get(t, 0.0))
        if rw > bw + 1e-9:
            inc.append({"ticker": t, "from": bw, "to": rw})
        elif rw < bw - 1e-9:
            dec.append({"ticker": t, "from": bw, "to": rw})

    inc.sort(key=lambda x: abs(x["to"] - x["from"]), reverse=True)
    dec.sort(key=lambda x: abs(x["to"] - x["from"]), reverse=True)

    return {
        "entered": entered,
        "exited": exited,
        "increased": inc[:10],
        "decreased": dec[:10],
    }


def build_insight_payload(
    *,
    base: Optional[Dict[str, Any]] = None,
    refine: Optional[Dict[str, Any]] = None,
    base_objective: Optional[str] = None,
    refine_objective: Optional[str] = None,
    base_constraints: Optional[Dict[str, Any]] = None,
    refine_constraints: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None,
    news_signals: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    preferences = preferences or {}
    base = base or {}
    refine = refine or {}
    base_constraints = base_constraints or {}
    refine_constraints = refine_constraints or {}

    base_obj = (base_objective or "unknown")
    refine_obj = (refine_objective or "unknown")

    base_w = (base.get("weights") or {}) if isinstance(base, dict) else {}
    ref_w = (refine.get("weights") or {}) if isinstance(refine, dict) else {}

    payload: Dict[str, Any] = {
        "version": "insight_v1",
        "preferences": preferences,
        "news_signals": news_signals or {},
        "base": {
            "objective": base_obj,
            "constraints": base_constraints,
            "metrics": {
                "return": _safe_float(base.get("return")),
                "vol": _safe_float(base.get("vol")),
                "sharpe": _safe_float(base.get("sharpe")),
                "max_weight": _safe_float(base.get("max_weight")),
                "effective_n": _safe_float(base.get("effective_n")),
                "active_assets": base.get("active_assets"),
                "return_pct": _safe_float(base.get("return_pct")),
                "vol_pct": _safe_float(base.get("vol_pct")),
                "max_weight_pct": _safe_float(base.get("max_weight_pct")),
            },
            "top_holdings": _top_k_from_weights(base_w, k=top_k),
            "top_risk_drivers": _top_k_from_rc(base, k=top_k),
        },
        "refine": {
            "objective": refine_obj,
            "constraints": refine_constraints,
            "metrics": {
                "return": _safe_float(refine.get("return")),
                "vol": _safe_float(refine.get("vol")),
                "sharpe": _safe_float(refine.get("sharpe")),
                "max_weight": _safe_float(refine.get("max_weight")),
                "effective_n": _safe_float(refine.get("effective_n")),
                "active_assets": refine.get("active_assets"),
                "return_pct": _safe_float(refine.get("return_pct")),
                "vol_pct": _safe_float(refine.get("vol_pct")),
                "max_weight_pct": _safe_float(refine.get("max_weight_pct")),
            },
            "top_holdings": _top_k_from_weights(ref_w, k=top_k),
            "top_risk_drivers": _top_k_from_rc(refine, k=top_k),
        },
        "delta": {"metrics": {}, "holdings_change": {}},
    }

    if base and refine:
        payload["delta"] = {
            "metrics": _compute_delta(base, refine),
            "holdings_change": _holdings_change(base_w, ref_w),
        }

    print(
        "[INSIGHT:payload] built",
        f"top_k={top_k}",
        f"base_obj={base_obj}",
        f"refine_obj={refine_obj}",
        f"has_delta={'metrics' in payload.get('delta', {})}",
    )
    print("[INSIGHT:payload] base_top_risk_drivers=", [x["ticker"] for x in payload["base"]["top_risk_drivers"][:5]])
    print("[INSIGHT:payload] refine_top_risk_drivers=", [x["ticker"] for x in payload["refine"]["top_risk_drivers"][:5]])

    return payload


def build_insight_prompts(payload: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    payload_json = json.dumps(payload, ensure_ascii=False)

    narrative_system = (
        "You are an Insight Generator for an agent-based portfolio decision product.\n"
        "Output MUST be plain text only (NOT JSON).\n"
        "Do NOT use markdown headings like '#', '##'.\n"
        "Do NOT invent numbers or tickers.\n"
        "Use only the payload.\n"
    )

    narrative_developer = (
        "Write a clear product report for a non-expert user.\n"
        "Requirements:\n"
        "- Write as if explaining to a smart friend who has never invested before. "
        "Use everyday language, and whenever you mention a metric "
        "(return, volatility, Sharpe, max_weight, effective_n), immediately add a short "
        "plain-English 'so what' explaining how it affects the user.\n"
        "- If you mention Max Sharpe, explain in the SAME sentence: "
        "'tries to maximize return per unit of risk'.\n"
        "- If you mention Min Variance, explain in the SAME sentence: "
        "'tries to reduce ups and downs (volatility)'.\n"
        "- If objective changed (base vs refine), explain what that means in practice.\n"
        "- Mention at least THREE exact metrics from payload.\n"
        "- Explain diversification using max_weight and effective_n in simple words.\n"
        "- Include a short 'What changed / what it means' paragraph and a "
        "'Main risk drivers' paragraph.\n"
        "- Risk drivers MUST reference tickers only from payload.base.top_risk_drivers or payload.refine.top_risk_drivers.\n"
        "- Keep it ~10–20 sentences total.\n"
        "- No JSON.\n"
    )

    narrative_user = (
        "Here is the deterministic portfolio payload as JSON.\n"
        "Write the user-facing insight report now.\n\n"
        f"{payload_json}"
    )

    json_system = "Return ONLY valid JSON. No markdown. No extra text."

    json_developer = (
        "Return ONLY valid JSON.\n"
        "The output MUST start with '{' and end with '}'.\n"
        "No markdown. No headings. No extra text.\n\n"
        "You MUST output EXACTLY these 7 top-level keys and NO OTHERS:\n"
        "headline, portfolio_story, risk_drivers, diversification_read, base_vs_refine, news_overlay, action_suggestions_optional.\n\n"
        "Schema:\n"
        "{\n"
        '  "headline": string,\n'
        '  "portfolio_story": [string, ...],\n'
        '  "risk_drivers": [{"ticker": string, "reason": string, "rc_pct": number|null}],\n'
        '  "diversification_read": {"max_weight": number|null, "effective_n": number|null, "comment": string},\n'
        '  "base_vs_refine": {"key_changes": [string, ...], "metric_deltas": object},\n'
        '  "news_overlay": [string, ...],\n'
        '  "action_suggestions_optional": [string, ...]\n'
        "}\n\n"
        "Rules:\n"
        "- ALL 7 keys must be present.\n"
        "- risk_drivers.ticker MUST be from payload.base.top_risk_drivers or payload.refine.top_risk_drivers.\n"
        "- Do NOT invent numbers or tickers.\n"
        "- Do NOT add any extra keys.\n"
    )

    json_user = (
        "Here is the deterministic portfolio payload as JSON.\n"
        "Produce the STRICT JSON object now.\n\n"
        f"{payload_json}"
    )

    print("[INSIGHT:prompt] prepared", f"bytes={len(payload_json)}")
    return {
        "narrative": {"system": narrative_system, "developer": narrative_developer, "user": narrative_user},
        "json": {"system": json_system, "developer": json_developer, "user": json_user},
    }


def verify_insight_output(insight_json: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []

    required_keys = [
        "headline",
        "portfolio_story",
        "risk_drivers",
        "diversification_read",
        "base_vs_refine",
        "news_overlay",
        "action_suggestions_optional",
    ]

    if not isinstance(insight_json, dict):
        return {"ok": False, "issues": ["insight_not_a_dict"], "cleaned": {}}

    cleaned: Dict[str, Any] = {k: insight_json.get(k) for k in required_keys if k in insight_json}
    extra_keys = [k for k in insight_json.keys() if k not in set(required_keys)]
    if extra_keys:
        issues.append(f"extra_top_level_keys_removed: {extra_keys}")

    for k in required_keys:
        if k not in cleaned:
            issues.append(f"missing_key_filled: {k}")
            if k in ("portfolio_story", "news_overlay", "action_suggestions_optional"):
                cleaned[k] = []
            elif k == "risk_drivers":
                cleaned[k] = []
            elif k == "diversification_read":
                cleaned[k] = {"max_weight": None, "effective_n": None, "comment": "not provided"}
            elif k == "base_vs_refine":
                cleaned[k] = {"key_changes": [], "metric_deltas": {}}
            else:
                cleaned[k] = "not provided"

    allowed = set()
    for side in ("base", "refine"):
        for item in ((payload.get(side) or {}).get("top_risk_drivers") or []):
            t = str(item.get("ticker"))
            if t:
                allowed.add(t)

    rd = cleaned.get("risk_drivers")
    if not isinstance(rd, list):
        issues.append("risk_drivers_not_list")
        rd = []
    kept = []
    for item in rd:
        if not isinstance(item, dict):
            issues.append("risk_driver_item_invalid")
            continue
        t = str(item.get("ticker") or "")
        if t and t in allowed:
            kept.append(
                {"ticker": t, "reason": str(item.get("reason") or "not provided"), "rc_pct": _safe_float(item.get("rc_pct"))}
            )
        else:
            issues.append(f"risk_driver_ticker_not_allowed: {t}")
    cleaned["risk_drivers"] = kept

    delta = ((payload.get("delta") or {}).get("metrics") or {})
    bvr = cleaned.get("base_vs_refine")
    if not isinstance(bvr, dict):
        issues.append("base_vs_refine_not_dict")
        bvr = {"key_changes": [], "metric_deltas": {}}

    if "key_changes" not in bvr or not isinstance(bvr.get("key_changes"), list):
        bvr["key_changes"] = []

    bvr["metric_deltas"] = delta
    cleaned["base_vs_refine"] = bvr

    div = cleaned.get("diversification_read")
    if not isinstance(div, dict):
        issues.append("diversification_read_not_dict")
        div = {"max_weight": None, "effective_n": None, "comment": "not provided"}
    div.setdefault("max_weight", None)
    div.setdefault("effective_n", None)
    div.setdefault("comment", "not provided")
    cleaned["diversification_read"] = div

    ok = len([x for x in issues if not x.startswith("missing_key_filled") and not x.startswith("extra_top_level_keys_removed")]) == 0
    print("[INSIGHT:verify]", "ok" if ok else "issues", issues[:5], f"(total={len(issues)})")
    return {"ok": ok, "issues": issues, "cleaned": cleaned}


def insight_agent_prepare(
    *,
    base_metrics: Optional[Dict[str, Any]],
    refine_metrics: Optional[Dict[str, Any]],
    preferences: Optional[Dict[str, Any]] = None,
    news_signals: Optional[Dict[str, Any]] = None,
    base_objective: Optional[str] = None,
    refine_objective: Optional[str] = None,
    base_constraints: Optional[Dict[str, Any]] = None,
    refine_constraints: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    payload = build_insight_payload(
        base=base_metrics,
        refine=refine_metrics,
        base_objective=base_objective,
        refine_objective=refine_objective,
        base_constraints=base_constraints,
        refine_constraints=refine_constraints,
        preferences=preferences,
        news_signals=news_signals,
        top_k=top_k,
    )
    prompts = build_insight_prompts(payload)
    return {"payload": payload, "prompts": prompts}

# ============================================================
# NEWS ACTIONS — deterministic helpers (schema + safety)
# ============================================================
def news_item_to_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Finnhub news item to compact evidence.
    Evidence should be short and stable: headline + date + source (+ optional url).
    """
    ts = item.get("datetime")
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat() if ts else None
    except Exception:
        dt = None

    headline = str(item.get("headline") or "").strip()
    if len(headline) > 180:
        headline = headline[:177] + "..."

    out = {
        "evidence_id": str(item.get("evidence_id") or ""),
        "headline": headline,
        "date": dt,
        "source": str(item.get("source") or "").strip() or None,
    }

    url = str(item.get("url") or "").strip()
    if url:
        out["url"] = url

    return out


_ALLOWED_NEWS_ACTION_TYPES = {
    "exclude_ticker",     # remove ticker from universe
    "set_w_max",      # lower max weight
    "shift_objective",    # to: "minvar" or "maxsharpe"
    "reduce_exposure",    # increase lambda_l2 etc.
    "hedge",              # optional (UI only, may not map to optimizer)
}

def clean_news_actions(
    actions: List[Dict[str, Any]],
    *,
    universe: List[str],
    news_items_by_ticker: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    global_news_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    """
    Deterministic cleaning:
    - keep only allowed action types
    - enforce tickers in universe
    - clamp w_max range
    - drop invalid/unknown fields safely
    """
    issues: List[str] = []
    news_items_by_ticker = news_items_by_ticker or {}
    global_news_items = global_news_items or []

    def _clean_evidence_list(ev: Any) -> List[Dict[str, Any]]:
        out_ev: List[Dict[str, Any]] = []
        if not isinstance(ev, list):
            return out_ev
        for x in ev:
            if not isinstance(x, dict):
                continue
            headline = str(x.get("headline") or "").strip()
            if not headline:
                continue
            if len(headline) > 180:
                headline = headline[:177] + "..."
            eid = str(x.get("evidence_id") or x.get("id") or "").strip() or None
            date = str(x.get("date") or "").strip() or None
            source = str(x.get("source") or "").strip() or None
            e = {"evidence_id": eid,"headline": headline, "date": date, "source": source}
            url = str(x.get("url") or "").strip()
            if url:
                e["url"] = url
            out_ev.append(e)
            if len(out_ev) >= 3:
                break
        return out_ev

    def _fallback_evidence_for_action(action_out: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Prefer ticker-specific evidence
        tkr = str(action_out.get("ticker") or "").upper().strip()
        pool: List[Dict[str, Any]] = []
        if tkr and tkr in news_items_by_ticker:
            pool = news_items_by_ticker.get(tkr) or []
        if not pool:
            pool = global_news_items

        # Most recent first (datetime desc) if available
        try:
            pool_sorted = sorted(pool, key=lambda it: int(it.get("datetime") or 0), reverse=True)
        except Exception:
            pool_sorted = pool

        ev: List[Dict[str, Any]] = []
        for it in pool_sorted[:3]:
            if isinstance(it, dict):
                ev.append(news_item_to_evidence(it))
        return ev

    allowed_tickers = set(str(t).upper().strip() for t in (universe or []))

    cleaned: List[Dict[str, Any]] = []
    for a in (actions or []):
        if not isinstance(a, dict):
            issues.append("action_not_dict")
            continue

        t = str(a.get("type") or "").strip()
        if t not in _ALLOWED_NEWS_ACTION_TYPES:
            issues.append(f"unsupported_action_type:{t}")
            continue

        out = {"type": t, "reason": str(a.get("reason") or "").strip()}
        # evidence from LLM (if present), else filled later via fallback
        out["evidence"] = _clean_evidence_list(a.get("evidence"))


        if t == "exclude_ticker":
            ticker = str(a.get("ticker") or "").upper().strip()
            if not ticker or ticker not in allowed_tickers:
                issues.append(f"exclude_outside_universe:{ticker}")
                continue
            out["ticker"] = ticker


        elif t == "set_w_max":
            try:
                v = float(a.get("value"))
            except Exception:
                issues.append("w_max_not_float")
                continue
            # sensible bounds
            if not (0.05 <= v <= 0.50):
                issues.append(f"w_max_out_of_range:{v}")
                continue
            out["value"] = v

        elif t == "shift_objective":
            to = str(a.get("to") or "").lower().strip()
            if to not in ("minvar", "maxsharpe"):
                issues.append(f"invalid_shift_objective:{to}")
                continue
            out["to"] = to

        elif t == "reduce_exposure":
            # optional ticker
            ticker = str(a.get("ticker") or "").upper().strip()
            if ticker:
                if ticker not in allowed_tickers:
                    issues.append(f"reduce_exposure_outside_universe:{ticker}")
                    continue
                out["ticker"] = ticker


            # optional intensity
            intensity = str(a.get("intensity") or "medium").lower().strip()
            if intensity not in ("low", "medium", "high"):
                intensity = "medium"
            out["intensity"] = intensity


        elif t == "hedge":
            # hedge may have instrument/hint but doesn't map to optimizer necessarily
            out["hedge_hint"] = str(a.get("hedge_hint") or "").strip()
        if not out.get("evidence"):
            out["evidence"] = _fallback_evidence_for_action(out)
            if not out["evidence"]:
                issues.append(f"missing_evidence:{out['type']}")


        cleaned.append(out)
    print("[NEWS:ACTIONS] sample_evidence_ids=", [
    (a.get("type"), [e.get("evidence_id") for e in (a.get("evidence") or [])])
    for a in cleaned[:3]
    ])


    return {"ok": (len(issues) == 0), "issues": issues, "actions": cleaned}


def apply_news_actions_to_params(
    *,
    selected_tickers: List[str],
    w_max: float,
    lambda_l2: float,
    objective_key: str,
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    universe = [str(t).upper().strip() for t in (selected_tickers or [])]
    w_max_new = float(w_max)
    lambda_new = float(lambda_l2)
    obj_new = str(objective_key or "maxsharpe").lower().strip()

    for a in (actions or []):
        t = a.get("type")
        if t == "exclude_ticker":
            ticker = str(a.get("ticker") or "").upper().strip()
            universe = [x for x in universe if x != ticker]

        elif t == "set_w_max":
            w_max_new = min(w_max_new, float(a["value"]))

        elif t == "shift_objective":
            obj_new = str(a.get("to") or obj_new).lower().strip()

        elif t == "reduce_exposure":
            intensity = str(a.get("intensity") or "medium").lower().strip()
            mult = 2.0 if intensity == "low" else (3.0 if intensity == "medium" else 5.0)
            lambda_new = lambda_new * mult

    return {
        "selected_tickers": universe,
        "w_max": w_max_new,
        "lambda_l2": lambda_new,
        "objective_key": obj_new,
    }


# ============================================================
# NEWS — Finnhub primary + fallback
# =========================================================

_FINNHUB_BASE = "https://finnhub.io/api/v1"

# ✅ NEW: persistent disk cache (survives Streamlit reruns / restarts)
_NEWS_CACHE_DIR = Path("data/news_cache")
_NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Simple in-memory cache:
#   key: (endpoint, frozenset(params.items()))
#   val: {"ts": epoch_seconds, "data": json}
_NEWS_CACHE: Dict[Any, Dict[str, Any]] = {}


def _utc_date_str(days_ago: int = 0) -> str:
    d = (datetime.now(timezone.utc) - timedelta(days=days_ago)).date()
    return d.isoformat()


def _cache_get(key: Any, ttl_s: int) -> Optional[Any]:
    hit = _NEWS_CACHE.get(key)
    if not hit:
        return None
    if (time.time() - float(hit["ts"])) > float(ttl_s):
        return None
    return hit["data"]


def _cache_set(key: Any, data: Any) -> None:
    _NEWS_CACHE[key] = {"ts": time.time(), "data": data}


def _stable_params_for_cache(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove secrets (token) and make params stable for hashing."""
    q = dict(params or {})
    q.pop("token", None)
    return q


def _disk_cache_path(endpoint: str, params_no_token: Dict[str, Any]) -> Path:
    payload = {"endpoint": endpoint, "params": params_no_token}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()
    safe_ep = endpoint.strip("/").replace("/", "_") or "root"
    return _NEWS_CACHE_DIR / f"finnhub_{safe_ep}_{h}.json"


def _disk_cache_get(endpoint: str, params_no_token: Dict[str, Any], ttl_s: int) -> Optional[Any]:
    p = _disk_cache_path(endpoint, params_no_token)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = float(obj.get("_fetched_at", 0.0))
        if ts <= 0:
            return None
        if (time.time() - ts) > float(ttl_s):
            return None
        return obj.get("data")
    except Exception:
        return None

import hashlib


def _disk_cache_set(endpoint: str, params_no_token: Dict[str, Any], data: Any) -> None:
    p = _disk_cache_path(endpoint, params_no_token)
    try:
        obj = {"_fetched_at": time.time(), "data": data}
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # cache write failures should never break the app
        return
def _news_sample_from_payload(data: Any) -> Dict[str, Any]:
    """
    Returns a compact sample proof from Finnhub payload.
    Works for list payloads (company-news, news).
    """
    if not isinstance(data, list) or len(data) == 0:
        return {"items": 0}

    first = data[0] if isinstance(data[0], dict) else None
    if not isinstance(first, dict):
        return {"items": len(data), "first_is_dict": False}

    return {
        "items": len(data),
        "id": first.get("id"),
        "datetime": first.get("datetime"),
        "source": first.get("source"),
        "headline": (str(first.get("headline") or "")[:140]),
        "url": first.get("url"),
    }


def _print_fetch_proof(tag: str, endpoint: str, url: str, extra: Dict[str, Any]) -> None:
    # Single-line-ish proof, easy to spot in logs
    try:
        print(f"[NEWS:PROOF] {tag} endpoint={endpoint} url={url} meta={json.dumps(extra, ensure_ascii=False)}")
    except Exception:
        print(f"[NEWS:PROOF] {tag} endpoint={endpoint} url={url} meta={extra}")


def _finnhub_get(endpoint: str, params: Dict[str, Any], *, timeout_s: int = 30, cache_ttl_s: int = 600) -> Any:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing FINNHUB_API_KEY in environment (or .env loaded).")

    url = f"{_FINNHUB_BASE}{endpoint}"
    q = dict(params or {})
    q["token"] = api_key

    # ✅ Use cache keys WITHOUT token for disk (so we don't store secrets)
    q_no_token = _stable_params_for_cache(q)

    # 1) ✅ disk cache (persistent)
    disk_hit = _disk_cache_get(endpoint, q_no_token, ttl_s=cache_ttl_s)
    if disk_hit is not None:
        p = _disk_cache_path(endpoint, q_no_token)
        print(f"[NEWS:CACHE] DISK HIT endpoint={endpoint} ttl={cache_ttl_s}s file={p.name}")

        # ✅ PROOF (cache)
        sample = _news_sample_from_payload(disk_hit)
        _print_fetch_proof("DISK_HIT", endpoint, url, {"cache_file": p.name, **sample})

        return disk_hit
    else:
        print(f"[NEWS:CACHE] DISK MISS endpoint={endpoint}")

    # 2) memory cache (fast within same process)
    mem_key = (endpoint, tuple(sorted(q_no_token.items())))
    mem_hit = _cache_get(mem_key, ttl_s=cache_ttl_s)
    if mem_hit is not None:
        print(f"[NEWS:CACHE] MEM HIT endpoint={endpoint} ttl={cache_ttl_s}s")

        # ✅ PROOF (cache)
        sample = _news_sample_from_payload(mem_hit)
        _print_fetch_proof("MEM_HIT", endpoint, url, sample)

        return mem_hit
    else:
        print(f"[NEWS:CACHE] MEM MISS endpoint={endpoint}")

    # 3) live request
    print(f"[NEWS:CACHE] LIVE REQUEST endpoint={endpoint} url={url}")

    t0 = time.time()
    r = requests.get(url, params=q, timeout=timeout_s)
    dt_ms = int((time.time() - t0) * 1000)

    # ✅ PROOF (HTTP layer)
    print(f"[NEWS:HTTP] endpoint={endpoint} status={r.status_code} elapsed_ms={dt_ms} bytes={len(r.content)}")

    if r.status_code != 200:
        # keep existing behavior
        raise RuntimeError(f"Finnhub HTTP {r.status_code} on {endpoint}: {r.text[:300]}")

    data = r.json()

    # ✅ PROOF (payload layer)
    sample = _news_sample_from_payload(data)
    _print_fetch_proof("LIVE_200", endpoint, url, {"elapsed_ms": dt_ms, **sample})

    # write caches
    _cache_set(mem_key, data)
    _disk_cache_set(endpoint, q_no_token, data)

    print(f"[NEWS:CACHE] SAVED endpoint={endpoint} (disk+mem)")
    return data


def _fmt_news_date_from_datetime(dt_epoch: Any) -> str:
    try:
        if dt_epoch is None:
            return "unknown"
        return datetime.fromtimestamp(int(dt_epoch), tz=timezone.utc).date().isoformat()
    except Exception:
        return "unknown"




def _dedup_news_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by url, else by (headline, source, datetime). Keep order."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items or []:
        url = str(it.get("url") or "").strip()
        if url:
            k = ("url", url)
        else:
            k = (
                "h",
                str(it.get("headline") or "").strip(),
                str(it.get("source") or "").strip(),
                int(it.get("datetime") or 0),
            )
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _limit_items(items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    max_items = int(max_items)
    if max_items <= 0:
        return []
    return (items or [])[:max_items]


def fetch_company_news_for_ticker(
    ticker: str,
    *,
    lookback_days: int = 7,
    max_items: int = 40,
    cache_ttl_s: int = 600,
    sleep_s: float = 0.25,
) -> List[Dict[str, Any]]:
    """Primary source: /company-news?symbol=...&from=...&to=..."""
    ticker = str(ticker).upper().strip()
    frm = _utc_date_str(days_ago=int(lookback_days))
    to = _utc_date_str(days_ago=0)

    data = _finnhub_get(
        "/company-news",
        {"symbol": ticker, "from": frm, "to": to},
        cache_ttl_s=cache_ttl_s,
    )
    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))

    if not isinstance(data, list):
        return []

    items = []
    for d in data:
        if not isinstance(d, dict):
            continue
        dt_epoch = int(d.get("datetime") or 0)
        date_str = _fmt_news_date_from_datetime(dt_epoch)

        source = str(d.get("source") or "").strip()
        headline = str(d.get("headline") or "").strip()
        url = str(d.get("url") or "").strip()

        items.append(
            {
                "id": d.get("id"),
                "ticker": ticker,                 # ✅ önemli: LLM tarafında filtreleme kolaylaşır
                "provider": "finnhub_company",     # ✅ opsiyonel ama debug için çok iyi
                "date": date_str,                 # ✅ epoch yerine UI/LLM için hazır alan
                "datetime": dt_epoch,
                "source": source,
                "headline": headline,
                "summary": str(d.get("summary") or ""),
                "url": url,
                "related": str(d.get("related") or ""),
                "category": str(d.get("category") or ""),
            }
)

        

    items = _dedup_news_items(items)
    items = _limit_items(items, max_items=max_items)
    return items

def fetch_company_news_for_ticker_window(
    ticker: str,
    *,
    from_date: str,
    to_date: str,
    max_items: int = 40,
    cache_ttl_s: int = 86400,
    sleep_s: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Historical company news fetch for a fixed date window.
    Uses the same Finnhub endpoint, cache, limit, and sleep logic as current news.
    """
    ticker = str(ticker).upper().strip()

    data = _finnhub_get(
        "/company-news",
        {"symbol": ticker, "from": from_date, "to": to_date},
        cache_ttl_s=cache_ttl_s,
    )

    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))

    if not isinstance(data, list):
        return []

    items = []
    for d in data:
        if not isinstance(d, dict):
            continue

        dt_epoch = int(d.get("datetime") or 0)
        date_str = _fmt_news_date_from_datetime(dt_epoch)

        items.append(
            {
                "id": d.get("id"),
                "ticker": ticker,
                "provider": "finnhub_company_historical",
                "date": date_str,
                "datetime": dt_epoch,
                "source": str(d.get("source") or "").strip(),
                "headline": str(d.get("headline") or "").strip(),
                "summary": str(d.get("summary") or ""),
                "url": str(d.get("url") or "").strip(),
                "related": str(d.get("related") or ""),
                "category": str(d.get("category") or ""),
            }
        )

    items = _dedup_news_items(items)
    items = _limit_items(items, max_items=max_items)
    return items
def fetch_market_news(
    *,
    category: str = "general",
    max_items: int = 100,
    cache_ttl_s: int = 600,
    sleep_s: float = 0.25,
) -> List[Dict[str, Any]]:
    """Fallback pool: /news?category=general"""
    category = str(category).strip().lower()
    data = _finnhub_get("/news", {"category": category, "minId": 0}, cache_ttl_s=cache_ttl_s)
    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))

    if not isinstance(data, list):
        return []

    items = []
    for d in data:
        if not isinstance(d, dict):
            continue
        dt_epoch = int(d.get("datetime") or 0)
        date_str = _fmt_news_date_from_datetime(dt_epoch)

        source = str(d.get("source") or "").strip()
        headline = str(d.get("headline") or "").strip()
        url = str(d.get("url") or "").strip()

        # market news'te ticker yok -> burada boş bırakacağız, sonra filtreleyen fonksiyon ekliyor/taşıyor
        items.append(
            {
                "id": d.get("id"),
                "ticker": "MARKET",
                "provider": "finnhub_market",
                "date": date_str,
                "datetime": dt_epoch,
                "source": source,
                "headline": headline,
                "summary": str(d.get("summary") or ""),
                "url": url,
                "related": str(d.get("related") or ""),
                "category": str(d.get("category") or ""),
            }
        )

    items = _dedup_news_items(items)
    items = _limit_items(items, max_items=max_items)
    return items


def _filter_market_news_for_ticker(market_items: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
    """Simple keyword filter: keep item if ticker appears in headline/summary/related (case-insensitive)."""
    t = str(ticker).upper().strip()
    out: List[Dict[str, Any]] = []
    for it in market_items or []:
        text = f"{it.get('headline','')} {it.get('summary','')} {it.get('related','')}".upper()
        if t and (t in text):
            it2 = dict(it)
            it2["ticker"] = t
            out.append(it2)   # ✅ BUNU EKLE

    return _dedup_news_items(out)



def news_agent_fetch_for_tickers(
    tickers: List[str],
    *,
    include_news: bool = True,  # ✅ hard guard
    lookback_days: int = 7,
    min_company_items: int = 1,
    max_items_per_ticker: int = 20,
    include_market_fallback: bool = True,
    market_category: str = "general",
    cache_ttl_s: int = 600,
    sleep_s: float = 0.25,
) -> Dict[str, Any]:
    """
    Deterministic news fetch layer (non-LLM):
      - If include_news is False -> returns immediately with empty content.
      - Otherwise:
          for each ticker:
            try company news
            if < min_company_items and include_market_fallback:
              use market news (filtered) as fallback
    """
    tickers = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))  # unique keep order

    if not include_news:
        return {
            "lookback_days": int(lookback_days),
            "sources": {t: "skipped" for t in tickers},
            "items_by_ticker": {t: [] for t in tickers},
            "evidence_map": {},
            "stats": {
                "tickers": len(tickers),
                "total_items": 0,
                "company_used": 0,
                "fallback_used": 0,
                "errors": {},
                "skipped": True,
            },
        }

    items_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    market_pool: List[Dict[str, Any]] = []
    if include_market_fallback:
        try:
            market_pool = fetch_market_news(
                category=market_category,
                max_items=200,
                cache_ttl_s=cache_ttl_s,
                sleep_s=sleep_s,
            )
        except Exception as e:
            errors["market_news"] = str(e)
            market_pool = []

    for t in tickers:
        try:
            comp = fetch_company_news_for_ticker(
                t,
                lookback_days=lookback_days,
                max_items=max_items_per_ticker,
                cache_ttl_s=cache_ttl_s,
                sleep_s=sleep_s,
            )
            if len(comp) >= int(min_company_items):
                items_by_ticker[t] = comp
                sources[t] = "company"
                continue

            if include_market_fallback and market_pool:
                fb = _filter_market_news_for_ticker(market_pool, t)
                fb = _limit_items(fb, max_items=max_items_per_ticker)
                if fb:
                    items_by_ticker[t] = fb
                    sources[t] = "market_fallback"
                else:
                    items_by_ticker[t] = []
                    sources[t] = "none"
            else:
                items_by_ticker[t] = []
                sources[t] = "none"

        except Exception as e:
            errors[t] = str(e)
            items_by_ticker[t] = []
            sources[t] = "error"

    total_items = sum(len(v) for v in items_by_ticker.values())
    used_company = sum(1 for s in sources.values() if s == "company")
    used_fb = sum(1 for s in sources.values() if s == "market_fallback")

    print(
        f"[NEWS] fetched tickers={len(tickers)} total_items={total_items} "
        f"company={used_company} fallback={used_fb} lookback_days={lookback_days}"
    )
    # ✅ Portfolio ile aynı evidence_id üretimi (tek kaynak)
    flat: List[Dict[str, Any]] = []
    for t in tickers:
        for it in (items_by_ticker.get(t) or []):
            cp = dict(it)
            cp["ticker"] = str(cp.get("ticker") or t).upper().strip() or "UNK"
            flat.append(cp)

    flat_with_ids, evidence_map = assign_evidence_ids_and_map(flat)

    # ✅ Tekrar ticker bazlı grupla (order korunur)
    new_items_by_ticker: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tickers}
    for it in flat_with_ids:
        t = str(it.get("ticker") or "").upper().strip() or "UNK"
        if t not in new_items_by_ticker:
            new_items_by_ticker[t] = []
        new_items_by_ticker[t].append(it)

    items_by_ticker = new_items_by_ticker
    # ✅ evidence_id -> full item lookup (dashboard en rahat bunu kullanır)
    news_items_by_id = {
        str(it.get("evidence_id")): it
        for it in (flat_with_ids or [])
        if str(it.get("evidence_id") or "").strip()
    }

    return {
        "lookback_days": int(lookback_days),
        "sources": sources,
        "items_by_ticker": items_by_ticker,
        "evidence_map": evidence_map,  
        "flat_items": flat_with_ids,
        "items_by_id": news_items_by_id,
        "stats": {
            "tickers": len(tickers),
            "total_items": total_items,
            "company_used": used_company,
            "fallback_used": used_fb,
            "errors": errors,
            "skipped": False,
        },
    }

def historical_news_agent_fetch_for_tickers(
    tickers: List[str],
    *,
    include_news: bool = True,
    lookback_days: int = 365,
    exclude_recent_days: int = 14,
    max_items_per_ticker: int = 20,
    cache_ttl_s: int = 86400,
    sleep_s: float = 0.25,
) -> Dict[str, Any]:
    """
    Historical news fetch for predictive evaluation only.

    It uses the same Finnhub company-news endpoint and the same safety controls
    as the normal news fetcher, but requests an older date window:

        from = today - lookback_days
        to   = today - exclude_recent_days

    This avoids evaluating today's news where future prices are unavailable.
    """
    tickers = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))

    if not include_news:
        return {
            "lookback_days": int(lookback_days),
            "exclude_recent_days": int(exclude_recent_days),
            "items_by_ticker": {t: [] for t in tickers},
            "flat_items": [],
            "evidence_map": {},
            "items_by_id": {},
            "stats": {
                "tickers": len(tickers),
                "total_items": 0,
                "errors": {},
                "skipped": True,
            },
        }
    to_days_ago = int(exclude_recent_days)
    from_days_ago = int(exclude_recent_days) + int(lookback_days)

    from_date = _utc_date_str(days_ago=from_days_ago)
    to_date = _utc_date_str(days_ago=to_days_ago)

    items_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    errors: Dict[str, str] = {}

    for t in tickers:
        try:
            items = fetch_company_news_for_ticker_window(
                t,
                from_date=from_date,
                to_date=to_date,
                max_items=max_items_per_ticker,
                cache_ttl_s=cache_ttl_s,
                sleep_s=sleep_s,
            )
            items_by_ticker[t] = items
        except Exception as e:
            errors[t] = str(e)
            items_by_ticker[t] = []

    flat: List[Dict[str, Any]] = []
    for t in tickers:
        for it in items_by_ticker.get(t, []):
            cp = dict(it)
            cp["ticker"] = str(cp.get("ticker") or t).upper().strip()
            flat.append(cp)

    flat_with_ids, evidence_map = assign_evidence_ids_and_map(flat)

    new_items_by_ticker: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tickers}
    for it in flat_with_ids:
        t = str(it.get("ticker") or "").upper().strip()
        if t not in new_items_by_ticker:
            new_items_by_ticker[t] = []
        new_items_by_ticker[t].append(it)

    news_items_by_id = {
        str(it.get("evidence_id")): it
        for it in flat_with_ids
        if str(it.get("evidence_id") or "").strip()
    }

    return {
        "lookback_days": int(lookback_days),
        "exclude_recent_days": int(exclude_recent_days),
        "from_date": from_date,
        "to_date": to_date,
        "items_by_ticker": new_items_by_ticker,
        "flat_items": flat_with_ids,
        "evidence_map": evidence_map,
        "items_by_id": news_items_by_id,
        "stats": {
            "tickers": len(tickers),
            "total_items": len(flat_with_ids),
            "errors": errors,
            "skipped": False,
        },
    }