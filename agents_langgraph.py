# agents_langgraph.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json  

from portfolio_core import run_portfolio_optimization, portfolio_stats, risk_contributions

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

    Heuristic (practical + safe for typical equity data):
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
        # Vol is almost always a decimal (0.18 = 18%).
        # But if someone accidentally stored 18.0, bring it back.
        port["vol"] = (v / 100.0) if abs(v) > 1.5 else v

    s = _safe_float(port.get("sharpe"))
    if s is not None:
        port["sharpe"] = s


# ------------------------------------------------------------
# Data "agent" helpers
# ------------------------------------------------------------
def data_agent_get_mu_cov(selected_tickers: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Loads annualized expected returns (mu) and annualized covariance (cov) for selected tickers.

    Returns:
      mu: pd.Series indexed by tickers
      cov: pd.DataFrame with index/cols tickers
    """
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    cov = pd.read_csv(DATA_DIR / "cov_annual.csv", index_col=0)

    if "mu_annual" not in summary.columns:
        raise ValueError("summary_per_asset_annual.csv must contain 'mu_annual' column.")

    mu_all = summary["mu_annual"].astype(float)

    cov_index = set(map(str, cov.index))
    cov_cols = set(map(str, cov.columns))
    mu_index = set(map(str, mu_all.index))

    # Preserve caller order (important for stable alignment)
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
    """
    Runs numerical portfolio optimization.

    Returns:
      result dict from portfolio_core.run_portfolio_optimization
    """
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

    # ✅ Normalize optimizer outputs so UI/LLM never see mixed scales
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
    """
    Same optimization but skips re-loading mu/cov from disk (faster + cleaner for LangGraph).
    """
    result = run_portfolio_optimization(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
        data_dir=DATA_DIR,
        save_csv=True,
    )

    # ✅ Normalize optimizer outputs so UI/LLM never see mixed scales
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
    """
    Computes portfolio-level return/vol, Sharpe, concentration stats, and risk contributions.
    Normalizes weights to sum=1 and clips negatives.

    ✅ Important change:
    - Default align_to_universe=True: compute metrics on the SAME ticker universe order
      (selected_tickers), filling missing weights with 0. This makes candidate comparisons
      more consistent (maxsharpe vs minvar) and reduces subtle metric mismatch.

    Returns:
      {
        "tickers": [... aligned tickers ...],
        "weights": {ticker: weight},
        "return": float (decimal),
        "vol": float (decimal),
        "sharpe": float | None,
        "max_weight": float,
        "effective_n": float,
        "active_assets": int,
        "rc_abs": [float],
        "rc_pct": [float],
      }
    """
    weights = weights or {}

    # Choose alignment universe
    if align_to_universe:
        # Use the provided universe (stable comparisons)
        universe = list(map(str, selected_tickers))
        if not universe:
            # fallback to weights keys if user passed empty universe
            universe = list(map(str, weights.keys()))
        mu, cov = data_agent_get_mu_cov(universe)
        tickers = list(mu.index)

        w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)

    else:
        # Old behavior: only tickers present in weights (subset)
        tickers_in_w = [t for t, wv in weights.items() if abs(float(wv)) > 1e-12]
        use_tickers = tickers_in_w or list(selected_tickers)
        mu, cov = data_agent_get_mu_cov(use_tickers)
        tickers = list(mu.index)
        w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)

    # Long-only normalization (your design)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Portfolio has sum=0. Provide at least one positive position.")
    w = w / s

    r, v = portfolio_stats(w, mu, cov)
    rc_abs, rc_pct = risk_contributions(w, cov)

    # ✅ Enforce metric units (decimals)
    r = _normalize_return_to_decimal(float(r))
    vol_f = float(v)
    vol_f = (vol_f / 100.0) if abs(vol_f) > 1.5 else vol_f

    sharpe = float((float(r) - float(rf)) / vol_f) if (vol_f is not None and vol_f > 0.0) else None

    weights_out = {t: float(w[i]) for i, t in enumerate(tickers)}
    active_assets = int(sum(1 for x in weights_out.values() if abs(float(x)) > 1e-6))

    max_weight = float(max(weights_out.values())) if weights_out else 0.0
    denom = float(sum((x * x for x in weights_out.values() if x > 0.0)))
    effective_n = float(1.0 / denom) if denom > 0.0 else 0.0

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
    """
    Produces user-facing explanation.

    ✅ Key change:
    - If final_metrics is provided (from risk_agent), we use it for return/vol/sharpe
      to avoid scale mismatch between optimizer outputs and risk_agent outputs.

    objective supports "max_sharpe" or "min_var" and maps to "maxsharpe"/"minvar".
    """
    preferences = preferences or {}

    # --- map objective string ---
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

    # ✅ Prefer final_metrics (risk_agent output) for consistency
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

    # ✅ normalize units just in case
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

    # Compare vs current if available
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

            # verdict
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
# Insight Generator (LLM agent) — ADDITIVE ONLY
# Two-call design:
#   1) Narrative TEXT for UI (no JSON)
#   2) Optional small JSON for UI signals (parsed + verified)
# Keeps verify_insight_output (strengthened)
# ============================================================

def _top_k_from_weights(weights: Dict[str, float], k: int = 10) -> List[Dict[str, Any]]:
    """Returns top-k holdings sorted by weight desc."""
    items = [(str(t), float(w)) for t, w in (weights or {}).items()]
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[: max(0, int(k))]
    return [{"ticker": t, "weight": w} for t, w in top]


def _top_k_from_rc(metrics: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
    """
    Builds top-k risk contributors from risk_agent output:
      metrics["tickers"], metrics["rc_pct"]
    """
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
    """
    Computes refine - base deltas for key metrics, robust to missing values.
    Expects outputs of risk_agent (return/vol/sharpe/max_weight/effective_n/active_assets).
    """
    def f(d, k):
        return _safe_float((d or {}).get(k))

    delta: Dict[str, Any] = {}
    for k in ("return", "vol", "sharpe", "max_weight", "effective_n"):
        b = f(base, k)
        r = f(refine, k)
        delta[k] = (r - b) if (b is not None and r is not None) else None

    # active_assets is int
    try:
        b_a = int((base or {}).get("active_assets")) if (base or {}).get("active_assets") is not None else None
        r_a = int((refine or {}).get("active_assets")) if (refine or {}).get("active_assets") is not None else None
        delta["active_assets"] = (r_a - b_a) if (b_a is not None and r_a is not None) else None
    except Exception:
        delta["active_assets"] = None

    return delta


def _holdings_change(base_w: Dict[str, float], refine_w: Dict[str, float], threshold: float = 1e-6) -> Dict[str, Any]:
    """
    Summarizes holding changes between base and refine.
    - entered: tickers with weight>thr in refine but <=thr in base
    - exited: tickers with weight>thr in base but <=thr in refine
    - increased/decreased: among common active tickers, compare weights
    """
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


# ✅ ONLY REQUIRED FIX:
# - Add *_pct fields into build_insight_payload() so the Insight LLM
#   sees percentages (51.0%) instead of decimals (0.51) and stops writing "0.51%".
# Everything else is unchanged.

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
    """
    Creates a deterministic input package for the Insight Generator LLM.
    """
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
                # decimals (still useful for math / consistency)
                "return": _safe_float(base.get("return")),
                "vol": _safe_float(base.get("vol")),
                "sharpe": _safe_float(base.get("sharpe")),
                "max_weight": _safe_float(base.get("max_weight")),
                "effective_n": _safe_float(base.get("effective_n")),
                "active_assets": base.get("active_assets"),
                # ✅ add percent fields for correct UI wording in Insight LLM
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
                # decimals
                "return": _safe_float(refine.get("return")),
                "vol": _safe_float(refine.get("vol")),
                "sharpe": _safe_float(refine.get("sharpe")),
                "max_weight": _safe_float(refine.get("max_weight")),
                "effective_n": _safe_float(refine.get("effective_n")),
                "active_assets": refine.get("active_assets"),
                # ✅ add percent fields for correct UI wording in Insight LLM
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
    """
    Returns TWO prompt packs:
      - prompts["narrative"]  -> long product-style report (PLAIN TEXT) for UI
      - prompts["json"]       -> strict JSON for UI signals (optional)
    """
    payload_json = json.dumps(payload, ensure_ascii=False)

    # ----------------------------
    # 1) Narrative (TEXT) for UI
    # ----------------------------
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
    "plain-English 'so what' explaining how it affects the user "
    "(e.g. 'more ups and downs', 'more stable month-to-month', "
    "'one stock can hurt you more').\n"

    "- If you mention Max Sharpe, explain in the SAME sentence: "
    "'tries to maximize return per unit of risk'.\n"
    "- If you mention Min Variance, explain in the SAME sentence: "
    "'tries to reduce ups and downs (volatility)'.\n"
    "- If objective changed (base vs refine), explain what that means in practice.\n"
    "- Mention at least THREE exact metrics from payload "
    "(return, vol, sharpe, max_weight, effective_n, active_assets).\n"
    "- Explain diversification using max_weight and effective_n in simple words.\n"
    "- Include a short 'What changed / what it means' paragraph and a "
    "'Main risk drivers' paragraph.\n"
    "- Risk drivers MUST reference tickers only from "
    "payload.base.top_risk_drivers or payload.refine.top_risk_drivers.\n"
    "- Keep it ~10–20 sentences total.\n"
    "- No JSON. No bullet formatting required (you can use short lines).\n"
)


    narrative_user = (
        "Here is the deterministic portfolio payload as JSON.\n"
        "Write the user-facing insight report now.\n\n"
        f"{payload_json}"
    )

    # ----------------------------
    # 2) Strict JSON (optional)
    # ----------------------------
    json_system = "Return ONLY valid JSON. No markdown. No extra text."

    # Keep your original 7-key schema (but make it extremely strict).
    # This is optional for UI signals; you can hide it.
    json_developer = (
        "Return ONLY valid JSON.\n"
        "The output MUST start with '{' and end with '}'.\n"
        "No markdown. No headings. No extra text.\n\n"
        "You MUST output EXACTLY these 7 top-level keys and NO OTHERS:\n"
        "headline, portfolio_story, risk_drivers, diversification_read, base_vs_refine, news_overlay, action_suggestions_optional.\n\n"
        "Schema:\n"
        "{\n"
        '  \"headline\": string,\n'
        '  \"portfolio_story\": [string, ...],\n'
        '  \"risk_drivers\": [{\"ticker\": string, \"reason\": string, \"rc_pct\": number|null}],\n'
        '  \"diversification_read\": {\"max_weight\": number|null, \"effective_n\": number|null, \"comment\": string},\n'
        '  \"base_vs_refine\": {\"key_changes\": [string, ...], \"metric_deltas\": object},\n'
        '  \"news_overlay\": [string, ...],\n'
        '  \"action_suggestions_optional\": [string, ...]\n'
        "}\n\n"
        "Rules:\n"
        "- ALL 7 keys must be present.\n"
        "- Mention at least THREE exact metrics from payload inside strings.\n"
        "- risk_drivers.ticker MUST be from payload.base.top_risk_drivers or payload.refine.top_risk_drivers.\n"
        "- Do NOT invent numbers or tickers.\n"
        "- Do NOT add any extra keys (no global, no metrics_changes, etc.).\n"
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
    """
    Strengthened deterministic verifier for the JSON insight output.
    Keeps your function name. More robust for 7B outputs:
      - drops extra keys
      - ensures all required keys exist
      - enforces risk_driver ticker whitelist
      - injects metric_deltas from payload.delta.metrics (source of truth)
    """
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

    # If model didn't return dict, fail cleanly
    if not isinstance(insight_json, dict):
        return {"ok": False, "issues": ["insight_not_a_dict"], "cleaned": {}}

    # Drop extra top-level keys (THIS fixes the "global", "metrics_changes" etc.)
    cleaned: Dict[str, Any] = {k: insight_json.get(k) for k in required_keys if k in insight_json}
    extra_keys = [k for k in insight_json.keys() if k not in set(required_keys)]
    if extra_keys:
        issues.append(f"extra_top_level_keys_removed: {extra_keys}")

    # Fill missing keys
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

    # Allowed tickers for risk_drivers
    allowed = set()
    for side in ("base", "refine"):
        for item in ((payload.get(side) or {}).get("top_risk_drivers") or []):
            t = str(item.get("ticker"))
            if t:
                allowed.add(t)

    # Sanitize risk_drivers list
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
            # Ensure keys exist
            kept.append(
                {
                    "ticker": t,
                    "reason": str(item.get("reason") or "not provided"),
                    "rc_pct": _safe_float(item.get("rc_pct")),
                }
            )
        else:
            issues.append(f"risk_driver_ticker_not_allowed: {t}")
    cleaned["risk_drivers"] = kept

    # Force metric_deltas from payload (source of truth)
    delta = ((payload.get("delta") or {}).get("metrics") or {})
    bvr = cleaned.get("base_vs_refine")
    if not isinstance(bvr, dict):
        issues.append("base_vs_refine_not_dict")
        bvr = {"key_changes": [], "metric_deltas": {}}

    if "key_changes" not in bvr or not isinstance(bvr.get("key_changes"), list):
        bvr["key_changes"] = []

    bvr["metric_deltas"] = delta
    cleaned["base_vs_refine"] = bvr

    # Diversification read: keep dict shape
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
    """
    Prepares payload + TWO prompt packs.
    """
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

