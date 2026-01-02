# agents_langgraph.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

    denom = float(np.sum(w ** 2)) if len(w) else 0.0
    eff_n = float(1.0 / denom) if denom > 0 else 0.0

    print(f"[CHECK{':' + tag if tag else ''}] sum_w={s:.6f} max_w={max_w:.6f} active_assets={active} effective_n={eff_n:.2f}")

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
                text.append("✅ The optimized portfolio reduces risk significantly, trading off some return to improve risk-adjusted performance.")
            elif vol >= v_c and ret > r_c:
                text.append("⚠️ The optimized portfolio increases risk to chase higher return (check if this matches your risk tolerance).")
            else:
                text.append("ℹ️ The optimized portfolio is a different trade-off; review the risk contribution chart to understand what changed.")

    return "\n\n".join(text)
