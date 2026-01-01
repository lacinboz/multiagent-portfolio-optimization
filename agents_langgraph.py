# agents_langgraph.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio_core import run_portfolio_optimization, portfolio_stats, risk_contributions

DATA_DIR = Path("data/processed_yahoo")


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
    return result


# ------------------------------------------------------------
# Risk "agent"
# ------------------------------------------------------------
def risk_agent(weights: Dict[str, float], selected_tickers: List[str]) -> Dict[str, Any]:
    """
    Computes portfolio-level return/vol and risk contributions for given weights.
    Normalizes weights to sum=1 and clips negatives.

    Returns:
      {
        "tickers": [... aligned tickers ...],
        "weights": {ticker: weight},
        "return": float,
        "vol": float,
        "rc_abs": [float],
        "rc_pct": [float],
      }
    """
    mu, cov = data_agent_get_mu_cov(selected_tickers)
    tickers = list(mu.index)

    w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    w = np.clip(w, 0.0, None)

    s = float(w.sum())
    if s <= 0:
        raise ValueError("Portfolio has sum=0. Provide at least one positive position.")

    w = w / s

    r, v = portfolio_stats(w, mu, cov)
    rc_abs, rc_pct = risk_contributions(w, cov)

    return {
        "tickers": tickers,
        "weights": {t: float(w[i]) for i, t in enumerate(tickers)},
        "return": float(r),
        "vol": float(v),
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
) -> str:
    """
    Produces user-facing explanation. (Still rule-based for now.)
    This signature is "LangGraph-friendly":
      - accepts preferences (optional) so graph can pass the state without hacks.
      - objective supports "max_sharpe" or "min_var" (dashboard style),
        and we map it to your result keys: "maxsharpe" / "minvar".
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

    port = result[key]

    weights_all = port["weights"]
    weights = {t: float(w) for t, w in weights_all.items() if abs(float(w)) > 1e-6}

    sorted_tickers = sorted(weights.keys(), key=lambda t: weights[t], reverse=True)
    top = sorted_tickers[:3]

    ret = float(port["return"])
    vol = float(port["vol"])

    sharpe = port.get("sharpe", None)
    sharpe = float(sharpe) if sharpe is not None else None

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

    text.append(f"Optimized expected return: **{ret*100:.1f}%**, volatility: **{vol*100:.1f}%**.")
    if sharpe is not None and np.isfinite(sharpe):
        text.append(f"Optimized Sharpe: **{sharpe:.2f}** (rf={rf:.2%}).")

    text.append(f"Concentration: max weight **{max_w*100:.1f}%**, effective holdings ≈ **{eff_n:.1f}**.")

    # Compare vs current if available
    if current_metrics is not None:
        r_c = float(current_metrics["return"])
        v_c = float(current_metrics["vol"])
        sharpe_c = (r_c - rf) / v_c if v_c > 0 else np.nan

        text.append("---")
        if sharpe is not None and np.isfinite(sharpe_c):
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
