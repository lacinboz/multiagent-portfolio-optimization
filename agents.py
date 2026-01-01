import numpy as np
import pandas as pd
from pathlib import Path

from portfolio_core import run_portfolio_optimization, portfolio_stats, risk_contributions

DATA_DIR = Path("data/processed_yahoo")


def data_agent_get_mu_cov(selected_tickers):
    """
    selected_tickers: list[str], e.g. ["AAPL", "MSFT", "NVDA"]
    returns: (mu, cov) annualized, sadece bu hisseler için

    Not:
    - mu: pd.Series (index=tickers)
    - cov: pd.DataFrame (index/cols=tickers)
    """
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    cov = pd.read_csv(DATA_DIR / "cov_annual.csv", index_col=0)

    # expected return series
    if "mu_annual" not in summary.columns:
        raise ValueError("summary_per_asset_annual.csv must contain 'mu_annual' column.")

    mu_all = summary["mu_annual"].astype(float)

    # cov alignment: ticker must exist in BOTH index and columns
    cov_index = set(map(str, cov.index))
    cov_cols = set(map(str, cov.columns))
    mu_index = set(map(str, mu_all.index))

    common = [t for t in selected_tickers if (t in mu_index and t in cov_index and t in cov_cols)]

    if len(common) == 0:
        raise ValueError("No common tickers found in mu and cov for the selected universe.")

    mu = mu_all.loc[common].astype(float)
    cov = cov.loc[common, common].astype(float)

    return mu, cov


def optimization_agent(selected_tickers, rf=0.02, w_max=0.30, lambda_l2=1e-3):
    """
    selected_tickers: list[str]
    returns: optimization result dict from portfolio_core.run_portfolio_optimization
    """
    mu, cov = data_agent_get_mu_cov(selected_tickers)

    result = run_portfolio_optimization(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
        data_dir=DATA_DIR,
        save_csv=True,  # dashboard için CSV’ler de oluşsun istersen
    )
    return result


def risk_agent(weights, selected_tickers):
    """
    weights: dict, e.g. {"AAPL": 0.4, "MSFT": 0.6}
    selected_tickers: list, order matters (["AAPL","MSFT"])

    returns:
    {
      tickers: [... aligned tickers ...],
      return: float,
      vol: float,
      rc_abs: list[float],
      rc_pct: list[float],
      weights: dict[ticker->float]   (normalized)
    }
    """
    mu, cov = data_agent_get_mu_cov(selected_tickers)

    tickers = list(mu.index)  # aligned tickers (may be subset of selected_tickers)

    # Build w safely (missing -> 0)
    w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)

    # No negatives
    w = np.clip(w, 0.0, None)

    s = float(w.sum())
    if s <= 0:
        raise ValueError("Current portfolio has sum=0. Please provide at least one positive position.")

    # Normalize (double safety)
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


def recommendation_agent(result, objective="max_sharpe", current_metrics=None, rf=0.02):
    if objective == "max_sharpe":
        port = result["maxsharpe"]
        obj_name = "Max Sharpe"
    else:
        port = result["minvar"]
        obj_name = "Min Variance"

    weights_all = port["weights"]
    weights = {t: float(w) for t, w in weights_all.items() if abs(float(w)) > 1e-6}

    # top holdings
    sorted_tickers = sorted(weights.keys(), key=lambda t: weights[t], reverse=True)
    top = sorted_tickers[:3]

    ret = float(port["return"])
    vol = float(port["vol"])
    sharpe = port.get("sharpe", None)
    sharpe = float(sharpe) if sharpe is not None else None

    # concentration + constraints signal
    max_w = max(weights.values()) if len(weights) else 0.0
    eff_n = (1.0 / sum((w**2 for w in weights.values()))) if len(weights) else 0.0  # effective number of holdings

    text = []
    text.append(f"Objective: **{obj_name}**.")
    text.append(f"The optimized portfolio invests in **{len(weights)}** active assets (non-zero weights).")

    if top:
        top_str = ", ".join([f"{t}: {weights[t]*100:.1f}%" for t in top])
        text.append(f"Top holdings: {top_str}.")

    text.append(f"Optimized expected return: **{ret*100:.1f}%**, volatility: **{vol*100:.1f}%**.")
    if sharpe is not None and np.isfinite(sharpe):
        text.append(f"Optimized Sharpe: **{sharpe:.2f}** (rf={rf:.2%}).")

    text.append(f"Concentration: max weight **{max_w*100:.1f}%**, effective holdings ≈ **{eff_n:.1f}**.")

    # ✅ Compare vs Current if available
    if current_metrics is not None:
        r_c = float(current_metrics["return"])
        v_c = float(current_metrics["vol"])
        sharpe_c = (r_c - rf) / v_c if v_c > 0 else np.nan

        text.append("---")
        text.append(
            f"Compared to your current portfolio: return **{r_c*100:.1f}% → {ret*100:.1f}%**, "
            f"volatility **{v_c*100:.1f}% → {vol*100:.1f}%**, "
            f"Sharpe **{sharpe_c:.2f} → {sharpe:.2f}**." if (sharpe is not None and np.isfinite(sharpe_c))
            else
            f"Compared to your current portfolio: return **{r_c*100:.1f}% → {ret*100:.1f}%**, "
            f"volatility **{v_c*100:.1f}% → {vol*100:.1f}%**."
        )

        # rule-based verdict
        if vol < v_c and ret >= r_c:
            text.append("✅ The optimized portfolio improves **both** risk and return.")
        elif vol < v_c and ret < r_c:
            text.append("✅ The optimized portfolio reduces risk significantly, trading off some return to improve risk-adjusted performance.")
        elif vol >= v_c and ret > r_c:
            text.append("⚠️ The optimized portfolio increases risk to chase higher return (check if this matches your risk tolerance).")
        else:
            text.append("ℹ️ The optimized portfolio is a different trade-off; review the risk contribution chart to understand what changed.")

    return "\n\n".join(text)
