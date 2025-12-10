import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

def data_agent_get_mu_cov(selected_tickers):
    """
    selected_tickers: list of str, e.g. ["AAPL", "MSFT", "NVDA"]
    returns: (mu, cov) annualized, sadece bu hisseler için
    """
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    cov = pd.read_csv(DATA_DIR / "cov_annual.csv", index_col=0)

    mu = summary["mu_annual"].copy()

    # sadece ortak olanları al
    common = [t for t in selected_tickers if t in mu.index and t in cov.index]
    mu = mu.loc[common].astype(float)
    cov = cov.loc[common, common].astype(float)

    return mu, cov


from portfolio_core import run_portfolio_optimization

def optimization_agent(selected_tickers, rf=0.02, w_max=0.30, lambda_l2=1e-3):
    """
    selected_tickers: list of str
    """

    mu, cov = data_agent_get_mu_cov(selected_tickers)

    result = run_portfolio_optimization(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
        data_dir=Path("data/processed"),
        save_csv=True,   # dashboard için CSV’ler de oluşsun istersen
    )
    return result


from portfolio_core import portfolio_stats, risk_contributions

def risk_agent(weights, selected_tickers):
    """
    weights: dict, e.g. {"AAPL": 0.4, "MSFT": 0.6}
    selected_tickers: list, order matters (["AAPL","MSFT"])
    """

    mu, cov = data_agent_get_mu_cov(selected_tickers)
    w = np.array([weights[t] for t in selected_tickers])

    r, v = portfolio_stats(w, mu, cov)
    rc_abs, rc_pct = risk_contributions(w, cov)

    return {
        "tickers": selected_tickers,
        "return": float(r),
        "vol": float(v),
        "rc_abs": rc_abs.tolist(),
        "rc_pct": rc_pct.tolist(),
    }

def recommendation_agent(result, objective="max_sharpe"):
    if objective == "max_sharpe":
        port = result["maxsharpe"]
    else:
        port = result["minvar"]

    weights = port["weights"]
    tickers = list(weights.keys())

    # 🔹 ağırlığa göre büyükten küçüğe sırala
    sorted_tickers = sorted(tickers, key=lambda t: weights[t], reverse=True)
    top = sorted_tickers[:3]

    ret = port["return"]
    vol = port["vol"]
    sharpe = port.get("sharpe", None)

    explanation = f"The optimized portfolio invests in {len(tickers)} stocks. "
    explanation += "The largest weights are: "
    explanation += ", ".join([f"{t}: {weights[t]*100:.1f}%" for t in top])
    explanation += f". The expected annual return is about {ret*100:.1f}% "
    explanation += f"with a volatility of {vol*100:.1f}%."
    if sharpe is not None:
        explanation += f" This corresponds to a Sharpe ratio of {sharpe:.2f}."

    return explanation



