import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# --- Dosya yolları ---
DATA_DIR = Path("data/processed")
RETURNS_CSV = DATA_DIR / "returns_hourly.csv"               
COV_ANNUAL_CSV = DATA_DIR / "cov_annual.csv"
SUMMARY_ANNUAL_CSV = DATA_DIR / "summary_per_asset_annual.csv"


# --- Yardımcı fonksiyonlar (genel) ---

def near_psd(A, eps=1e-8):
    """Negatif en küçük özdeğerleri 0'a çekerek PSD kovaryans matrisi üretir."""
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, a_min=eps, a_max=None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)

def portfolio_stats(w, mu, cov):
    """Verilen ağırlıklar için beklenen getiri ve volatilite (yıllık)."""
    w = np.asarray(w)
    r = float(w @ mu.values)                        
    v = float(np.sqrt(w @ cov.values @ w))         
    return r, v

def risk_contributions(w, cov):
    """Her varlığın toplam portföy volatilitesine katkısı."""
    w = np.asarray(w)
    sigma_p = np.sqrt(w @ cov.values @ w)
    mrc = (cov.values @ w) / sigma_p          # marginal risk contribution
    rc = w * mrc                              # absolute risk contribution
    return rc, rc / rc.sum()                  # absolute, percentage

def sharpe_ratio(w, mu, cov, rf=0.0):
    """Sharpe oranı = (getiri - rf) / vola."""
    r, v = portfolio_stats(w, mu, cov)
    return (r - rf) / v if v > 0 else -np.inf


# --- ASIL ÖNEMLİ KISIM: Optimizasyon fonksiyonu ---

def run_portfolio_optimization(mu, cov, rf=0.02, w_max=0.30, lambda_l2=1e-3,
                               data_dir=DATA_DIR, save_csv=True):
    """
    mu: pd.Series  -> yıllık beklenen getiriler (seçili hisseler için)
    cov: pd.DataFrame -> yıllık kovaryans matrisi (aynı hisseler için)
    rf: float -> yıllık risksiz faiz oranı
    w_max: float -> tek hisse için maksimum ağırlık
    lambda_l2: float -> L2 cezası (daha dengeli dağılım için küçük bir ceza)
    """

    # hizalama garantisi (mu indexleri ile cov index/sütunları)
    tickers = list(mu.index)
    cov = cov.loc[tickers, tickers].astype(float)
    mu = mu.astype(float)

    n = len(tickers)
    print("Tickers used:", tickers)

    # --- Kovaryans PSD mi kontrol et ---
    eigvals = np.linalg.eigvalsh(cov.values)
    print("Smallest eigenvalue:", eigvals.min())

    if eigvals.min() < 0:
        print("Covariance is not PSD; applying near PSD.")
        cov_np = near_psd(cov.values)
        cov = pd.DataFrame(cov_np, index=tickers, columns=tickers)

    # --- Optimizasyon ayarları ---
    bounds = [(0.0, w_max)] * n
    w0 = np.full(n, 1/n)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    # --- Hedef fonksiyonlar ---

    def obj_min_var(w):
        """Varyans + L2 cezası (daha dengeli dağılım için)."""
        return (w @ cov.values @ w) + lambda_l2 * np.sum(w**2)

    def obj_neg_sharpe(w):
        return -sharpe_ratio(w, mu, cov, rf=rf)

    # --- Min-Var portföyü ---
    res_minvar = minimize(obj_min_var, w0,
                          method="SLSQP", bounds=bounds, constraints=constraints)

    w_minvar = pd.Series(res_minvar.x, index=tickers)
    r_minvar, v_minvar = portfolio_stats(w_minvar.values, mu, cov)
    rc_abs, rc_pct = risk_contributions(w_minvar.values, cov)

    print("Min-Var success:", res_minvar.success, "| message:", res_minvar.message)
    print("Return (annually):", r_minvar, " | Vol (annually):", v_minvar)
    print("Weights:\n", w_minvar.round(4))

    # --- Max-Sharpe portföyü ---
    res_maxsharpe = minimize(obj_neg_sharpe, w0,
                             method="SLSQP", bounds=bounds, constraints=constraints)

    w_maxsharpe = pd.Series(res_maxsharpe.x, index=tickers)
    r_ms, v_ms = portfolio_stats(w_maxsharpe.values, mu, cov)
    sharpe_ms = (r_ms - rf) / v_ms

    print("Max-Sharpe success:", res_maxsharpe.success, "| message:", res_maxsharpe.message)
    print("Sharpe:", sharpe_ms, " | Return:", r_ms, " | Vol:", v_ms)
    print("Weights:\n", w_maxsharpe.round(4))

    # --- Efficient frontier için yardımcı fonksiyon ---

    def min_var_for_target_return(target):
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, mu=mu, t=target: w @ mu.values - t}
        ]
        res = minimize(obj_min_var, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        return res

    grid = np.linspace(mu.min()*0.8, mu.max()*1.2, 25)

    frontier_rows = []
    for t in grid:
        res = min_var_for_target_return(t)
        if res.success:
            w = res.x
            r, v = portfolio_stats(w, mu, cov)
            row = {
                "target_return": t,
                "realized_return": r,
                "vol": v
            }
            for i, k in enumerate(tickers):
                row[f"w_{k}"] = w[i]
            frontier_rows.append(row)

    frontier = pd.DataFrame(frontier_rows)

    print("Efficient frontier point count:", len(frontier))

    
    if save_csv:
        out_minvar = pd.DataFrame({
            "ticker": tickers,
            "weight": w_minvar.values,
            "rc_abs": rc_abs,
            "rc_pct": rc_pct
        })
        out_minvar.to_csv(data_dir / "portfolio_minvar.csv", index=False)

        out_maxsharpe = pd.DataFrame({
            "ticker": tickers,
            "weight": w_maxsharpe.values
        })
        out_maxsharpe.to_csv(data_dir / "portfolio_maxsharpe.csv", index=False)

        frontier.to_csv(data_dir / "efficient_frontier.csv", index=False)

        print("Saved:",
            (data_dir / "portfolio_minvar.csv").as_posix(),
            (data_dir / "portfolio_maxsharpe.csv").as_posix(),
            (data_dir / "efficient_frontier.csv").as_posix())

   
    result = {
        "tickers": tickers,
        "rf": rf,
        "w_max": w_max,
        "lambda_l2": lambda_l2,
        "minvar": {
            "success": bool(res_minvar.success),
            "return": float(r_minvar),
            "vol": float(v_minvar),
            "weights": {t: float(w_minvar[t]) for t in tickers},
            "rc_pct": {t: float(rc_pct[i]) for i, t in enumerate(tickers)}
        },
        "maxsharpe": {
            "success": bool(res_maxsharpe.success),
            "return": float(r_ms),
            "vol": float(v_ms),
            "sharpe": float(sharpe_ms),
            "weights": {t: float(w_maxsharpe[t]) for t in tickers}
        },
        "frontier": [
            {
                "target_return": float(row["target_return"]),
                "realized_return": float(row["realized_return"]),
                "vol": float(row["vol"])
            }
            for _, row in frontier.iterrows()
        ]
    }

    return result
