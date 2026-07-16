import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

DATA_DIR = Path("data/processed_yahoo")
RETURNS_CSV = DATA_DIR / "returns_daily.csv"               
COV_ANNUAL_CSV = DATA_DIR / "cov_annual.csv"
SUMMARY_ANNUAL_CSV = DATA_DIR / "summary_per_asset_annual.csv"




def near_psd(A, eps=1e-8):
    """Produces a PSD covariance matrix by clipping negative eigenvalues to a small positive threshold."""
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, a_min=eps, a_max=None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)

def portfolio_stats(w, mu, cov):
    """Computes expected return and volatility for the given weights on an annualized basis."""
    w = np.asarray(w)
    r = float(w @ mu.values)                        
    v = float(np.sqrt(w @ cov.values @ w))         
    return r, v

def risk_contributions(w, cov):
    """Computes each asset's contribution to total portfolio volatility."""
    w = np.asarray(w)
    sigma_p = np.sqrt(w @ cov.values @ w)
    mrc = (cov.values @ w) / sigma_p          # marginal risk contribution
    rc = w * mrc                              # absolute risk contribution
    return rc, rc / rc.sum()                  # absolute, percentage

def sharpe_ratio(w, mu, cov, rf=0.0):
    """Sharpe ratio = (return - risk-free rate) / volatility."""
    r, v = portfolio_stats(w, mu, cov)
    return (r - rf) / v if v > 0 else -np.inf

def run_portfolio_optimization(mu, cov, rf=0.02, w_max=0.30, lambda_l2=1e-3,
                               data_dir=DATA_DIR, save_csv=True):
    """
    mu: pd.Series -> annual expected returns for the selected assets
    cov: pd.DataFrame -> annual covariance matrix for the same assets
    rf: float -> annual risk-free rate
    w_max: float -> maximum weight per asset
    lambda_l2: float -> L2 penalty for encouraging a more diversified allocation
    """

    # Ensure alignment between the mu index and the covariance matrix rows/columns
    tickers = list(mu.index)
    cov = cov.loc[tickers, tickers].astype(float)
    mu = mu.astype(float)

    n = len(tickers)
    print("Tickers used:", tickers)

    # --- Feasibility guard for max weight constraint ---
    min_feasible_wmax = (1.0 / n) if n > 0 else 1.0
    effective_w_max = float(w_max)

    if n * effective_w_max < 1.0:
        effective_w_max = min_feasible_wmax + 1e-6
        print(
            f"[AUTO-ADJUST] Infeasible cap detected: "
            f"n={n}, requested w_max={w_max:.4f}, "
            f"minimum feasible={min_feasible_wmax:.4f}. "
            f"Using effective_w_max={effective_w_max:.4f}."
        )
    print(f"Requested w_max: {w_max:.4f} | Effective w_max: {effective_w_max:.4f}")

    # --- Check whether the covariance matrix is PSD ---
    eigvals = np.linalg.eigvalsh(cov.values)
    print("Smallest eigenvalue:", eigvals.min())

    if eigvals.min() < 0:
        print("Covariance is not PSD; applying near PSD.")
        cov_np = near_psd(cov.values)
        cov = pd.DataFrame(cov_np, index=tickers, columns=tickers)

   # --- Optimization settings ---
    bounds = [(0.0, effective_w_max)] * n
    w0 = np.full(n, 1/n)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

   # --- Objective functions ---

    def obj_min_var(w):
        """Variance plus an L2 penalty to encourage a more diversified allocation."""
        return (w @ cov.values @ w) + lambda_l2 * np.sum(w**2)

    def obj_neg_sharpe(w):
        return -sharpe_ratio(w, mu, cov, rf=rf)

    # --- Minimum-variance portfolio ---
    res_minvar = minimize(obj_min_var, w0,
                          method="SLSQP", bounds=bounds, constraints=constraints)

    w_minvar = pd.Series(res_minvar.x, index=tickers)
    r_minvar, v_minvar = portfolio_stats(w_minvar.values, mu, cov)
    rc_abs, rc_pct = risk_contributions(w_minvar.values, cov)

    print("Min-Var success:", res_minvar.success, "| message:", res_minvar.message)
    print("Return (annually):", r_minvar, " | Vol (annually):", v_minvar)
    print("Weights:\n", w_minvar.round(4))

    # --- Maximum-Sharpe portfolio ---
    res_maxsharpe = minimize(obj_neg_sharpe, w0,
                             method="SLSQP", bounds=bounds, constraints=constraints)

    w_maxsharpe = pd.Series(res_maxsharpe.x, index=tickers)
    r_ms, v_ms = portfolio_stats(w_maxsharpe.values, mu, cov)
    sharpe_ms = (r_ms - rf) / v_ms

    print("Max-Sharpe success:", res_maxsharpe.success, "| message:", res_maxsharpe.message)
    print("Sharpe:", sharpe_ms, " | Return:", r_ms, " | Vol:", v_ms)
    print("Weights:\n", w_maxsharpe.round(4))

    # --- Helper function for the efficient frontier ---

      

    def min_var_for_target_return(target):
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, t=target: float(w @ mu.values) - float(t)}
        ]
        res = minimize(obj_min_var, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        return res

    # Find the feasible return range: [r_minvar, r_max_return]
    def obj_neg_return(w):
        return -float(w @ mu.values)

    res_maxret = minimize(
        obj_neg_return,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if res_maxret.success:
        r_maxret = float(res_maxret.x @ mu.values)
    else:
        # Fallback case, rarely needed
        r_maxret = float(mu.max())
    print("[FRONTIER DEBUG]")
    print("n_assets:", n)
    print("effective_w_max:", effective_w_max)
    print("lambda_l2:", lambda_l2)
    print("r_minvar:", r_minvar)
    print("r_maxret:", r_maxret)

    print("minvar weights")
    print(w_minvar.round(4))

    print("maxret weights")
    print(pd.Series(res_maxret.x, index=tickers).round(4))

    lo = float(r_minvar)
    hi = float(r_maxret)
    if hi < lo:
        lo, hi = hi, lo

    # Use a denser grid to obtain a smoother curve
    grid = np.linspace(lo, hi, 60)

    frontier_rows = []
    for t in grid:
        res = min_var_for_target_return(t)
        if res.success:
            w = res.x
            r, v = portfolio_stats(w, mu, cov)
            row = {
                "target_return": float(t),
                "realized_return": float(r),
                "vol": float(v),
            }
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
        "effective_w_max": effective_w_max,
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