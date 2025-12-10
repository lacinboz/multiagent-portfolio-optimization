import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


DATA_DIR = Path("data/processed")
RETURNS_CSV = DATA_DIR / "returns_hourly.csv"               
COV_ANNUAL_CSV = DATA_DIR / "cov_annual.csv"
SUMMARY_ANNUAL_CSV = DATA_DIR / "summary_per_asset_annual.csv"

# reading files 
cov = pd.read_csv(COV_ANNUAL_CSV, index_col=0)
summary = pd.read_csv(SUMMARY_ANNUAL_CSV, index_col=0)

# creating summary
mu = summary["mu_annual"].copy()
tickers_mu = mu.index.tolist()
tickers_cov = cov.index.tolist()

# selecting common tickers
common = [t for t in tickers_mu if t in tickers_cov]
mu = mu.loc[common].astype(float)
cov = cov.loc[common, common].astype(float)

assert cov.shape[0] == cov.shape[1] == mu.shape[0], "Shape inconsistency is present"

tickers = mu.index.tolist()
n = len(tickers)
print("Tickers used:", tickers)

# getting the smallest eigenvalue
eigvals = np.linalg.eigvalsh(cov.values)
print("Smallest eigenvalue:", eigvals.min())


def near_psd(A, eps=1e-8):
    """Negative smallest eigenvalues make it 0 to create PSD matrix."""
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, a_min=eps, a_max=None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)

if eigvals.min() < 0:
    print("Covariance is not PSD; applying near PSD.")
    cov_np = near_psd(cov.values)
    cov = pd.DataFrame(cov_np, index=tickers, columns=tickers)


def portfolio_stats(w, mu, cov):
    w = np.asarray(w)
    r = float(w @ mu.values)                        
    v = float(np.sqrt(w @ cov.values @ w))         
    return r, v

# calculating risk contributions: 
def risk_contributions(w, cov):
    """ Contirbution for every ticker w * (Σ w) / σ_p """
    w = np.asarray(w)
    sigma_p = np.sqrt(w @ cov.values @ w)
    mrc = (cov.values @ w) / sigma_p              
    rc = w * mrc                                  
    return rc, rc / rc.sum()  
                  
# calculating sharpe ratio: 

def sharpe_ratio(w, mu, cov, rf=0.0):
    r, v = portfolio_stats(w, mu, cov)
    return (r - rf) / v if v > 0 else -np.inf


rf = 0.02              
w_max = 0.30        
bounds = [(0.0, w_max)] * n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
w0 = np.full(n, 1/n)
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

lambda_l2 = 1e-3  

def obj_min_var(w, cov):
    """Variance + L2 penalty (For better distribution)."""
    return (w @ cov.values @ w) + lambda_l2 * np.sum(w**2)


res_minvar = minimize(obj_min_var, w0, args=(cov,),
                      method="SLSQP", bounds=bounds, constraints=constraints)

w_minvar = pd.Series(res_minvar.x, index=tickers)
r_minvar, v_minvar = portfolio_stats(w_minvar.values, mu, cov)
rc_abs, rc_pct = risk_contributions(w_minvar.values, cov)

print("Min-Var success:", res_minvar.success, "| mesaj:", res_minvar.message)
print("Return (annually):", r_minvar, " | Vol (annually):", v_minvar)
print(":Weights\n", w_minvar.round(4))

rf = 0.02
def obj_neg_sharpe(w, mu, cov, rf):
    return -sharpe_ratio(w, mu, cov, rf=rf)

res_maxsharpe = minimize(obj_neg_sharpe, w0, args=(mu, cov, rf),
                         method="SLSQP", bounds=bounds, constraints=constraints)

w_maxsharpe = pd.Series(res_maxsharpe.x, index=tickers)
r_ms, v_ms = portfolio_stats(w_maxsharpe.values, mu, cov)
print("Max-Sharpe Success:", res_maxsharpe.success, "| message:", res_maxsharpe.message)
print("Sharpe:", (r_ms - rf)/v_ms, " | Return:", r_ms, " | Vol:", v_ms)
print("Weights:\n", w_maxsharpe.round(4))


def min_var_for_target_return(target, mu, cov):
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w, mu=mu, t=target: w @ mu.values - t}
    ]
    res = minimize(obj_min_var, w0, args=(cov,), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res

grid = np.linspace(mu.min()*0.8, mu.max()*1.2, 25)

frontier_rows = []
for t in grid:
    res = min_var_for_target_return(t, mu, cov)
    if res.success:
        w = res.x
        r, v = portfolio_stats(w, mu, cov)
        frontier_rows.append({"target_return": t, "realized_return": r, "vol": v, **{f"w_{k}": w[i] for i,k in enumerate(tickers)}})

frontier = pd.DataFrame(frontier_rows)
frontier.to_csv(DATA_DIR / "efficient_frontier.csv", index=False)

print("Efficient frontier point count:", len(frontier))
frontier.head()

out_minvar = pd.DataFrame({
    "ticker": tickers,
    "weight": w_minvar.values,
    "rc_abs": rc_abs,
    "rc_pct": rc_pct
})
out_minvar.to_csv(DATA_DIR / "portfolio_minvar.csv", index=False)

out_maxsharpe = pd.DataFrame({
    "ticker": tickers,
    "weight": w_maxsharpe.values
})
out_maxsharpe.to_csv(DATA_DIR / "portfolio_maxsharpe.csv", index=False)

print("Saved:",
      (DATA_DIR / "portfolio_minvar.csv").as_posix(),
      (DATA_DIR / "portfolio_maxsharpe.csv").as_posix(),
      (DATA_DIR / "efficient_frontier.csv").as_posix())
