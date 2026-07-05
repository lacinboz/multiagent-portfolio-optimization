"""
Figure: Efficient Frontier (101-ticker universe)
✅ No look-ahead bias: uses training-only mu/cov
   (data/processed_yahoo/summary_per_asset_annual.csv)
   (data/processed_yahoo/cov_annual.csv)
Run this from your project root.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# ── paths ──────────────────────────────────────────────────
MU_PATH  = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH = Path("data/processed_yahoo/cov_annual.csv")

RF       = 0.02
W_MAX    = 0.30
LAMBDA   = 1e-3
N_POINTS = 60


# ── helpers ─────────────────────────────────────────────────
def _near_psd(A, eps=1e-8):
    vals, vecs = np.linalg.eigh(A)
    return vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T


def load_mu_cov():
    summary = pd.read_csv(MU_PATH, index_col=0)
    cov_df  = pd.read_csv(COV_PATH, index_col=0)
    common  = [t for t in summary.index if t in cov_df.index]
    mu  = summary.loc[common, "mu_annual"].astype(float)
    cov = cov_df.loc[common, common].astype(float)
    return mu, cov


def max_sharpe(mu, cov_np, rf=RF, w_max=W_MAX):
    n   = len(mu)
    eff = max(w_max, 1/n + 1e-6)
    w0  = np.full(n, 1/n)

    def neg_sharpe(w):
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov_np @ w))
        return -(r - rf) / v if v > 0 else np.inf

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=[(0, eff)]*n,
                   constraints=[{"type": "eq",
                                  "fun": lambda w: w.sum()-1}])
    w = np.clip(res.x, 0, None); w /= w.sum()
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov_np @ w))
    return {"weights": w, "return": r, "vol": v,
            "sharpe": (r-rf)/v if v > 0 else 0}


def min_variance(mu, cov_np, w_max=W_MAX):
    n   = len(mu)
    eff = max(w_max, 1/n + 1e-6)
    w0  = np.full(n, 1/n)

    def port_var(w):
        return float(w @ cov_np @ w)

    res = minimize(port_var, w0, method="SLSQP",
                   bounds=[(0, eff)]*n,
                   constraints=[{"type": "eq",
                                  "fun": lambda w: w.sum()-1}])
    w = np.clip(res.x, 0, None); w /= w.sum()
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov_np @ w))
    return {"weights": w, "return": r, "vol": v}


def efficient_frontier(mu, cov_np, n_points=N_POINTS,
                        w_max=W_MAX):
    n   = len(mu)
    eff = max(w_max, 1/n + 1e-6)
    mv  = min_variance(mu, cov_np, w_max)
    ms  = max_sharpe(mu, cov_np, w_max=w_max)
    r_min = mv["return"]
    r_max = ms["return"] * 1.05
    targets = np.linspace(r_min, r_max, n_points)

    points = []
    for target in targets:
        def port_var(w): return float(w @ cov_np @ w)
        res = minimize(port_var, np.full(n, 1/n),
                       method="SLSQP",
                       bounds=[(0, eff)]*n,
                       constraints=[
                           {"type": "eq",
                            "fun": lambda w: w.sum()-1},
                           {"type": "eq",
                            "fun": lambda w, t=target:
                                float(w @ mu) - t},
                       ])
        if res.success:
            w = np.clip(res.x, 0, None); w /= w.sum()
            r = float(w @ mu)
            v = float(np.sqrt(w @ cov_np @ w))
            points.append({"return": r, "vol": v})

    return points


# ── main ────────────────────────────────────────────────────
mu_s, cov_df = load_mu_cov()
tickers  = list(mu_s.index)
mu_np    = mu_s.values.copy()
cov_np   = cov_df.values.copy()
if np.linalg.eigvalsh(cov_np).min() < 0:
    cov_np = _near_psd(cov_np)

print(f"Universe size: {len(tickers)} tickers")
print("✅ Using training-only mu/cov (no look-ahead bias)")

ms = max_sharpe(mu_np, cov_np)
mv = min_variance(mu_np, cov_np)
frontier = efficient_frontier(mu_np, cov_np)

print(f"Max Sharpe : S={ms['sharpe']:.4f} | "
      f"Return={ms['return']*100:.2f}% | "
      f"Vol={ms['vol']*100:.2f}%")
print(f"Min Var    : Return={mv['return']*100:.2f}% | "
      f"Vol={mv['vol']*100:.2f}%")
print(f"Frontier   : {len(frontier)} points")

# ── plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))

frontier_vol = [p["vol"] for p in frontier]
frontier_ret = [p["return"] for p in frontier]

ax.plot(frontier_vol, frontier_ret,
        color="#2e75b6", linewidth=2.2,
        label="Efficient Frontier", zorder=2)

ax.scatter(ms["vol"], ms["return"],
           color="#2e8b57", s=140, marker="*", zorder=5,
           label=f"Max Sharpe (S={ms['sharpe']:.2f})")

ax.scatter(mv["vol"], mv["return"],
           color="#c00000", s=80, marker="o", zorder=5,
           label="Min Variance")

ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")
ax.set_title(f"Efficient Frontier ({len(tickers)}-ticker "
             f"NASDAQ universe)",
             fontsize=12, fontweight="bold")

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()

out = "figure1_efficient_frontier.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")