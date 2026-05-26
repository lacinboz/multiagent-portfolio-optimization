
# portfolio_prediction_core
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

DATA_DIR = Path("data/processed_yahoo")


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

# portfolio_prediction_core
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

DATA_DIR = Path("data/processed_yahoo")





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

def run_portfolio_optimization_prediction_ablation(
    mu,
    cov,
    rf=0.02,
    w_max=0.30,
    lambda_l2=1e-3,
    news_constraints=None,
    data_dir=DATA_DIR,
    save_csv=False,  # ablation'da False default — dosya kirletme
):
    """
    Ablation study için optimizer.
    Tek farkı: relaxed_w_max destekler (A2/B2 configs için).
    Dashboard'daki run_portfolio_optimization_prediction'a dokunmaz.
    """
    tickers = list(mu.index)
    cov = cov.loc[tickers, tickers].astype(float)
    mu = mu.astype(float)
    n = len(tickers)

    min_feasible_wmax = (1.0 / n) if n > 0 else 1.0
    effective_w_max = float(w_max)
    if n * effective_w_max < 1.0:
        effective_w_max = min_feasible_wmax + 1e-6

    eigvals = np.linalg.eigvalsh(cov.values)
    if eigvals.min() < 0:
        cov_np = near_psd(cov.values)
        cov = pd.DataFrame(cov_np, index=tickers, columns=tickers)

    # ── Per-ticker bounds: relaxed_w_max desteği ──────────────
    # Tek fark burası. A2/B2'de GOOGL için (0.0, 0.32) olur,
    # diğer config'lerde hepsi (0.0, 0.30) kalır.
    bounds = []
    for t in tickers:
        upper = effective_w_max
        if news_constraints and t in news_constraints:
            cdict = news_constraints[t]
            if "relaxed_w_max" in cdict:
                upper = max(float(cdict["relaxed_w_max"]), upper)
        bounds.append((0.0, upper))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    if news_constraints:
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        for ticker, cdict in news_constraints.items():
            if ticker not in ticker_to_idx:
                continue
            idx = ticker_to_idx[ticker]
            if "min_weight" in cdict:
                min_w = float(cdict["min_weight"])
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, i=idx, mw=min_w: w[i] - mw
                })
            if "max_weight" in cdict:
                max_w = float(cdict["max_weight"])
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, i=idx, mw=max_w: mw - w[i]
                })

    w0 = np.full(n, 1.0 / n)

    def obj_min_var(w):
        return (w @ cov.values @ w) + lambda_l2 * np.sum(w**2)

    def obj_neg_sharpe(w):
        return -sharpe_ratio(w, mu, cov, rf=rf)

    # ── Min-Var ────────────────────────────────────────────────
    res_minvar = minimize(obj_min_var, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
    # SLSQP başarısız olursa trust-constr ile tekrar dene
    if not res_minvar.success:
        res_minvar = minimize(obj_min_var, w0, method="trust-constr",
                              bounds=bounds, constraints=constraints)

    w_minvar = pd.Series(res_minvar.x, index=tickers)
    r_minvar, v_minvar = portfolio_stats(w_minvar.values, mu, cov)
    rc_abs, rc_pct = risk_contributions(w_minvar.values, cov)

    # ── Max-Sharpe ─────────────────────────────────────────────
    res_maxsharpe = minimize(obj_neg_sharpe, w0, method="SLSQP",
                             bounds=bounds, constraints=constraints)
    if not res_maxsharpe.success:
        res_maxsharpe = minimize(obj_neg_sharpe, w0, method="trust-constr",
                                 bounds=bounds, constraints=constraints)

    w_maxsharpe = pd.Series(res_maxsharpe.x, index=tickers)
    r_ms, v_ms = portfolio_stats(w_maxsharpe.values, mu, cov)
    sharpe_ms = (r_ms - rf) / v_ms if v_ms > 0 else 0.0

    print(f"[ABLATION] Min-Var success: {res_minvar.success} | {res_minvar.message}")
    print(f"[ABLATION] Max-Sharpe success: {res_maxsharpe.success} | {res_maxsharpe.message}")
    print(f"[ABLATION] Weights: { {t: f'{w_maxsharpe[t]:.1%}' for t in tickers} }")

    return {
        "tickers": tickers,
        "minvar": {
            "success": bool(res_minvar.success),
            "return": float(r_minvar),
            "vol": float(v_minvar),
            "weights": {t: float(w_minvar[t]) for t in tickers},
        },
        "maxsharpe": {
            "success": bool(res_maxsharpe.success),
            "return": float(r_ms),
            "vol": float(v_ms),
            "sharpe": float(sharpe_ms),
            "weights": {t: float(w_maxsharpe[t]) for t in tickers},
        },
        "bounds_used": {t: bounds[i] for i, t in enumerate(tickers)},
        "constraint_debug": [],
    }
def build_prediction_adjusted_bounds(
    tickers,
    base_w_max,
    prediction_probs,
    alpha=0.20,
    min_w=0.0,
    clip_upper=1.0
):
    """
    LLM / prediction probability based dynamic allocation constraints.

    Constraint form:
        w_i <= base_w_max + alpha * (p_i - 0.5)

    Parameters
    ----------
    tickers : list[str]
        Portfolio assets.

    base_w_max : float
        Base maximum allocation per asset.

    prediction_probs : dict
        Example:
        {
            "AAPL": 0.78,
            "MSFT": 0.61,
            "TSLA": 0.32
        }

    alpha : float
        Strength of prediction influence.

    min_w : float
        Minimum allowed upper bound after adjustment.

    clip_upper : float
        Maximum possible cap.

    Returns
    -------
    bounds : list[tuple]
        Bounds usable directly in scipy.optimize.minimize
    adjusted_caps : dict
        Human-readable adjusted caps per asset.
    """

    bounds = []
    adjusted_caps = {}

    for ticker in tickers:

        p = prediction_probs.get(ticker, 0.5)
        p = float(np.clip(p, 0.0, 1.0))

        adjusted_cap = base_w_max + alpha * (p - 0.5)

        # numerical safety
        adjusted_cap = max(min_w, adjusted_cap)
        adjusted_cap = min(clip_upper, adjusted_cap)

        bounds.append((0.0, adjusted_cap))

        adjusted_caps[ticker] = float(adjusted_cap)

    return bounds, adjusted_caps

def run_portfolio_optimization_prediction(
    mu,
    cov,
    rf=0.02,
    w_max=0.30,
    lambda_l2=1e-3,
    news_constraints=None,
    data_dir=DATA_DIR,
    save_csv=True
):
    """
    Portfolio optimization with threshold-based
    news probability constraints.

    News predictions DO NOT modify:
        - expected returns
        - covariance matrix

    They only constrain feasible allocations.
    """

    tickers = list(mu.index)

    cov = cov.loc[tickers, tickers].astype(float)
    mu = mu.astype(float)

    n = len(tickers)

    print("Tickers used:", tickers)

    print("\n========== PREDICTION OPTIMIZER DEBUG ==========")

    print("Incoming news constraints:")
    print(news_constraints)

    print("RF:", rf)
    print("Requested w_max:", w_max)

    print("Mu:")
    print(mu)

    print("================================================\n")
    # =====================================================
    # Feasibility guard
    # =====================================================

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

    print(
        f"Requested w_max: {w_max:.4f} | "
        f"Effective w_max: {effective_w_max:.4f}"
    )

    # =====================================================
    # PSD check
    # =====================================================

    eigvals = np.linalg.eigvalsh(cov.values)

    print("Smallest eigenvalue:", eigvals.min())

    if eigvals.min() < 0:

        print("Covariance is not PSD; applying near PSD.")

        cov_np = near_psd(cov.values)

        cov = pd.DataFrame(
            cov_np,
            index=tickers,
            columns=tickers
        )

    # =====================================================
    # Standard bounds
    # =====================================================

    bounds = [(0.0, effective_w_max)] * n

    # =====================================================
    # Base constraints
    # =====================================================

    constraints = [
        {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }
    ]

    # =====================================================
    # News-based threshold constraints
    # =====================================================

    if news_constraints:

        ticker_to_idx = {
            t: i for i, t in enumerate(tickers)
        }

        for ticker, cdict in news_constraints.items():
            print(f"\nProcessing constraint for {ticker}")
            print("Constraint dict:", cdict)

            if ticker not in ticker_to_idx:
                continue

            idx = ticker_to_idx[ticker]

            # ---------------------------------------------
            # Bullish -> minimum allocation
            # ---------------------------------------------

            if "min_weight" in cdict:

                min_w = float(cdict["min_weight"])

                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, i=idx, mw=min_w:
                        w[i] - mw
                })

                print(
                    f"[NEWS CONSTRAINT] "
                    f"{ticker} -> w >= {min_w:.4f}"
                )

            # ---------------------------------------------
            # Bearish -> maximum allocation
            # ---------------------------------------------

            if "max_weight" in cdict:

                max_w = float(cdict["max_weight"])

                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, i=idx, mw=max_w:
                        mw - w[i]
                })

                print(
                    f"[NEWS CONSTRAINT] "
                    f"{ticker} -> w <= {max_w:.4f}"
                )

    # =====================================================
    # Initial weights
    # =====================================================

    w0 = np.full(n, 1 / n)

    # =====================================================
    # Objective functions
    # =====================================================

    def obj_min_var(w):

        return (
            (w @ cov.values @ w)
            + lambda_l2 * np.sum(w**2)
        )

    def obj_neg_sharpe(w):

        return -sharpe_ratio(
            w,
            mu,
            cov,
            rf=rf
        )

    # =====================================================
    # Min-Variance portfolio
    # =====================================================

    res_minvar = minimize(
        obj_min_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w_minvar = pd.Series(
        res_minvar.x,
        index=tickers
    )

    r_minvar, v_minvar = portfolio_stats(
        w_minvar.values,
        mu,
        cov
    )

    rc_abs, rc_pct = risk_contributions(
        w_minvar.values,
        cov
    )

    print(
        "Min-Var success:",
        res_minvar.success,
        "| message:",
        res_minvar.message
    )

    # =====================================================
    # Max-Sharpe portfolio
    # =====================================================

    res_maxsharpe = minimize(
        obj_neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w_maxsharpe = pd.Series(
        res_maxsharpe.x,
        index=tickers
    )

    r_ms, v_ms = portfolio_stats(
        w_maxsharpe.values,
        mu,
        cov
    )

    sharpe_ms = (
        (r_ms - rf) / v_ms
    )

    print(
        "Max-Sharpe success:",
        res_maxsharpe.success,
        "| message:",
        res_maxsharpe.message
    )
    print("\n===== FINAL MAX SHARPE WEIGHTS =====")

    for t in tickers:
        print(f"{t}: {w_maxsharpe[t]:.6f}")

    print("===================================\n")

    # =====================================================
    # Efficient frontier
    # =====================================================

    def min_var_for_target_return(target):

        cons = constraints + [
            {
                'type': 'eq',
                'fun': lambda w, t=target:
                    float(w @ mu.values) - float(t)
            }
        ]

        res = minimize(
            obj_min_var,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons
        )

        return res

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
        r_maxret = float(mu.max())

    lo = float(r_minvar)
    hi = float(r_maxret)

    if hi < lo:
        lo, hi = hi, lo

    grid = np.linspace(lo, hi, 60)

    frontier_rows = []

    for t in grid:

        res = min_var_for_target_return(t)

        if res.success:

            w = res.x

            r, v = portfolio_stats(
                w,
                mu,
                cov
            )

            frontier_rows.append({
                "target_return": float(t),
                "realized_return": float(r),
                "vol": float(v),
            })

    frontier = pd.DataFrame(frontier_rows)

    # =====================================================
    # Save outputs
    # =====================================================

    if save_csv:

        out_minvar = pd.DataFrame({
            "ticker": tickers,
            "weight": w_minvar.values,
            "rc_abs": rc_abs,
            "rc_pct": rc_pct
        })

        out_minvar.to_csv(
            data_dir / "portfolio_minvar.csv",
            index=False
        )

        out_maxsharpe = pd.DataFrame({
            "ticker": tickers,
            "weight": w_maxsharpe.values
        })

        out_maxsharpe.to_csv(
            data_dir / "portfolio_maxsharpe.csv",
            index=False
        )

        frontier.to_csv(
            data_dir / "efficient_frontier.csv",
            index=False
        )

    # =====================================================
    # Constraint debug table
    # =====================================================

    constraint_debug_rows = []

    # Max-Sharpe portfolio üzerinden compare edeceğiz
    final_weights = {
        t: float(w_maxsharpe[t])
        for t in tickers
    }

    for ticker in tickers:

        # -------------------------------------------------
        # Baseline cap (normal optimizer limit)
        # -------------------------------------------------

        baseline_cap = effective_w_max

        # -------------------------------------------------
        # News-adjusted cap
        # -------------------------------------------------

        adjusted_cap = baseline_cap

        constraint_type = "None"

        if news_constraints and ticker in news_constraints:

            cdict = news_constraints[ticker]

            if "max_weight" in cdict:

                adjusted_cap = float(cdict["max_weight"])

                constraint_type = "Bearish Max Cap"

            elif "min_weight" in cdict:

                adjusted_cap = float(cdict["min_weight"])

                constraint_type = "Bullish Min Floor"

        # -------------------------------------------------
        # Final optimized allocation
        # -------------------------------------------------

        final_weight = final_weights.get(ticker, 0.0)

        # -------------------------------------------------
        # Did constraint actually bind?
        # -------------------------------------------------

        constraint_binding = False

        if constraint_type == "Bearish Max Cap":

            constraint_binding = (
                abs(final_weight - adjusted_cap) < 1e-3
            )

        elif constraint_type == "Bullish Min Floor":

            constraint_binding = (
                abs(final_weight - adjusted_cap) < 1e-3
            )

        # -------------------------------------------------
        # Store row
        # -------------------------------------------------

        # constraint_debug_rows.append içini şöyle değiştir:

        constraint_debug_rows.append({
            "ticker": ticker,
            "constraint_type": constraint_type,
            "baseline_cap": float(baseline_cap),
            
            # Bullish için min_weight, bearish için max_weight, none için None
            "adjusted_min_weight": float(cdict["min_weight"]) if (
                news_constraints and ticker in news_constraints 
                and "min_weight" in news_constraints[ticker]
            ) else None,
            
            "adjusted_max_weight": float(cdict["max_weight"]) if (
                news_constraints and ticker in news_constraints 
                and "max_weight" in news_constraints[ticker]
            ) else None,
            
            # Geriye dönük uyumluluk için adjusted_cap da kalsın
            "adjusted_cap": float(adjusted_cap),
            
            "final_weight": float(final_weight),
            "cap_delta": float(adjusted_cap - baseline_cap),
            "constraint_binding": bool(constraint_binding),
        })
    constraint_debug_df = pd.DataFrame(
        constraint_debug_rows
    )
    print("\n===== CONSTRAINT DEBUG TABLE =====")
    print(constraint_debug_df)
    print("==================================\n")
    # =====================================================
    # Final result
    # =====================================================

    result = {
        "tickers": tickers,
        "rf": rf,
        "w_max": w_max,
        "effective_w_max": effective_w_max,
        "lambda_l2": lambda_l2,

        "constraint_type":
            "threshold_probability_constraints",

        "news_constraints":
            news_constraints,

        "minvar": {
            "success": bool(res_minvar.success),
            "return": float(r_minvar),
            "vol": float(v_minvar),
            "weights": {
                t: float(w_minvar[t])
                for t in tickers
            },
            "rc_pct": {
                t: float(rc_pct[i])
                for i, t in enumerate(tickers)
            }
        },

        "maxsharpe": {
            "success": bool(res_maxsharpe.success),
            "return": float(r_ms),
            "vol": float(v_ms),
            "sharpe": float(sharpe_ms),
            "weights": {
                t: float(w_maxsharpe[t])
                for t in tickers
            }
        },
        "constraint_debug": (
            constraint_debug_df.to_dict(orient="records")
        ),

        "frontier": [
            {
                "target_return":
                    float(row["target_return"]),

                "realized_return":
                    float(row["realized_return"]),

                "vol":
                    float(row["vol"])
            }
            for _, row in frontier.iterrows()
        ]
    }

    return result

def build_prediction_adjusted_bounds(
    tickers,
    base_w_max,
    prediction_probs,
    alpha=0.20,
    min_w=0.0,
    clip_upper=1.0
):
    """
    LLM / prediction probability based dynamic allocation constraints.

    Constraint form:
        w_i <= base_w_max + alpha * (p_i - 0.5)

    Parameters
    ----------
    tickers : list[str]
        Portfolio assets.

    base_w_max : float
        Base maximum allocation per asset.

    prediction_probs : dict
        Example:
        {
            "AAPL": 0.78,
            "MSFT": 0.61,
            "TSLA": 0.32
        }

    alpha : float
        Strength of prediction influence.

    min_w : float
        Minimum allowed upper bound after adjustment.

    clip_upper : float
        Maximum possible cap.

    Returns
    -------
    bounds : list[tuple]
        Bounds usable directly in scipy.optimize.minimize
    adjusted_caps : dict
        Human-readable adjusted caps per asset.
    """

    bounds = []
    adjusted_caps = {}

    for ticker in tickers:

        p = prediction_probs.get(ticker, 0.5)
        p = float(np.clip(p, 0.0, 1.0))

        adjusted_cap = base_w_max + alpha * (p - 0.5)

        # numerical safety
        adjusted_cap = max(min_w, adjusted_cap)
        adjusted_cap = min(clip_upper, adjusted_cap)

        bounds.append((0.0, adjusted_cap))

        adjusted_caps[ticker] = float(adjusted_cap)

    return bounds, adjusted_caps

def run_portfolio_optimization_prediction(
    mu,
    cov,
    rf=0.02,
    w_max=0.30,
    lambda_l2=1e-3,
    news_constraints=None,
    data_dir=DATA_DIR,
    save_csv=True
):
    """
    Portfolio optimization with threshold-based
    news probability constraints.

    News predictions DO NOT modify:
        - expected returns
        - covariance matrix

    They only constrain feasible allocations.
    """

    tickers = list(mu.index)

    cov = cov.loc[tickers, tickers].astype(float)
    mu = mu.astype(float)

    n = len(tickers)

    print("Tickers used:", tickers)

    print("\n========== PREDICTION OPTIMIZER DEBUG ==========")

    print("Incoming news constraints:")
    print(news_constraints)

    print("RF:", rf)
    print("Requested w_max:", w_max)

    print("Mu:")
    print(mu)

    print("================================================\n")
    # =====================================================
    # Feasibility guard
    # =====================================================

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

    print(
        f"Requested w_max: {w_max:.4f} | "
        f"Effective w_max: {effective_w_max:.4f}"
    )

    # =====================================================
    # PSD check
    # =====================================================

    eigvals = np.linalg.eigvalsh(cov.values)

    print("Smallest eigenvalue:", eigvals.min())

    if eigvals.min() < 0:

        print("Covariance is not PSD; applying near PSD.")

        cov_np = near_psd(cov.values)

        cov = pd.DataFrame(
            cov_np,
            index=tickers,
            columns=tickers
        )

    # =====================================================
    # Standard bounds
    # =====================================================

    bounds = [(0.0, effective_w_max)] * n

    # =====================================================
    # Base constraints
    # =====================================================

    constraints = [
        {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }
    ]

    # =====================================================
    # News-based threshold constraints
    # =====================================================

    if news_constraints:

        ticker_to_idx = {
            t: i for i, t in enumerate(tickers)
        }

        for ticker, cdict in news_constraints.items():
            print(f"\nProcessing constraint for {ticker}")
            print("Constraint dict:", cdict)

            if ticker not in ticker_to_idx:
                continue

            idx = ticker_to_idx[ticker]

            # ---------------------------------------------
            # Bullish -> minimum allocation
            # ---------------------------------------------

            if "min_weight" in cdict:

                min_w = float(cdict["min_weight"])

                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, i=idx, mw=min_w:
                        w[i] - mw
                })

                print(
                    f"[NEWS CONSTRAINT] "
                    f"{ticker} -> w >= {min_w:.4f}"
                )

            # ---------------------------------------------
            # Bearish -> maximum allocation
            # ---------------------------------------------

            if "max_weight" in cdict:

                max_w = float(cdict["max_weight"])

                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, i=idx, mw=max_w:
                        mw - w[i]
                })

                print(
                    f"[NEWS CONSTRAINT] "
                    f"{ticker} -> w <= {max_w:.4f}"
                )

    # =====================================================
    # Initial weights
    # =====================================================

    w0 = np.full(n, 1 / n)

    # =====================================================
    # Objective functions
    # =====================================================

    def obj_min_var(w):

        return (
            (w @ cov.values @ w)
            + lambda_l2 * np.sum(w**2)
        )

    def obj_neg_sharpe(w):

        return -sharpe_ratio(
            w,
            mu,
            cov,
            rf=rf
        )

    # =====================================================
    # Min-Variance portfolio
    # =====================================================

    res_minvar = minimize(
        obj_min_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w_minvar = pd.Series(
        res_minvar.x,
        index=tickers
    )

    r_minvar, v_minvar = portfolio_stats(
        w_minvar.values,
        mu,
        cov
    )

    rc_abs, rc_pct = risk_contributions(
        w_minvar.values,
        cov
    )

    print(
        "Min-Var success:",
        res_minvar.success,
        "| message:",
        res_minvar.message
    )

    # =====================================================
    # Max-Sharpe portfolio
    # =====================================================

    res_maxsharpe = minimize(
        obj_neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w_maxsharpe = pd.Series(
        res_maxsharpe.x,
        index=tickers
    )

    r_ms, v_ms = portfolio_stats(
        w_maxsharpe.values,
        mu,
        cov
    )

    sharpe_ms = (
        (r_ms - rf) / v_ms
    )

    print(
        "Max-Sharpe success:",
        res_maxsharpe.success,
        "| message:",
        res_maxsharpe.message
    )
    print("\n===== FINAL MAX SHARPE WEIGHTS =====")

    for t in tickers:
        print(f"{t}: {w_maxsharpe[t]:.6f}")

    print("===================================\n")

    # =====================================================
    # Efficient frontier
    # =====================================================

    def min_var_for_target_return(target):

        cons = constraints + [
            {
                'type': 'eq',
                'fun': lambda w, t=target:
                    float(w @ mu.values) - float(t)
            }
        ]

        res = minimize(
            obj_min_var,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons
        )

        return res

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
        r_maxret = float(mu.max())

    lo = float(r_minvar)
    hi = float(r_maxret)

    if hi < lo:
        lo, hi = hi, lo

    grid = np.linspace(lo, hi, 60)

    frontier_rows = []

    for t in grid:

        res = min_var_for_target_return(t)

        if res.success:

            w = res.x

            r, v = portfolio_stats(
                w,
                mu,
                cov
            )

            frontier_rows.append({
                "target_return": float(t),
                "realized_return": float(r),
                "vol": float(v),
            })

    frontier = pd.DataFrame(frontier_rows)

    # =====================================================
    # Save outputs
    # =====================================================

    if save_csv:

        out_minvar = pd.DataFrame({
            "ticker": tickers,
            "weight": w_minvar.values,
            "rc_abs": rc_abs,
            "rc_pct": rc_pct
        })

        out_minvar.to_csv(
            data_dir / "portfolio_minvar.csv",
            index=False
        )

        out_maxsharpe = pd.DataFrame({
            "ticker": tickers,
            "weight": w_maxsharpe.values
        })

        out_maxsharpe.to_csv(
            data_dir / "portfolio_maxsharpe.csv",
            index=False
        )

        frontier.to_csv(
            data_dir / "efficient_frontier.csv",
            index=False
        )

    # =====================================================
    # Constraint debug table
    # =====================================================

    constraint_debug_rows = []

    # Max-Sharpe portfolio üzerinden compare edeceğiz
    final_weights = {
        t: float(w_maxsharpe[t])
        for t in tickers
    }

    for ticker in tickers:

        # -------------------------------------------------
        # Baseline cap (normal optimizer limit)
        # -------------------------------------------------

        baseline_cap = effective_w_max

        # -------------------------------------------------
        # News-adjusted cap
        # -------------------------------------------------

        adjusted_cap = baseline_cap

        constraint_type = "None"

        if news_constraints and ticker in news_constraints:

            cdict = news_constraints[ticker]

            if "max_weight" in cdict:

                adjusted_cap = float(cdict["max_weight"])

                constraint_type = "Bearish Max Cap"

            elif "min_weight" in cdict:

                adjusted_cap = float(cdict["min_weight"])

                constraint_type = "Bullish Min Floor"

        # -------------------------------------------------
        # Final optimized allocation
        # -------------------------------------------------

        final_weight = final_weights.get(ticker, 0.0)

        # -------------------------------------------------
        # Did constraint actually bind?
        # -------------------------------------------------

        constraint_binding = False

        if constraint_type == "Bearish Max Cap":

            constraint_binding = (
                abs(final_weight - adjusted_cap) < 1e-3
            )

        elif constraint_type == "Bullish Min Floor":

            constraint_binding = (
                abs(final_weight - adjusted_cap) < 1e-3
            )

        # -------------------------------------------------
        # Store row
        # -------------------------------------------------

        # constraint_debug_rows.append içini şöyle değiştir:

        constraint_debug_rows.append({
            "ticker": ticker,
            "constraint_type": constraint_type,
            "baseline_cap": float(baseline_cap),
            
            # Bullish için min_weight, bearish için max_weight, none için None
            "adjusted_min_weight": float(cdict["min_weight"]) if (
                news_constraints and ticker in news_constraints 
                and "min_weight" in news_constraints[ticker]
            ) else None,
            
            "adjusted_max_weight": float(cdict["max_weight"]) if (
                news_constraints and ticker in news_constraints 
                and "max_weight" in news_constraints[ticker]
            ) else None,
            
            # Geriye dönük uyumluluk için adjusted_cap da kalsın
            "adjusted_cap": float(adjusted_cap),
            
            "final_weight": float(final_weight),
            "cap_delta": float(adjusted_cap - baseline_cap),
            "constraint_binding": bool(constraint_binding),
        })
    constraint_debug_df = pd.DataFrame(
        constraint_debug_rows
    )
    print("\n===== CONSTRAINT DEBUG TABLE =====")
    print(constraint_debug_df)
    print("==================================\n")
    # =====================================================
    # Final result
    # =====================================================

    result = {
        "tickers": tickers,
        "rf": rf,
        "w_max": w_max,
        "effective_w_max": effective_w_max,
        "lambda_l2": lambda_l2,

        "constraint_type":
            "threshold_probability_constraints",

        "news_constraints":
            news_constraints,

        "minvar": {
            "success": bool(res_minvar.success),
            "return": float(r_minvar),
            "vol": float(v_minvar),
            "weights": {
                t: float(w_minvar[t])
                for t in tickers
            },
            "rc_pct": {
                t: float(rc_pct[i])
                for i, t in enumerate(tickers)
            }
        },

        "maxsharpe": {
            "success": bool(res_maxsharpe.success),
            "return": float(r_ms),
            "vol": float(v_ms),
            "sharpe": float(sharpe_ms),
            "weights": {
                t: float(w_maxsharpe[t])
                for t in tickers
            }
        },
        "constraint_debug": (
            constraint_debug_df.to_dict(orient="records")
        ),

        "frontier": [
            {
                "target_return":
                    float(row["target_return"]),

                "realized_return":
                    float(row["realized_return"]),

                "vol":
                    float(row["vol"])
            }
            for _, row in frontier.iterrows()
        ]
    }

    return result