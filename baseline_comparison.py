"""
baseline_comparison_unified.py
=======================================================================
Unified baseline comparison — all 5 methods evaluated on the SAME data,
SAME tickers, SAME chronological 70/30 train/test split.

KEY DESIGN PRINCIPLE (matches the actual production pipeline):
  - The pipeline is SNAPSHOT-BASED with periodic rebalancing.
  - Model is trained on the TRAIN period.
  - At each rebalancing date, latest signals are generated from the
    most recent test data available up to that point.
  - Portfolio is re-optimized with those signals.
  - Portfolio is HELD for HORIZON_DAYS (7) until next rebalancing.
  - This matches how build_portfolio_graph_prediction_constraint() works
    and how portfolio_prediction_core.py is called.

Methods
-------
  1. Equal-Weight (1/N)            — trivial benchmark
  2. Plain MVO (no news)           — standard mean-variance, no text
  3. Zhang (2022) Long-Short       — FinBERT → daily long-short portfolio
  4. BL + FinBERT Views            — Colasanto et al. (2022)
  5. NC-MVO [Ours]                 — FinBERT → LogReg → Constraints → MVO

Our pipeline (for NC-MVO) exactly as in production:
  News → FinBERT → Logistic Regression → Threshold Constraints → MVO
  Using:
    - build_news_probability_constraints() from news_constraint_integration.py
    - prediction_constrained_optimization_agent() from agents_langgraph.py
    - optimization_agent_from_mu_cov() from agents_langgraph.py
    - mu/cov from data/processed_yahoo/ (build_returns_yahoo.py outputs)

Evaluation:
  - Chronological 70/30 split (same as train_news_flow_predictor)
  - Rebalancing: every HORIZON_DAYS=7 trading days
  - Returns: daily close-to-close on data/raw/daily_yahoo/
  - Metrics: annualised Sharpe, return, vol, max drawdown, alpha vs SPY

Outputs → data/baseline_comparison_unified/
  comparison_table.csv
  comparison_results.json
  cumulative_returns.png
  daily_returns_<method>.csv
  weights_<method>.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
from news_return_predictor import (
    ALL_TICKERS_RAW_PATH,
    HORIZON_DAYS,
    build_ticker_date_prediction_dataset_v2,
    prepare_model_frame,
    load_prediction_model,
    OUT_DIR as NEWS_OUT_DIR,
)
from agents_langgraph import (
    data_agent_get_mu_cov,
    prediction_constrained_optimization_agent,
    optimization_agent_from_mu_cov,
)
from news_constraint_integration import build_news_probability_constraints

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
PRICE_DIR        = Path("data/raw/daily_yahoo")
BEST_MODEL_PATH  = NEWS_OUT_DIR / "best_news_prediction_model.joblib"
OUT_DIR          = Path("data/baseline_comparison_unified")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TEST_SPLIT  = 0.70
RF_ANNUAL         = 0.02
W_MAX             = 0.30
LAMBDA_L2         = 1e-3
MIN_ABS_RETURN    = 0.02

# Rebalancing cadence — matches HORIZON_DAYS in news_return_predictor.py
REBALANCE_EVERY   = HORIZON_DAYS   # 7 trading days

# Zhang (2022)
ZHANG_N_LONG      = 5
ZHANG_N_SHORT     = 5
ZHANG_MIN_NEWS    = 50

# Colasanto (2022) BL
BL_TAU            = 0.05
BL_RISK_AVERSION  = 4.4644
BL_N_MC_PATHS     = 10_000
BL_HORIZON_DAYS   = 7

# NC-MVO (Ours) — MUST match news_constraint_integration.py exactly
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA             = 0.02


# ════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _load_daily_prices(tickers: List[str]) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        path = PRICE_DIR / f"{t}_daily.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "timestamp" not in df.columns or "close" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["close"]     = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp")
        s  = df.set_index("timestamp")["close"].astype(float)
        s.index = s.index.normalize()
        frames[t] = s
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames)
    prices.index.name = "date"
    return prices


def _portfolio_metrics(daily_returns: pd.Series, rf_annual: float = RF_ANNUAL) -> Dict[str, Any]:
    r = daily_returns.dropna()
    if r.empty:
        return {}
    rf_d = rf_annual / 252
    er   = r - rf_d
    ann_ret = float((1 + r).prod() ** (252 / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe  = float(er.mean() / er.std() * np.sqrt(252)) if er.std() > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = float((cum / cum.cummax() - 1).min())
    return {
        "annualised_return":     ann_ret,
        "annualised_volatility": ann_vol,
        "sharpe_ratio":          sharpe,
        "max_drawdown":          max_dd,
        "pct_profitable_days":   float((r > 0).mean()),
        "mean_daily_return":     float(r.mean()),
        "std_daily_return":      float(r.std()),
        "n_trading_days":        int(len(r)),
    }



def _near_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    return vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T


def _mvo_maxsharpe_local(mu: np.ndarray, Sigma: np.ndarray,
                          rf: float = RF_ANNUAL, w_max: float = W_MAX) -> np.ndarray:
    """Local max-Sharpe solver — mirrors portfolio_core.py."""
    n = len(mu)
    eff = max(w_max, 1.0 / n + 1e-6)
    if np.linalg.eigvalsh(Sigma).min() < 0:
        Sigma = _near_psd(Sigma)
    w0 = np.full(n, 1.0 / n)
    def neg_sharpe(w):
        r = float(w @ mu)
        v = float(np.sqrt(w @ Sigma @ w))
        return -(r - rf) / v if v > 0 else np.inf
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=[(0.0, eff)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def _compute_period_returns(
    weights: Dict[str, float],
    price_returns: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Daily portfolio returns for (start_date, end_date] with fixed weights."""
    mask   = (price_returns.index > start_date) & (price_returns.index <= end_date)
    period = price_returns.loc[mask]
    if period.empty:
        return pd.Series(dtype=float)
    valid = [t for t in weights if t in period.columns]
    if not valid:
        return pd.Series(dtype=float)
    w = np.array([weights[t] for t in valid], dtype=float)
    s = w.sum()
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s
    ret = period[valid].fillna(0.0).values @ w
    return pd.Series(ret, index=period.index)


def _rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    every: int = REBALANCE_EVERY,
) -> List[pd.Timestamp]:
    """Rebalance dates spaced every `every` trading days starting from test start."""
    test_dates = trading_dates[trading_dates >= start_date]
    return [test_dates[i] for i in range(0, len(test_dates), every)]


# ════════════════════════════════════════════════════════════════════════════
# METHOD 1 — EQUAL WEIGHT
# ════════════════════════════════════════════════════════════════════════════

def run_equal_weight(
    tickers: List[str],
    prices: pd.DataFrame,
    test_start_date: pd.Timestamp,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("METHOD 1 — Equal Weight (1/N)")
    print("=" * 60)
    valid = [t for t in tickers if t in prices.columns]
    if not valid:
        return {"ok": False, "reason": "No valid tickers."}
    w             = {t: 1.0 / len(valid) for t in valid}
    price_returns = prices.pct_change().dropna(how="all")
    rebal_dates   = _rebalance_dates(price_returns.index, test_start_date)
    all_ret       = []
    for i, rd in enumerate(rebal_dates):
        end = rebal_dates[i+1] if i+1 < len(rebal_dates) else price_returns.index[-1]
        all_ret.append(_compute_period_returns(w, price_returns, rd, end))
    if not all_ret:
        return {"ok": False, "reason": "No periods."}
    ret_s = pd.concat(all_ret).sort_index()
    ret_s = ret_s[~ret_s.index.duplicated(keep="first")]
    m = _portfolio_metrics(ret_s)
    print(f"Sharpe={m.get('sharpe_ratio',float('nan')):.4f}  "
          f"Return={m.get('annualised_return',float('nan'))*100:.2f}%  "
          f"Vol={m.get('annualised_volatility',float('nan'))*100:.2f}%")
    return {
        "ok": True, "method": "1. Equal Weight (1/N)",
        "description": "Equal allocation — no optimisation, no news",
        "metrics": m,
        "daily_returns": ret_s,
        "returns_df": ret_s.to_frame("portfolio_return"),
        "test_start_date": str(test_start_date.date()),
        "weights": w,
    }


# ════════════════════════════════════════════════════════════════════════════
# METHOD 2 — PLAIN MVO
# ════════════════════════════════════════════════════════════════════════════

def run_plain_mvo(
    mu: pd.Series,
    cov: pd.DataFrame,
    prices: pd.DataFrame,
    test_start_date: pd.Timestamp,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("METHOD 2 — Plain MVO (no news)")
    print("=" * 60)
    try:
        result = optimization_agent_from_mu_cov(
            mu=mu, cov=cov, rf=RF_ANNUAL, w_max=W_MAX, lambda_l2=LAMBDA_L2
        )
        w_dict = result.get("maxsharpe", {}).get("weights", {}) or {}
        w = {t: float(v) for t, v in w_dict.items() if float(v) > 1e-6}
    except Exception as e:
        return {"ok": False, "reason": f"MVO failed: {e}"}
    if not w:
        return {"ok": False, "reason": "Empty MVO weights."}
    print(f"Active: {len(w)}  Top5: {sorted(w.items(), key=lambda x:x[1], reverse=True)[:5]}")
    price_returns = prices.pct_change().dropna(how="all")
    rebal_dates   = _rebalance_dates(price_returns.index, test_start_date)
    all_ret       = []
    for i, rd in enumerate(rebal_dates):
        end = rebal_dates[i+1] if i+1 < len(rebal_dates) else price_returns.index[-1]
        all_ret.append(_compute_period_returns(w, price_returns, rd, end))
    if not all_ret:
        return {"ok": False, "reason": "No periods."}
    ret_s = pd.concat(all_ret).sort_index()
    ret_s = ret_s[~ret_s.index.duplicated(keep="first")]
    m = _portfolio_metrics(ret_s)
    print(f"Sharpe={m.get('sharpe_ratio',float('nan')):.4f}  "
          f"Return={m.get('annualised_return',float('nan'))*100:.2f}%  "
          f"Vol={m.get('annualised_volatility',float('nan'))*100:.2f}%")
    return {
        "ok": True, "method": "2. Plain MVO (no news)",
        "description": "Max-Sharpe MVO — historical mu/cov, no news signal",
        "metrics": m,
        "daily_returns": ret_s,
        "returns_df": ret_s.to_frame("portfolio_return"),
        "test_start_date": str(test_start_date.date()),
        "weights": w,
        "static_sharpe": result.get("maxsharpe",{}).get("sharpe"),
    }


# ════════════════════════════════════════════════════════════════════════════
# METHOD 3 — ZHANG (2022) LONG-SHORT
# ════════════════════════════════════════════════════════════════════════════

def _zhang_score(row: pd.Series) -> float:
    pp = float(row.get("prob_positive", 0) or 0)
    pn = float(row.get("prob_negative", 0) or 0)
    pz = float(row.get("prob_neutral",  0) or 0)
    if pp >= pn and pp >= pz:   return pp
    elif pn > pp and pn >= pz:  return -pn
    return 0.0


def run_zhang_longshort(
    raw_df: pd.DataFrame,
    prices: pd.DataFrame,
    test_start_date: pd.Timestamp,
    n_long: int = ZHANG_N_LONG,
    n_short: int = ZHANG_N_SHORT,
    min_news: int = ZHANG_MIN_NEWS,
) -> Dict[str, Any]:
    """
    Daily zero-cost long-short — faithful to Zhang (2022) Sec. 4.1.
    This method is inherently daily-rebalancing per the paper.
    """
    print("\n" + "=" * 60)
    print("METHOD 3 — Zhang (2022) Long-Short")
    print("=" * 60)
    df = raw_df.copy()
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=["news_date_dt"]).copy()
    df["zhang_score"] = df.apply(_zhang_score, axis=1)
    scores = (
        df.groupby(["ticker","news_date_dt"])
        .agg(sentiment_score=("zhang_score","mean"), n_articles=("zhang_score","count"))
        .reset_index()
    )
    test_scores   = scores[scores["news_date_dt"] >= test_start_date].copy()
    price_returns = prices.pct_change().dropna(how="all")
    daily = []
    for date, day_sc in test_scores.groupby("news_date_dt"):
        if len(df[df["news_date_dt"] == date]) < min_news:
            continue
        ds = day_sc.sort_values("sentiment_score", ascending=False)
        cl = ds[ds["sentiment_score"] > 0]
        cs = ds[ds["sentiment_score"] < 0]
        if cl.empty or cs.empty:
            continue
        lt = cl.head(n_long)["ticker"].tolist()
        st = cs.tail(n_short)["ticker"].tolist()
        nd_list = price_returns.index[price_returns.index > date]
        if len(nd_list) == 0:
            continue
        nd  = nd_list[0]
        row = price_returns.loc[nd]
        lr  = [float(row[t]) for t in lt if t in row.index and not np.isnan(row[t])]
        sr  = [float(row[t]) for t in st if t in row.index and not np.isnan(row[t])]
        if not lr or not sr:
            continue
        daily.append({"date": nd, "portfolio_return": np.mean(lr) - np.mean(sr),
                      "long_return": np.mean(lr), "short_return": np.mean(sr),
                      "n_long": len(lr), "n_short": len(sr)})
    if not daily:
        return {"ok": False, "reason": f"No days with >= {min_news} articles."}
    ret_df = pd.DataFrame(daily).set_index("date")
    ret_s  = ret_df["portfolio_return"]
    m = _portfolio_metrics(ret_s)
    print(f"Trading days: {len(ret_df)}")
    print(f"Sharpe={m.get('sharpe_ratio',float('nan')):.4f}  "
          f"Return={m.get('annualised_return',float('nan'))*100:.2f}%  "
          f"Vol={m.get('annualised_volatility',float('nan'))*100:.2f}%")
    return {
        "ok": True, "method": "3. Zhang (2022) Long-Short",
        "description": (f"FinBERT → daily zero-cost long-short "
                        f"(top-{n_long}/bottom-{n_short}, min {min_news} articles/day). "
                        f"Daily rebalancing per paper."),
        "paper": "Zhang (2022) Portfolio Construction with News Sentiment using a LLM",
        "metrics": m,
        "daily_returns": ret_s,
        "returns_df": ret_df,
        "test_start_date": str(test_start_date.date()),
    }


# ════════════════════════════════════════════════════════════════════════════
# METHOD 4 — BLACK-LITTERMAN + FINBERT (Colasanto et al., 2022)
# ════════════════════════════════════════════════════════════════════════════

def _bl_c(ticker: str, test_df: pd.DataFrame) -> float:
    sub = test_df[test_df["ticker"] == ticker]
    if sub.empty:
        return 0.0
    s = (sub["prob_positive"].fillna(0).astype(float)
         - sub["prob_negative"].fillna(0).astype(float))
    return float(np.clip(s.mean(), -1.0, 1.0))


def _mc_yield(ticker: str, c: float, horizon: int = BL_HORIZON_DAYS,
              n_paths: int = BL_N_MC_PATHS, hist_days: int = 252) -> float:
    path = PRICE_DIR / f"{ticker}_daily.csv"
    if not path.exists():
        return float(c * 0.05)
    cl = pd.read_csv(path)
    cl["timestamp"] = pd.to_datetime(cl["timestamp"], errors="coerce")
    cl["close"]     = pd.to_numeric(cl["close"], errors="coerce")
    cl = cl.dropna(subset=["timestamp","close"]).sort_values("timestamp")
    close = cl["close"].astype(float).values
    if len(close) < hist_days + horizon + 10:
        return float(c * 0.05)
    hist  = close[-(hist_days + horizon):-horizon]
    S0    = float(close[-(horizon + 1)])
    lr    = np.diff(np.log(hist))
    mu_d, sg = float(lr.mean()), float(lr.std())
    if sg <= 0:
        return float(c * 0.05)
    rng   = np.random.default_rng(42)
    Z     = rng.standard_normal((n_paths, horizon))
    final = S0 * np.exp(np.sum((mu_d - 0.5*sg**2) + sg*Z, axis=1))
    Sm, Sn = float(final.max()), float(final.min())
    S_T = (S0 + (Sm - S0)*c if c > 0.01
           else S0 - (S0 - Sn)*abs(c) if c < -0.01 else S0)
    return float(np.log(max(S_T, 1e-6) / S0))


def _bl_posterior(mu_arr, cov_arr, P, q, confidence,
                  tau=BL_TAU, delta=BL_RISK_AVERSION) -> np.ndarray:
    n     = len(mu_arr)
    Pi    = delta * cov_arr @ np.full(n, 1.0/n)
    if len(q) == 0:
        return Pi
    k = len(q)
    Omega = np.zeros((k, k))
    for j in range(k):
        pj = P[j]
        cj = float(np.clip(confidence[j], 1e-6, 1-1e-6))
        Omega[j,j] = ((1 - cj) / cj) * float(pj @ cov_arr @ pj)
    tS_inv = np.linalg.inv(tau * cov_arr)
    O_inv  = np.linalg.inv(Omega)
    A      = tS_inv + P.T @ O_inv @ P
    b      = tS_inv @ Pi + P.T @ O_inv @ q
    return np.linalg.inv(A) @ b


def run_bl_finbert(
    raw_df: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    prices: pd.DataFrame,
    test_start_date: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Colasanto et al. (2022):
    One-shot portfolio construction using BL views from FinBERT sentiment.
    Fixed weights held with HORIZON_DAYS rebalancing.
    """
    print("\n" + "=" * 60)
    print("METHOD 4 — BL + FinBERT  (Colasanto et al., 2022)")
    print("=" * 60)
    test_df = raw_df.copy()
    test_df["news_date_dt"] = pd.to_datetime(test_df["news_date"], errors="coerce")
    test_df = test_df[test_df["news_date_dt"] >= test_start_date].copy()
    universe = list(mu.index)
    mu_arr   = mu.values.copy().astype(float)
    cov_arr  = cov.values.copy().astype(float)

    # Sentiment scores from test period articles
    c_map = {t: _bl_c(t, test_df) for t in universe}
    print(f"Sentiment scores: { {t: f'{c_map[t]:.3f}' for t in universe} }")

    # Monte Carlo views
    q_list, P_rows, conf_list = [], [], []
    for i, t in enumerate(universe):
        c = c_map[t]
        if abs(c) < 0.01:
            continue
        y     = _mc_yield(t, c)
        p_row = np.zeros(len(universe)); p_row[i] = 1.0
        conf  = float(np.clip(0.5 + abs(c)*0.4, 0.5, 0.9))
        q_list.append(y); P_rows.append(p_row); conf_list.append(conf)
        print(f"  {t}: c={c:.3f}, view={y:.4f}, conf={conf:.2f}")

    # BL posterior
    if q_list:
        E_R = _bl_posterior(mu_arr, cov_arr, np.array(P_rows),
                            np.array(q_list), np.array(conf_list))
    else:
        print("  No views — using market equilibrium")
        E_R = mu_arr.copy()

    print(f"BL delta: { {t: f'{E_R[i]-mu_arr[i]:+.4f}' for i,t in enumerate(universe)} }")

    # MVO with BL posterior
    try:
        w_arr = _mvo_maxsharpe_local(E_R, cov_arr)
        w     = {t: float(w_arr[i]) for i,t in enumerate(universe) if float(w_arr[i]) > 1e-6}
    except Exception as e:
        return {"ok": False, "reason": f"BL MVO failed: {e}"}
    print(f"BL weights: { {t: f'{v:.1%}' for t,v in w.items()} }")

    # Apply to test period
    price_returns = prices.pct_change().dropna(how="all")
    rebal_dates   = _rebalance_dates(price_returns.index, test_start_date)
    all_ret       = []
    for i, rd in enumerate(rebal_dates):
        end = rebal_dates[i+1] if i+1 < len(rebal_dates) else price_returns.index[-1]
        all_ret.append(_compute_period_returns(w, price_returns, rd, end))
    if not all_ret:
        return {"ok": False, "reason": "No periods."}
    ret_s = pd.concat(all_ret).sort_index()
    ret_s = ret_s[~ret_s.index.duplicated(keep="first")]
    m = _portfolio_metrics(ret_s)
    print(f"Trading days: {len(ret_s)}")
    print(f"Sharpe={m.get('sharpe_ratio',float('nan')):.4f}  "
          f"Return={m.get('annualised_return',float('nan'))*100:.2f}%  "
          f"Vol={m.get('annualised_volatility',float('nan'))*100:.2f}%")
    return {
        "ok": True, "method": "4. BL + FinBERT (Colasanto 2022)",
        "description": (f"FinBERT→MC GBM→BL views→MVO. "
                        f"One-shot, rebalanced every {REBALANCE_EVERY}d."),
        "paper": "Colasanto et al. (2022) Neural Computing and Applications",
        "metrics": m,
        "daily_returns": ret_s,
        "returns_df": ret_s.to_frame("portfolio_return"),
        "test_start_date": str(test_start_date.date()),
        "weights": w,
        "sentiment_scores": c_map,
        "n_views": len(q_list),
    }


# ════════════════════════════════════════════════════════════════════════════
# METHOD 5 — NC-MVO [OURS]
# ════════════════════════════════════════════════════════════════════════════

def run_ncmvo(
    dataset: pd.DataFrame,
    model_bundle: Dict[str, Any],
    mu: pd.Series,
    cov: pd.DataFrame,
    prices: pd.DataFrame,
    test_start_date: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Our pipeline — exactly as in node_optimize_prediction_constraint():

      At each rebalancing date:
        1. Get latest signals from most recent test data up to that date
           (same as save_latest_ticker_prediction_signals: tail(1) per ticker)
        2. Baseline MVO via optimization_agent_from_mu_cov()
        3. Constraints via build_news_probability_constraints()
           (bull>=0.60 -> min_weight, bear<=0.40 -> max_weight)
        4. Constrained MVO via prediction_constrained_optimization_agent()
        5. Hold for HORIZON_DAYS

    mu and cov are computed from the FULL price history (build_returns_yahoo.py).
    They are FIXED — constraints only affect feasible allocations, not mu/cov.
    """
    print("\n" + "=" * 60)
    print("METHOD 5 — NC-MVO  [Ours]")
    print("=" * 60)

    model        = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    # Prepare test frame (same as prepare_model_frame in production)
    df, _ = prepare_model_frame(dataset, use_ticker_features=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    test_df  = df[df["news_date_dt"] >= test_start_date].copy()
    if test_df.empty:
        return {"ok": False, "reason": "No test data after split date."}

    universe      = list(mu.index)
    price_returns = prices.pct_change().dropna(how="all")
    rebal_dates   = _rebalance_dates(price_returns.index, test_start_date)

    if not rebal_dates:
        return {"ok": False, "reason": "No rebalancing dates."}

    print(f"Test: {test_start_date.date()} → {test_df['news_date_dt'].max().date()}")
    print(f"Rebalancing periods: {len(rebal_dates)} × {REBALANCE_EVERY} days")

    # Baseline MVO (no constraints) — computed ONCE from historical mu/cov
    # This is the same as the baseline in node_optimize_prediction_constraint
    try:
        baseline_res = optimization_agent_from_mu_cov(
            mu=mu, cov=cov, rf=RF_ANNUAL, w_max=W_MAX, lambda_l2=LAMBDA_L2
        )
        baseline_weights = baseline_res.get("maxsharpe", {}).get("weights", {}) or {}
    except Exception as e:
        return {"ok": False, "reason": f"Baseline MVO failed: {e}"}

    print(f"Baseline top5: {sorted(baseline_weights.items(), key=lambda x:x[1], reverse=True)[:5]}")

    all_ret   = []
    log_rows  = []

    for i, rd in enumerate(rebal_dates):
        end = rebal_dates[i+1] if i+1 < len(rebal_dates) else price_returns.index[-1]

        # Latest signals up to this rebalancing date
        avail = test_df[test_df["news_date_dt"] <= rd]

        if avail.empty:
            w = {t: float(v) for t,v in baseline_weights.items() if float(v) > 1e-6}
            print(f"  {rd.date()}: no signals yet → baseline")
        else:
            # tail(1) per ticker = latest signal per ticker
            # (same as save_latest_ticker_prediction_signals)
            latest = (
                avail.sort_values("news_date_dt")
                .groupby("ticker").tail(1).copy()
            )
            ticker_here = [t for t in latest["ticker"].tolist() if t in universe]

            if not ticker_here:
                w = {t: float(v) for t,v in baseline_weights.items() if float(v) > 1e-6}
            else:
                # Predict probabilities
                sub = latest[latest["ticker"].isin(ticker_here)]
                X   = sub[feature_cols].astype(float)
                proba = model.predict_proba(X)[:, 1]

                latest_signals_df = sub[["ticker"]].copy()
                latest_signals_df["predicted_positive_probability"] = proba

                # Build constraints — exactly like news_constraint_integration.py
                news_constraints = build_news_probability_constraints(
                    latest_signals=latest_signals_df,
                    baseline_weights=baseline_weights,
                    bullish_threshold=BULLISH_THRESHOLD,
                    bearish_threshold=BEARISH_THRESHOLD,
                    delta=DELTA,
                    w_max=W_MAX,
                )

                # Constrained MVO — same call as in portfolio_langgraph_withllm.py
                try:
                    con_res = prediction_constrained_optimization_agent(
                        mu=mu, cov=cov,
                        news_constraints=news_constraints,
                        rf=RF_ANNUAL, w_max=W_MAX, lambda_l2=LAMBDA_L2,
                    )
                    w_dict = con_res.get("maxsharpe", {}).get("weights", {}) or {}
                    w = {t: float(v) for t,v in w_dict.items() if float(v) > 1e-6}
                except Exception as e:
                    print(f"  [WARN] Constrained MVO failed {rd.date()}: {e}")
                    w = {t: float(v) for t,v in baseline_weights.items()
                         if float(v) > 1e-6}

                nb = sum(1 for c in news_constraints.values() if c.get("type") == "bullish")
                nr = sum(1 for c in news_constraints.values() if c.get("type") == "bearish")
                print(f"  {rd.date()}: bull={nb} bear={nr} active={len(w)}")
                log_rows.append({"date": rd, "n_bullish": nb, "n_bearish": nr,
                                 "n_active": len(w)})

        period_ret = _compute_period_returns(w, price_returns, rd, end)
        all_ret.append(period_ret)

    if not all_ret:
        return {"ok": False, "reason": "No valid periods."}

    ret_s = pd.concat(all_ret).sort_index()
    ret_s = ret_s[~ret_s.index.duplicated(keep="first")]
    m = _portfolio_metrics(ret_s)
    print(f"\nTotal trading days: {len(ret_s)}")
    print(f"Sharpe={m.get('sharpe_ratio',float('nan')):.4f}  "
          f"Return={m.get('annualised_return',float('nan'))*100:.2f}%  "
          f"Vol={m.get('annualised_volatility',float('nan'))*100:.2f}%")
    return {
        "ok": True, "method": "5. NC-MVO [Ours]",
        "description": (
            f"FinBERT→LogReg (bull≥{BULLISH_THRESHOLD}, bear≤{BEARISH_THRESHOLD}, "
            f"δ={DELTA})→MVO. Rebalanced every {REBALANCE_EVERY}d."
        ),
        "metrics": m,
        "daily_returns": ret_s,
        "returns_df": ret_s.to_frame("portfolio_return"),
        "test_start_date": str(test_start_date.date()),
        "n_rebalances": len(rebal_dates),
        "rebalance_log": log_rows,
    }


# ════════════════════════════════════════════════════════════════════════════
# TABLE + PLOT
# ════════════════════════════════════════════════════════════════════════════

def build_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    metric_map = {
        "annualised_return":     "Annualised Return (%)",
        "annualised_volatility": "Annualised Volatility (%)",
        "sharpe_ratio":          "Sharpe Ratio",
        "max_drawdown":          "Max Drawdown (%)",
        "pct_profitable_days":   "% Profitable Days",
        "mean_daily_return":     "Mean Daily Return (%)",
        "n_trading_days":        "N Trading Days",
    }
    pct_keys = {"annualised_return","annualised_volatility",
                "max_drawdown","mean_daily_return","pct_profitable_days"}
    rows = {v: {} for v in metric_map.values()}
    for res in results:
        if not res.get("ok"):
            continue
        col = res["method"]
        m = res.get("metrics", {})
        for k, label in metric_map.items():
            v = m.get(k, np.nan)
            if k in pct_keys and v is not None and not np.isnan(float(v)):
                v = float(v) * 100
            rows[label][col] = (round(float(v), 4)
                                if v is not None and not np.isnan(float(v)) else "N/A")
    return pd.DataFrame(rows).T


def plot_cumulative_returns(results: List[Dict], save_path: Path) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.dates as mdates
        colors = ["#9E9E9E","#2196F3","#FF9800","#9C27B0","#F44336"]
        fig, ax = plt.subplots(figsize=(13,6))
        for i, res in enumerate(results):
            if not res.get("ok"): continue
            r = res.get("daily_returns")
            if r is None or r.empty: continue
            cum = (1 + r.dropna()).cumprod()
            lw  = 2.8 if "Ours" in res["method"] else 1.6
            ls  = "-"  if "Ours" in res["method"] else "--"
            ax.plot(cum.index, cum.values, label=res["method"],
                    color=colors[i % len(colors)], linewidth=lw, linestyle=ls)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(f"Cumulative Returns — Baseline Comparison "
                     f"(rebalanced every {REBALANCE_EVERY} trading days)", fontsize=13)
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate(); plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Plot saved: {save_path}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_unified_comparison(
    raw_path: str      = ALL_TICKERS_RAW_PATH,
    model_path: str    = str(BEST_MODEL_PATH),
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "═"*70)
    print("UNIFIED BASELINE COMPARISON")
    print(f"EW | Plain MVO | Zhang(2022) | BL+FinBERT(Colasanto2022) | NC-MVO[Ours]")
    print(f"Rebalancing: every {REBALANCE_EVERY} trading days (HORIZON_DAYS={HORIZON_DAYS})")
    print("═"*70)

    # ── Load raw data ─────────────────────────────────────────────────────
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    raw_df["news_date_dt"] = pd.to_datetime(raw_df["news_date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["news_date_dt","ticker"])
    raw_df["ticker"] = raw_df["ticker"].astype(str).str.upper().str.strip()
    print(f"\nRaw: {len(raw_df):,} rows | {raw_df['ticker'].nunique()} tickers | "
          f"{raw_df['news_date'].min()} → {raw_df['news_date'].max()}")

    # ── Build prediction dataset (SAME as training pipeline) ─────────────
    dataset = build_ticker_date_prediction_dataset_v2(
        raw_path=raw_path, min_abs_return_for_signal=MIN_ABS_RETURN
    )
    if dataset.empty:
        raise RuntimeError("Prediction dataset is empty.")
    dataset = dataset.sort_values("news_date_dt").reset_index(drop=True)
    split_idx       = int(len(dataset) * TRAIN_TEST_SPLIT)
    test_start_date = dataset.iloc[split_idx]["news_date_dt"]
    print(f"\nSplit 70/30 | Train: {split_idx} rows | Test: {len(dataset)-split_idx} rows")
    print(f"Test starts: {test_start_date.date()}")

    # ── Load mu/cov from build_returns_yahoo.py outputs ───────────────────
    all_tickers = raw_df["ticker"].dropna().unique().tolist()
    try:
        mu, cov = data_agent_get_mu_cov(all_tickers)
        print(f"\nmu/cov: {len(mu)} tickers")
    except Exception as e:
        raise RuntimeError(f"mu/cov load failed: {e}")

    # ── Load prices ───────────────────────────────────────────────────────
    prices = _load_daily_prices(list(mu.index))
    if prices.empty:
        raise RuntimeError("No price data.")
    print(f"Prices: {len(prices)} days × {len(prices.columns)} tickers")

    # ── Load model ────────────────────────────────────────────────────────
    model_bundle = None
    if Path(model_path).exists():
        try:
            model_bundle = load_prediction_model(model_path=model_path)
            roc = model_bundle["metrics"].get("roc_auc","N/A")
            ba  = model_bundle["metrics"].get("balanced_accuracy","N/A")
            print(f"\nModel loaded | ROC-AUC={roc} | Balanced-Acc={ba}")
        except Exception as e:
            print(f"[WARN] Model load failed: {e}")
    else:
        print(f"[WARN] Model not found: {model_path}")

    # ── Run all 5 methods ─────────────────────────────────────────────────
    results = []
    results.append(run_equal_weight(list(mu.index), prices, test_start_date))
    results.append(run_plain_mvo(mu, cov, prices, test_start_date))
    results.append(run_zhang_longshort(raw_df, prices, test_start_date))
    results.append(run_bl_finbert(raw_df, mu, cov, prices, test_start_date))
    if model_bundle is not None:
        results.append(run_ncmvo(dataset, model_bundle, mu, cov, prices, test_start_date))
    else:
        results.append({"ok": False, "method": "5. NC-MVO [Ours]",
                        "reason": "Model not available"})

    # ── Table ─────────────────────────────────────────────────────────────
    print("\n\n" + "═"*70)
    print("COMPARISON TABLE")
    print("═"*70)
    table = build_comparison_table(results)
    print(table.to_string())

    print("\n" + "─"*78)
    print(f"{'Method':<38} {'Sharpe':>7} {'Ret%':>8} {'Vol%':>8} {'MaxDD%':>8}")
    print("─"*78)
    for res in results:
        m  = res.get("metrics",{})
        ok = "✓" if res.get("ok") else "✗"
        sh = m.get("sharpe_ratio", float("nan"))
        re = m.get("annualised_return", float("nan"))
        vo = m.get("annualised_volatility", float("nan"))
        md = m.get("max_drawdown", float("nan"))
        print(f"{ok} {res.get('method','?'):<36} "
              f"{sh:>7.3f} "
              f"{re*100 if not np.isnan(re) else float('nan'):>8.2f} "
              f"{vo*100 if not np.isnan(vo) else float('nan'):>8.2f} "
              f"{md*100 if not np.isnan(md) else float('nan'):>8.2f}")
    print("─"*78)

    # ── Save ──────────────────────────────────────────────────────────────
    if save_outputs:
        table.to_csv(OUT_DIR / "comparison_table.csv")
        print(f"\nSaved: {OUT_DIR / 'comparison_table.csv'}")
        metrics_out = {}
        for res in results:
            key = res.get("method","?").replace(" ","_")
            metrics_out[key] = {
                "ok": res.get("ok",False),
                "method": res.get("method",""),
                "description": res.get("description",""),
                "paper": res.get("paper","this_work"),
                "test_start_date": res.get("test_start_date",""),
                "trading_days": res.get("metrics",{}).get("n_trading_days",0),
                "metrics": res.get("metrics",{}),
            }
            if res.get("ok") and res.get("weights"):
                safe = key.replace("/","_")
                with open(OUT_DIR / f"weights_{safe}.json","w") as f:
                    json.dump(res["weights"], f, indent=2)
        with open(OUT_DIR / "comparison_results.json","w",encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {OUT_DIR / 'comparison_results.json'}")
        plot_cumulative_returns(results, OUT_DIR / "cumulative_returns.png")
        for res in results:
            if not res.get("ok"): continue
            rdf = res.get("returns_df")
            if rdf is not None:
                safe = (res["method"].replace(" ","_").replace(".","")
                        .replace("[","").replace("]","").replace("/","_"))
                rdf.to_csv(OUT_DIR / f"daily_returns_{safe}.csv")

    return {
        "results": results,
        "comparison_table": table,
        "test_start_date": str(test_start_date.date()),
        "n_train": split_idx,
        "n_test": len(dataset) - split_idx,
    }


if __name__ == "__main__":
    run_unified_comparison(
        raw_path=ALL_TICKERS_RAW_PATH,
        model_path=str(BEST_MODEL_PATH),
        save_outputs=True,
    )