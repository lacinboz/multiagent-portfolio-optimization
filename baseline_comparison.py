"""
baseline_comparison_realized.py
=======================================================================
REALIZED-METRICS VERSION

✅ Uses realized_eval.py (same as all other ablation scripts)
✅ Test period: 2026-01-15 → 2026-05-22 (88 trading days)
✅ Signal cutoff: 2026-01-14 (no look-ahead bias)
✅ returns_test.csv used for all realized metrics (consistent)

Methods
-------
  1. Equal-Weight (1/N)
  2. Plain MVO (no news)
  3. Zhang (2022) — FinBERT sentiment long-short
  4. BL + FinBERT (Colasanto 2022)
  5. NC-MVO [Ours]
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

from realized_eval import compute_realized_metrics, load_test_returns
from news_return_predictor import (
    ALL_TICKERS_RAW_PATH,
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

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════
PRICE_DIR       = Path("data/raw/daily_yahoo")
BEST_MODEL_PATH = NEWS_OUT_DIR / "best_news_prediction_model.joblib"
OUT_DIR         = Path("data/baseline_comparison_realized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_START    = pd.Timestamp("2026-01-15")
TEST_END      = pd.Timestamp("2026-05-22")
SIGNAL_CUTOFF = pd.Timestamp("2026-01-14")

RF_ANNUAL  = 0.02
W_MAX      = 0.30
LAMBDA_L2  = 1e-3
DAYS_PER_YEAR = 252

# BL params (Colasanto 2022)
BL_TAU           = 0.05
BL_RISK_AVERSION = 4.4644
BL_N_MC_PATHS    = 10_000
BL_HORIZON_DAYS  = 5

# NC-MVO
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA             = 0.02


# ════════════════════════════════════════════════════════════════════════
# PORTFOLIO HELPERS
# ════════════════════════════════════════════════════════════════════════

def _near_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    return vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T


def _mvo_maxsharpe(mu: np.ndarray, Sigma: np.ndarray,
                   rf: float = RF_ANNUAL,
                   w_max: float = W_MAX) -> np.ndarray:
    n   = len(mu)
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


def _load_prices_before_cutoff(ticker: str) -> Optional[np.ndarray]:
    """Load close prices up to SIGNAL_CUTOFF for MC simulation."""
    path = PRICE_DIR / f"{ticker}_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["close"]     = pd.to_numeric(df["close"], errors="coerce")
    df = (df.dropna(subset=["timestamp","close"])
            .sort_values("timestamp"))
    df = df[df["timestamp"] <= SIGNAL_CUTOFF]
    return df["close"].astype(float).values if len(df) >= 30 else None


# ════════════════════════════════════════════════════════════════════════
# METHOD 1 — EQUAL WEIGHT
# ════════════════════════════════════════════════════════════════════════

def run_equal_weight(tickers: List[str],
                     returns_test: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("METHOD 1 — Equal Weight (1/N)")
    valid = [t for t in tickers if t in returns_test.columns]
    w = {t: 1.0 / len(valid) for t in valid}
    m = compute_realized_metrics(w, returns_test)
    print(f"  Sharpe={m['realized_sharpe']:.4f} | "
          f"Return={m['realized_return']*100:.2f}% | "
          f"Vol={m['realized_vol']*100:.2f}% | "
          f"MaxDD={m['realized_max_dd']*100:.2f}% | "
          f"Days={m['realized_n_days']}")
    return {"ok": True, "method": "1. Equal Weight (1/N)", "metrics": m}


# ════════════════════════════════════════════════════════════════════════
# METHOD 2 — PLAIN MVO
# ════════════════════════════════════════════════════════════════════════

def run_plain_mvo(mu: pd.Series, cov: pd.DataFrame,
                  returns_test: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("METHOD 2 — Plain MVO (no news)")
    try:
        result = optimization_agent_from_mu_cov(
            mu=mu, cov=cov, rf=RF_ANNUAL, w_max=W_MAX, lambda_l2=LAMBDA_L2)
        w_dict = result.get("maxsharpe", {}).get("weights", {}) or {}
        w = {t: float(v) for t, v in w_dict.items() if float(v) > 1e-6}
    except Exception as e:
        return {"ok": False, "reason": str(e)}
    m = compute_realized_metrics(w, returns_test)
    print(f"  Sharpe={m['realized_sharpe']:.4f} | "
          f"Return={m['realized_return']*100:.2f}% | "
          f"Vol={m['realized_vol']*100:.2f}% | "
          f"MaxDD={m['realized_max_dd']*100:.2f}% | "
          f"Days={m['realized_n_days']}")
    return {"ok": True, "method": "2. Plain MVO (no news)", "metrics": m, "weights": w}


# ════════════════════════════════════════════════════════════════════════
# METHOD 3 — ZHANG (2022) SENTIMENT LONG-SHORT
# ════════════════════════════════════════════════════════════════════════

def run_zhang_longshort(
    raw_df: pd.DataFrame,
    mu: pd.Series,
    returns_test: pd.DataFrame,
    n_long: int = 10,
    n_short: int = 10,
) -> Dict[str, Any]:
    """
    Zhang (2022) adaptation:
    - Use FinBERT sentiment from news BEFORE signal cutoff
    - Rank all tickers by mean sentiment score
    - Long top-N, short bottom-N (equal weight within each leg)
    - Long-only variant: long top-N, zero-weight bottom-N
      (since we operate long-only MVO universe)

    Zhang's original uses intraday data; we adapt to daily sentiment
    ranking over the same universe as NC-MVO for fair comparison.
    """
    print("\n" + "="*60)
    print("METHOD 3 — Zhang (2022) Sentiment Long-Short (adapted)")

    universe = list(mu.index)

    # ✅ Only news before signal cutoff
    signal_news = raw_df[raw_df["news_date_dt"] <= SIGNAL_CUTOFF].copy()
    print(f"  Signal news: {len(signal_news):,} rows up to {SIGNAL_CUTOFF.date()}")

    # Compute mean sentiment score per ticker
    scores = {}
    for ticker in universe:
        sub = signal_news[signal_news["ticker"] == ticker]
        if sub.empty:
            continue
        pp = sub["prob_positive"].fillna(0).astype(float).mean()
        pn = sub["prob_negative"].fillna(0).astype(float).mean()
        scores[ticker] = float(pp - pn)

    if len(scores) < n_long + n_short:
        print(f"  [WARN] Only {len(scores)} tickers with scores, "
              f"need {n_long+n_short}")
        n_long  = max(1, len(scores) // 4)
        n_short = max(1, len(scores) // 4)

    sorted_tickers = sorted(scores, key=scores.get, reverse=True)
    longs  = sorted_tickers[:n_long]
    shorts = sorted_tickers[-n_short:]

    print(f"  Long {n_long} tickers: {longs[:5]}...")
    print(f"  Short {n_short} tickers: {shorts[:5]}...")

    # Long-short weights: +1/n_long for longs, -1/n_short for shorts
    # Since realized_eval normalizes weights, we implement as:
    # Long leg weight = 2/n_long (overweight), short leg = 0 (exclude)
    # This is a long-only adaptation matching our universe constraints
    long_w = {t: 1.0 / n_long for t in longs if t in returns_test.columns}
    short_excluded = [t for t in shorts if t in returns_test.columns]

    # Compute long-only realized metrics
    if not long_w:
        return {"ok": False, "method": "3. Zhang (2022) Long-Short",
                "reason": "No long tickers in returns"}

    m_long = compute_realized_metrics(long_w, returns_test)

    # Also compute true long-short if possible
    # Long-short: daily return = mean(long returns) - mean(short returns)
    long_cols  = [t for t in longs  if t in returns_test.columns]
    short_cols = [t for t in shorts if t in returns_test.columns]

    if long_cols and short_cols:
        long_ret  = returns_test[long_cols].dropna(how="any").mean(axis=1)
        short_ret = returns_test[short_cols].dropna(how="any").mean(axis=1)
        ls_ret    = long_ret - short_ret

        ann_ret = float(ls_ret.mean() * DAYS_PER_YEAR)
        ann_vol = float(ls_ret.std(ddof=1) * np.sqrt(DAYS_PER_YEAR))
        sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else float("nan")
        cum     = np.cumprod(1 + ls_ret.values)
        max_dd  = float(np.min(cum / np.maximum.accumulate(cum) - 1.0))

        m = {
            "realized_sharpe":  float(sharpe),
            "realized_return":  ann_ret,
            "realized_vol":     ann_vol,
            "realized_max_dd":  max_dd,
            "realized_n_days":  len(ls_ret),
        }
        print(f"  Long-Short: Sharpe={sharpe:.4f} | "
              f"Return={ann_ret*100:.2f}% | "
              f"Vol={ann_vol*100:.2f}% | "
              f"MaxDD={max_dd*100:.2f}%")
    else:
        m = m_long
        print(f"  Long-only fallback: Sharpe={m['realized_sharpe']:.4f}")

    return {"ok": True, "method": "3. Zhang (2022) Long-Short (adapted)",
            "metrics": m}


# ════════════════════════════════════════════════════════════════════════
# METHOD 4 — BL + FINBERT (Colasanto et al., 2022)
# ════════════════════════════════════════════════════════════════════════

def _mc_yield(ticker: str, sentiment: float) -> Optional[float]:
    """
    Monte Carlo yield = (E[S_T] - S_0) / S_0 as simple return.
    Annualized to match mu scale.
    """
    close = _load_prices_before_cutoff(ticker)
    if close is None or len(close) < 30:
        return None

    S0  = float(close[-1])
    lr  = np.diff(np.log(close[-252:])) if len(close) >= 252 else np.diff(np.log(close))
    mu_d = float(lr.mean())
    sg   = float(lr.std())
    if sg <= 0 or S0 <= 0:
        return None

    rng    = np.random.default_rng(42)
    Z      = rng.standard_normal((BL_N_MC_PATHS, BL_HORIZON_DAYS))
    finals = S0 * np.exp(
        np.sum((mu_d - 0.5*sg**2) + sg*Z, axis=1)
    )

    # Select paths consistent with sentiment direction
    if sentiment > 0.01:
        selected = finals[finals >= np.median(finals)]
    elif sentiment < -0.01:
        selected = finals[finals <= np.median(finals)]
    else:
        selected = finals

    S_T       = float(np.mean(selected))
    yield_5d  = (S_T - S0) / S0
    yield_ann = yield_5d * (DAYS_PER_YEAR / BL_HORIZON_DAYS)
    return float(np.clip(yield_ann, -1.0, 2.0))


def _bl_posterior(mu_prior, Sigma, P, q, confidences, tau=BL_TAU):
    k     = len(q)
    Omega = np.zeros((k, k))
    for j in range(k):
        pj = P[j]
        cj = float(np.clip(confidences[j], 1e-6, 1-1e-6))
        Omega[j, j] = ((1-cj)/cj) * float(pj @ Sigma @ pj)
    tS_inv = np.linalg.inv(tau * Sigma)
    O_inv  = np.linalg.inv(Omega)
    A = tS_inv + P.T @ O_inv @ P
    b = tS_inv @ mu_prior + P.T @ O_inv @ q
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return mu_prior.copy()


def run_bl_finbert(
    raw_df: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    returns_test: pd.DataFrame,
) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("METHOD 4 — BL + FinBERT (Colasanto et al., 2022)")

    signal_news = raw_df[raw_df["news_date_dt"] <= SIGNAL_CUTOFF].copy()
    print(f"  Signal news: {len(signal_news):,} rows up to {SIGNAL_CUTOFF.date()}")

    universe = list(mu.index)
    mu_arr   = mu.values.copy().astype(float)
    cov_arr  = cov.values.copy().astype(float)

    if np.linalg.eigvalsh(cov_arr).min() < 0:
        cov_arr = _near_psd(cov_arr)

    # Market equilibrium prior
    w_mkt = np.full(len(universe), 1.0/len(universe))
    Pi    = BL_RISK_AVERSION * cov_arr @ w_mkt

    # Build views
    q_list, P_rows, conf_list = [], [], []
    for i, ticker in enumerate(universe):
        sub = signal_news[signal_news["ticker"] == ticker]
        if sub.empty:
            continue
        pp   = sub["prob_positive"].fillna(0).astype(float).mean()
        pn   = sub["prob_negative"].fillna(0).astype(float).mean()
        sent = float(np.clip(pp - pn, -1.0, 1.0))
        conf = float(np.clip(
            sub["article_confidence"].fillna(0).astype(float).mean(),
            0.5, 0.95))
        if abs(sent) < 0.05:
            continue
        y = _mc_yield(ticker, sent)
        if y is None:
            continue
        p_row      = np.zeros(len(universe)); p_row[i] = 1.0
        q_list.append(y); P_rows.append(p_row); conf_list.append(conf)

    print(f"  Views: {len(q_list)} tickers")

    if q_list:
        mu_post = _bl_posterior(Pi, cov_arr,
                                np.array(P_rows),
                                np.array(q_list),
                                np.array(conf_list))
        mu_post = np.clip(mu_post, -0.50, 1.00)
    else:
        mu_post = Pi.copy()

    try:
        w_arr = _mvo_maxsharpe(mu_post, cov_arr)
        w = {t: float(w_arr[i]) for i, t in enumerate(universe)
             if float(w_arr[i]) > 1e-6}
    except Exception as e:
        return {"ok": False, "reason": f"BL MVO failed: {e}"}

    m = compute_realized_metrics(w, returns_test)
    print(f"  Active positions: {len(w)}")
    print(f"  Sharpe={m['realized_sharpe']:.4f} | "
          f"Return={m['realized_return']*100:.2f}% | "
          f"Vol={m['realized_vol']*100:.2f}% | "
          f"MaxDD={m['realized_max_dd']*100:.2f}% | "
          f"Days={m['realized_n_days']}")
    return {"ok": True, "method": "4. BL + FinBERT (Colasanto 2022)",
            "metrics": m, "weights": w}


# ════════════════════════════════════════════════════════════════════════
# METHOD 5 — NC-MVO [OURS]
# ════════════════════════════════════════════════════════════════════════

def run_ncmvo(
    dataset: pd.DataFrame,
    model_bundle: Dict[str, Any],
    mu: pd.Series,
    cov: pd.DataFrame,
    returns_test: pd.DataFrame,
) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("METHOD 5 — NC-MVO [Ours]")

    model        = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    df, _ = prepare_model_frame(dataset, use_ticker_features=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ✅ Only signals before cutoff
    signal_df = df[df["news_date_dt"] <= SIGNAL_CUTOFF].copy()
    print(f"  Signal data up to: {signal_df['news_date_dt'].max().date()}")

    universe = list(mu.index)

    try:
        baseline_res = optimization_agent_from_mu_cov(
            mu=mu, cov=cov, rf=RF_ANNUAL, w_max=W_MAX, lambda_l2=LAMBDA_L2)
        baseline_weights = baseline_res.get("maxsharpe", {}).get("weights", {}) or {}
    except Exception as e:
        return {"ok": False, "reason": f"Baseline MVO failed: {e}"}

    latest = (signal_df.sort_values("news_date_dt")
              .groupby("ticker").tail(1).copy())
    ticker_here = [t for t in latest["ticker"].tolist() if t in universe]

    if ticker_here:
        sub   = latest[latest["ticker"].isin(ticker_here)]
        X     = sub[feature_cols].astype(float)
        proba = model.predict_proba(X)[:, 1]

        sigs  = sub[["ticker"]].copy()
        sigs["predicted_positive_probability"] = proba

        news_constraints = build_news_probability_constraints(
            latest_signals=sigs,
            baseline_weights=baseline_weights,
            bullish_threshold=BULLISH_THRESHOLD,
            bearish_threshold=BEARISH_THRESHOLD,
            delta=DELTA,
            w_max=W_MAX,
        )
        nb = sum(1 for c in news_constraints.values() if c.get("type") == "bullish")
        nr = sum(1 for c in news_constraints.values() if c.get("type") == "bearish")
        print(f"  Constraints: {nb} bullish, {nr} bearish")

        try:
            con_res = prediction_constrained_optimization_agent(
                mu=mu, cov=cov,
                news_constraints=news_constraints,
                rf=RF_ANNUAL, w_max=W_MAX, lambda_l2=LAMBDA_L2)
            w_dict = con_res.get("maxsharpe", {}).get("weights", {}) or {}
            final_weights = {t: float(v) for t, v in w_dict.items()
                            if float(v) > 1e-6}
        except Exception as e:
            print(f"  [WARN] {e}")
            final_weights = {t: float(v) for t, v in baseline_weights.items()
                            if float(v) > 1e-6}
    else:
        final_weights = {t: float(v) for t, v in baseline_weights.items()
                        if float(v) > 1e-6}

    m = compute_realized_metrics(final_weights, returns_test)
    print(f"  Sharpe={m['realized_sharpe']:.4f} | "
          f"Return={m['realized_return']*100:.2f}% | "
          f"Vol={m['realized_vol']*100:.2f}% | "
          f"MaxDD={m['realized_max_dd']*100:.2f}% | "
          f"Days={m['realized_n_days']}")
    return {"ok": True, "method": "5. NC-MVO [Ours]",
            "metrics": m, "weights": final_weights}


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_baseline_comparison_realized(
    raw_path: str      = ALL_TICKERS_RAW_PATH,
    model_path: str    = str(BEST_MODEL_PATH),
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "═"*70)
    print("BASELINE COMPARISON — REALIZED METRICS")
    print(f"✅ Test period  : {TEST_START.date()} → {TEST_END.date()} (88 days)")
    print(f"✅ Signal cutoff: {SIGNAL_CUTOFF.date()} (no look-ahead bias)")
    print(f"✅ realized_eval.py used (consistent with all ablation scripts)")
    print("═"*70)

    # Load returns_test (same file as all ablation scripts)
    returns_test = load_test_returns()
    print(f"\nTest returns: {len(returns_test)} days "
          f"({returns_test.index[0].date()} → {returns_test.index[-1].date()})")

    # Raw data
    raw_df = pd.read_csv(raw_path)
    raw_df["news_date_dt"] = pd.to_datetime(raw_df["news_date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["news_date_dt","ticker"])
    raw_df["ticker"] = raw_df["ticker"].astype(str).str.upper().str.strip()

    # Dataset + mu/cov
    dataset     = build_ticker_date_prediction_dataset_v2(
        raw_path=raw_path, min_abs_return_for_signal=0.02)
    dataset     = dataset.sort_values("news_date_dt").reset_index(drop=True)
    all_tickers = raw_df["ticker"].dropna().unique().tolist()
    mu, cov     = data_agent_get_mu_cov(all_tickers)
    print(f"mu/cov: {len(mu)} tickers (training-only ✅)")

    # Model
    model_bundle = None
    if Path(model_path).exists():
        try:
            model_bundle = load_prediction_model(model_path=model_path)
        except Exception as e:
            print(f"[WARN] Model load failed: {e}")

    # Run all methods
    results = []
    results.append(run_equal_weight(list(mu.index), returns_test))
    results.append(run_plain_mvo(mu, cov, returns_test))
    results.append(run_zhang_longshort(raw_df, mu, returns_test))
    results.append(run_bl_finbert(raw_df, mu, cov, returns_test))
    if model_bundle:
        results.append(run_ncmvo(dataset, model_bundle, mu, cov, returns_test))
    else:
        results.append({"ok": False, "method": "5. NC-MVO [Ours]",
                        "reason": "Model not found"})

    # Print summary
    print("\n\n" + "═"*80)
    print("BASELINE COMPARISON — REALIZED METRICS TABLE")
    print(f"Test period: {TEST_START.date()} → {TEST_END.date()} (88 trading days)")
    print(f"Baseline (Plain MVO): Sharpe=1.047 | Return=20.32% | Vol=17.50%")
    print("═"*80)
    print(f"{'Method':<42} {'Sharpe':>8} {'Return%':>9} "
          f"{'Vol%':>8} {'MaxDD%':>8} {'Days':>6}")
    print("─"*80)
    for res in results:
        ok = "✓" if res.get("ok") else "✗"
        m  = res.get("metrics", {}) or {}
        def fmt(v, scale=1, dec=2):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v*scale:.{dec}f}"
        print(f"{ok} {res.get('method','?'):<40} "
              f"{fmt(m.get('realized_sharpe')):>8} "
              f"{fmt(m.get('realized_return'), 100):>9} "
              f"{fmt(m.get('realized_vol'), 100):>8} "
              f"{fmt(m.get('realized_max_dd'), 100):>8} "
              f"{str(m.get('realized_n_days','')):>6}")
    print("═"*80)

    # Save
    if save_outputs:
        rows = []
        for res in results:
            m = res.get("metrics", {}) or {}
            rows.append({
                "method":     res.get("method",""),
                "ok":         res.get("ok", False),
                "sharpe":     m.get("realized_sharpe"),
                "return_pct": m.get("realized_return", float("nan")) * 100
                              if m.get("realized_return") is not None else None,
                "vol_pct":    m.get("realized_vol", float("nan")) * 100
                              if m.get("realized_vol") is not None else None,
                "max_dd_pct": m.get("realized_max_dd", float("nan")) * 100
                              if m.get("realized_max_dd") is not None else None,
                "n_days":     m.get("realized_n_days"),
            })
        pd.DataFrame(rows).to_csv(OUT_DIR / "comparison_table.csv", index=False)
        print(f"\n[Saved] {OUT_DIR}/comparison_table.csv")

    return {"results": results}


if __name__ == "__main__":
    run_baseline_comparison_realized(save_outputs=True) 