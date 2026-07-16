# news_ablation_realized_study.py
# ============================================================
# Hocaın istediği ablation:
# "News koyduk çünkü daha iyi oldu" — bunu realized metriklerle göster.
#
# Mode B: newssiz baseline vs prediction-constrained (newsli)
# Mode A: newssiz baseline vs FinBERT-adjusted (newsli)
#
# Her iki mod için aynı test dönemi, aynı 101 ticker,
# aynı fiyat verileri kullanılır.
#
# Kullanım:
#   python news_ablation_realized_study.py
#
# Gerekli dosyalar:
#   data/news_prediction/best_news_prediction_model.joblib
#   data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv
#   data/processed_yahoo/summary_per_asset_annual.csv
#   data/processed_yahoo/cov_annual.csv
#   data/raw/daily_yahoo/<TICKER>_daily.csv
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────
MU_PATH       = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH      = Path("data/processed_yahoo/cov_annual.csv")
PRICE_DIR     = Path("data/raw/daily_yahoo")
RAW_PATH      = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
MODEL_PATH    = Path("data/news_prediction/best_news_prediction_model.joblib")
OUT_DIR       = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── parameters ───────────────────────────────────────────────
RF                = 0.02
W_MAX             = 0.30
LAMBDA_L2         = 1e-3
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA             = 0.02
ALPHA             = 0.08   # Mode A return adjustment
BETA_COV          = 0.35   # Mode A covariance adjustment

TEST_START = "2026-01-15"
TEST_END   = "2026-05-22"
AS_OF_DATE = "2026-01-15"   # sinyaller bu tarihten önceki verilerle üretilir


# ─────────────────────────────────────────────────────────────
# Portfolio helpers
# ─────────────────────────────────────────────────────────────

def _near_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    return vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T


def _load_mu_cov(tickers: Optional[List[str]] = None):
    summary = pd.read_csv(MU_PATH, index_col=0)
    cov_df  = pd.read_csv(COV_PATH, index_col=0)
    if tickers is None:
        common = [t for t in summary.index if t in cov_df.index]
    else:
        common = [t for t in tickers if t in summary.index and t in cov_df.index]
    mu  = summary.loc[common, "mu_annual"].astype(float)
    cov = cov_df.loc[common, common].astype(float)
    return mu, cov


def _optimize(mu, cov, rf, w_max, lambda_l2, extra_constraints=None):
    tickers = list(mu.index)
    n = len(tickers)
    eff_wmax = max(w_max, 1.0 / n + 1e-6)
    cov_np = cov.values.copy()
    if np.linalg.eigvalsh(cov_np).min() < 0:
        cov_np = _near_psd(cov_np)
    cov_f = pd.DataFrame(cov_np, index=tickers, columns=tickers)

    bounds = [(0.0, eff_wmax)] * n
    cons   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        cons += extra_constraints
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov_f.values @ w))
        return -(r - rf) / v if v > 0 else np.inf

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(neg_sharpe, w0, method="trust-constr", bounds=bounds, constraints=cons)

    w = pd.Series(np.clip(res.x, 0, None), index=tickers)
    w = w / w.sum()
    return {t: float(w[t]) for t in tickers}


# ─────────────────────────────────────────────────────────────
# Price data
# ─────────────────────────────────────────────────────────────

def _load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        path = PRICE_DIR / f"{t}_daily.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df.index = df.index.tz_localize(None)
            frames[t] = df["close"].astype(float)
        except Exception:
            pass
    if not frames:
        raise RuntimeError(f"No price data found in {PRICE_DIR}")
    prices = pd.DataFrame(frames).loc[start:end].dropna(axis=1, how="all")
    return prices


def _realized_metrics(weights: Dict[str, float], prices: pd.DataFrame, rf: float) -> Dict[str, float]:
    common = [t for t in weights if t in prices.columns and weights[t] > 1e-8]
    if not common:
        return {"sharpe": float("nan"), "return": float("nan"), "vol": float("nan"), "max_dd": float("nan")}

    w = np.array([weights[t] for t in common])
    w = w / w.sum()

    px = prices[common].ffill().dropna()
    daily_ret = px.pct_change().dropna()
    port_ret  = daily_ret.values @ w

    ann_ret = float(np.mean(port_ret) * 252)
    ann_vol = float(np.std(port_ret, ddof=1) * np.sqrt(252))
    sharpe  = float((ann_ret - rf) / ann_vol) if ann_vol > 0 else float("nan")

    cum = np.cumprod(1 + port_ret)
    max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1.0))

    return {"sharpe": sharpe, "return": ann_ret, "vol": ann_vol, "max_dd": max_dd}


# ─────────────────────────────────────────────────────────────
# Mode B: prediction-constrained
# ─────────────────────────────────────────────────────────────

def _build_mode_b_dataset_as_of(raw_path: str, as_of_date: str) -> pd.DataFrame:
    """
    Raw dataset'ten as_of_date öncesindeki verileri kullanarak
    ticker-date seviyesinde feature dataset oluşturur.
    news_return_predictor.py'daki build_ticker_date_prediction_dataset_v2
    ile aynı mantık.
    """
    df = pd.read_csv(raw_path)
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt", "ticker", "future_return",
        "article_sentiment", "article_confidence",
        "prob_positive", "prob_negative", "prob_neutral",
        "combined_weight", "past_5d_return", "past_20d_return", "past_20d_volatility",
    ]).copy()

    # Sadece as_of_date öncesi veriler — look-ahead bias yok
    as_of = pd.to_datetime(as_of_date)
    df = df[df["news_date_dt"] < as_of].copy()

    if df.empty:
        raise ValueError(f"No data before {as_of_date} in raw dataset")

    df = df[df["future_return"].abs() >= 0.02].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    df["is_positive_article"]    = (df["prob_positive"] > df["prob_negative"]).astype(int)
    df["is_negative_article"]    = (df["prob_negative"] > df["prob_positive"]).astype(int)
    df["sentiment_confidence"]   = df["article_sentiment"] * df["article_confidence"]

    grouped_rows = []
    for (ticker, news_date, news_date_dt), g in df.groupby(["ticker", "news_date", "news_date_dt"]):
        weights = g["article_confidence"].astype(float)
        w_sum   = float(weights.sum())
        def wmean(col):
            vals = g[col].astype(float)
            return float(np.average(vals, weights=weights)) if w_sum > 0 else float(vals.mean())
        grouped_rows.append({
            "ticker":                   ticker,
            "news_date":                news_date,
            "news_date_dt":             news_date_dt,
            "article_count":            int(len(g)),
            "weighted_sentiment":       wmean("article_sentiment"),
            "sentiment_std":            float(g["article_sentiment"].std()) if len(g) > 1 else 0.0,
            "mean_confidence":          float(g["article_confidence"].mean()),
            "positive_ratio":           float(g["is_positive_article"].mean()),
            "negative_ratio":           float(g["is_negative_article"].mean()),
            "mean_sentiment_confidence": float(g["sentiment_confidence"].mean()),
            "past_5d_return":           float(g["past_5d_return"].mean()),
            "past_20d_return":          float(g["past_20d_return"].mean()),
            "past_20d_volatility":      float(g["past_20d_volatility"].mean()),
            "future_return":            float(g["future_return"].mean()),
        })

    out = pd.DataFrame(grouped_rows).sort_values(["ticker", "news_date_dt"]).reset_index(drop=True)

    flow_parts = []
    for ticker, g in out.groupby("ticker"):
        g = g.sort_values("news_date_dt").copy()
        for w in [5, 20]:
            g[f"sentiment_flow_{w}d"] = g["weighted_sentiment"].shift(1).rolling(w, min_periods=1).mean()
            g[f"confidence_flow_{w}d"] = g["mean_confidence"].shift(1).rolling(w, min_periods=1).mean()
        flow_parts.append(g)

    out = pd.concat(flow_parts, ignore_index=True).dropna().reset_index(drop=True)
    out["target_direction"] = (out["future_return"] > 0).astype(int)

    print(f"Mode B as-of dataset: {len(out)} rows, {out['ticker'].nunique()} tickers")
    print(f"  date range: {out['news_date_dt'].min().date()} → {out['news_date_dt'].max().date()}")
    return out


def _generate_mode_b_signals(model_bundle: dict, dataset: pd.DataFrame, tickers: List[str]) -> Dict[str, float]:
    """
    Eğitilmiş modeli kullanarak her ticker için predicted_positive_probability üretir.
    Her ticker'ın en son satırı kullanılır.
    """
    model       = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    base_feature_cols = [
        "article_count", "weighted_sentiment", "sentiment_std",
        "mean_confidence", "positive_ratio", "negative_ratio",
        "mean_sentiment_confidence", "sentiment_flow_5d", "confidence_flow_5d",
        "sentiment_flow_20d", "confidence_flow_20d",
        "past_5d_return", "past_20d_return", "past_20d_volatility",
    ]

    # Ticker dummies ekle
    df = dataset.dropna(subset=base_feature_cols + ["news_date_dt", "ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=float)
    df = pd.concat([df, ticker_dummies], axis=1)

    # Training'de olmayan sütunları 0 ile doldur
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    probs = {}
    for ticker in tickers:
        ticker_data = df[df["ticker"] == ticker].copy()
        if ticker_data.empty:
            continue
        latest_row = ticker_data.sort_values("news_date_dt").tail(1)
        X_pred = latest_row[feature_cols].astype(float)
        p = float(model.predict_proba(X_pred)[0, 1])
        probs[ticker] = p

    print(f"Mode B signals: {len(probs)} tickers")
    bullish = sum(1 for p in probs.values() if p >= BULLISH_THRESHOLD)
    bearish = sum(1 for p in probs.values() if p <= BEARISH_THRESHOLD)
    print(f"  Bullish (>={BULLISH_THRESHOLD}): {bullish}, Bearish (<={BEARISH_THRESHOLD}): {bearish}")
    return probs


def _build_constraints_from_probs(probs: Dict[str, float], baseline_weights: Dict[str, float]) -> tuple:
    """newssiz baseline ağırlıklara göre constraint dict ve scipy constraint listesi döndürür."""
    news_constraints = {}
    for ticker, prob in probs.items():
        if ticker not in baseline_weights:
            continue
        base_w = float(baseline_weights[ticker])
        if base_w < 1e-3:
            continue
        if prob >= BULLISH_THRESHOLD:
            news_constraints[ticker] = {
                "type": "bullish",
                "min_weight": min(base_w + DELTA, W_MAX - 1e-4),
            }
        elif prob <= BEARISH_THRESHOLD:
            news_constraints[ticker] = {
                "type": "bearish",
                "max_weight": max(0.0, base_w - DELTA),
            }

    tickers_list = list(baseline_weights.keys())
    ticker_to_idx = {t: i for i, t in enumerate(tickers_list)}
    sci_cons = []
    for ticker, cdict in news_constraints.items():
        if ticker not in ticker_to_idx:
            continue
        idx = ticker_to_idx[ticker]
        if "min_weight" in cdict:
            mw = float(cdict["min_weight"])
            sci_cons.append({"type": "ineq", "fun": lambda w, i=idx, m=mw: w[i] - m})
        if "max_weight" in cdict:
            mw = float(cdict["max_weight"])
            sci_cons.append({"type": "ineq", "fun": lambda w, i=idx, m=mw: m - w[i]})

    return news_constraints, sci_cons


# ─────────────────────────────────────────────────────────────
# Mode A: FinBERT-adjusted mu/cov
# ─────────────────────────────────────────────────────────────

def _build_mode_a_ticker_signals(raw_path: str, as_of_date: str, tickers: List[str]) -> Dict[str, dict]:
    """
    as_of_date öncesindeki en son haber verisini kullanarak
    her ticker için FinBERT sentiment sinyali üretir.
    probabilistic_news_integration.py ile aynı mantık ama
    mevcut datasetten (API çağrısı yapmadan).
    """
    df = pd.read_csv(raw_path)
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt", "ticker", "article_sentiment",
        "article_confidence", "combined_weight",
    ]).copy()

    as_of = pd.to_datetime(as_of_date)
    # Son 7 gün öncesi haberler (as_of'tan 7 gün öncesi ile as_of arası)
    lookback_start = as_of - pd.Timedelta(days=7)
    df = df[(df["news_date_dt"] >= lookback_start) & (df["news_date_dt"] < as_of)].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"].isin(tickers)].copy()

    signals = {}
    for ticker, g in df.groupby("ticker"):
        weights = g["combined_weight"].astype(float).values
        sentiments = g["article_sentiment"].astype(float).values
        confidences = g["article_confidence"].astype(float).values

        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        sentiment_score    = float(np.sum(weights * sentiments))
        confidence_score   = float(np.sum(weights * confidences))
        sentiment_variance = float(np.sum(weights * (sentiments - sentiment_score) ** 2))

        signals[ticker] = {
            "sentiment_score":    max(-1.0, min(1.0, sentiment_score)),
            "confidence_score":   max(0.0,  min(1.0, confidence_score)),
            "sentiment_variance": max(0.0,  sentiment_variance),
        }

    # Haber olmayan tickerlar için nötr sinyal
    for t in tickers:
        if t not in signals:
            signals[t] = {"sentiment_score": 0.0, "confidence_score": 0.0, "sentiment_variance": 0.0}

    print(f"Mode A ticker signals: {len([s for s in signals.values() if s['sentiment_score'] != 0.0])} tickers with non-zero sentiment")
    return signals


def _adjust_mu(mu: pd.Series, signals: Dict[str, dict], alpha: float) -> pd.Series:
    adjusted = mu.copy().astype(float)
    for ticker in adjusted.index:
        sig = signals.get(ticker.upper())
        if sig is None:
            continue
        s = sig["sentiment_score"]
        c = sig["confidence_score"]
        adjusted.loc[ticker] += alpha * s * c
    return adjusted


def _adjust_cov(cov: pd.DataFrame, signals: Dict[str, dict], beta: float) -> pd.DataFrame:
    adjusted = cov.copy().astype(float)
    for ticker in adjusted.index:
        sig = signals.get(ticker.upper())
        if sig is None:
            continue
        c   = sig["confidence_score"]
        var = sig["sentiment_variance"]
        uncertainty = 0.55 * (1.0 - c) + 0.45 * min(1.0, 3.0 * var)
        adjusted.loc[ticker, ticker] *= (1.0 + beta * uncertainty)
    adjusted = 0.5 * (adjusted + adjusted.T)
    return adjusted


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_news_ablation_study():
    print("\n" + "=" * 70)
    print("NEWS ABLATION REALIZED PERFORMANCE STUDY")
    print("Mode A (FinBERT-adjusted) and Mode B (Prediction-constrained)")
    print(f"Test period: {TEST_START} → {TEST_END}")
    print(f"Signal as-of date: {AS_OF_DATE}")
    print("=" * 70)

    # 1) Load mu/cov (full 101-ticker)
    mu, cov = _load_mu_cov()
    tickers = list(mu.index)
    print(f"\nUniverse: {len(tickers)} tickers")

    # 2) Baseline (no news)
    baseline_weights = _optimize(mu, cov, RF, W_MAX, LAMBDA_L2)
    n_active = sum(1 for v in baseline_weights.values() if v > 1e-6)
    print(f"Baseline active assets: {n_active}")

    # 3) Load test-period prices
    print(f"\nLoading prices ({TEST_START} → {TEST_END})...")
    prices = _load_prices(tickers, TEST_START, TEST_END)
    print(f"Price matrix: {prices.shape[0]} trading days × {prices.shape[1]} tickers")

    # 4) Baseline realized metrics
    baseline_m = _realized_metrics(baseline_weights, prices, RF)
    print(f"\nBaseline (no news): Sharpe={baseline_m['sharpe']:.4f}  "
          f"Return={baseline_m['return']*100:.2f}%  "
          f"Vol={baseline_m['vol']*100:.2f}%  "
          f"MaxDD={baseline_m['max_dd']*100:.2f}%")

    rows = [{
        "mode":          "Baseline (no news)",
        "sharpe":        round(baseline_m["sharpe"],           4),
        "return_pct":    round(baseline_m["return"]   * 100,   2),
        "vol_pct":       round(baseline_m["vol"]      * 100,   2),
        "max_dd_pct":    round(baseline_m["max_dd"]   * 100,   2),
        "delta_sharpe":  0.0,
        "delta_return":  0.0,
        "delta_vol":     0.0,
        "delta_max_dd":  0.0,
        "n_constraints": 0,
        "turnover_pct":  0.0,
    }]

    # ── MODE B ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("MODE B: Prediction-Constrained Optimization")

    SIGNALS_CSV = Path("data/news_prediction/latest_news_prediction_signals_as_of_20260114.csv")

    if not SIGNALS_CSV.exists():
        print(f"  [SKIP] Signals file not found: {SIGNALS_CSV}")
    else:
        signals_df = pd.read_csv(SIGNALS_CSV)
        signals_df["ticker"] = signals_df["ticker"].astype(str).str.upper().str.strip()
        probs = dict(zip(signals_df["ticker"], signals_df["predicted_positive_probability"].astype(float)))
        probs = {t: p for t, p in probs.items() if t in tickers}
        print(f"  Loaded {len(probs)} ticker signals from {SIGNALS_CSV.name}")

        news_constraints_b, sci_cons_b = _build_constraints_from_probs(probs, baseline_weights)
        n_bull = sum(1 for c in news_constraints_b.values() if c["type"] == "bullish")
        n_bear = sum(1 for c in news_constraints_b.values() if c["type"] == "bearish")
        print(f"  Active constraints: {n_bull} bullish, {n_bear} bearish")

        constrained_w_b = _optimize(mu, cov, RF, W_MAX, LAMBDA_L2, sci_cons_b)

        turnover_b = sum(abs(constrained_w_b.get(t, 0) - baseline_weights.get(t, 0)) for t in tickers) / 2.0

        m_b = _realized_metrics(constrained_w_b, prices, RF)
        ds  = m_b["sharpe"] - baseline_m["sharpe"]
        dr  = m_b["return"] - baseline_m["return"]
        dv  = m_b["vol"]    - baseline_m["vol"]
        dd  = m_b["max_dd"] - baseline_m["max_dd"]

        print(f"  Realized: Sharpe={m_b['sharpe']:.4f} (Δ{ds:+.4f})  "
              f"Return={m_b['return']*100:.2f}% (Δ{dr*100:+.2f}%)  "
              f"Vol={m_b['vol']*100:.2f}% (Δ{dv*100:+.2f}%)  "
              f"MaxDD={m_b['max_dd']*100:.2f}% (Δ{dd*100:+.2f}%)")

        rows.append({
            "mode":          "Mode B (Prediction-Constrained)",
            "sharpe":        round(m_b["sharpe"],       4),
            "return_pct":    round(m_b["return"] * 100, 2),
            "vol_pct":       round(m_b["vol"]    * 100, 2),
            "max_dd_pct":    round(m_b["max_dd"] * 100, 2),
            "delta_sharpe":  round(ds,             4),
            "delta_return":  round(dr * 100,        2),
            "delta_vol":     round(dv * 100,        2),
            "delta_max_dd":  round(dd * 100,        2),
            "n_constraints": n_bull + n_bear,
            "turnover_pct":  round(turnover_b * 100, 2),
        })

    # ── MODE A ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("MODE A: FinBERT-Adjusted MVO")

    signals_a  = _build_mode_a_ticker_signals(RAW_PATH, AS_OF_DATE, tickers)
    mu_adj     = _adjust_mu(mu,  signals_a, ALPHA)
    cov_adj    = _adjust_cov(cov, signals_a, BETA_COV)

    newsli_w_a = _optimize(mu_adj, cov_adj, RF, W_MAX, LAMBDA_L2)
    turnover_a = sum(abs(newsli_w_a.get(t, 0) - baseline_weights.get(t, 0)) for t in tickers) / 2.0

    # realized metrikler için orijinal fiyat verisi kullan
    # (Mode A mu/cov adjusted ama gerçek fiyat değişmedi)
    m_a = _realized_metrics(newsli_w_a, prices, RF)
    ds  = m_a["sharpe"] - baseline_m["sharpe"]
    dr  = m_a["return"] - baseline_m["return"]
    dv  = m_a["vol"]    - baseline_m["vol"]
    dd  = m_a["max_dd"] - baseline_m["max_dd"]

    tickers_with_signal = sum(1 for s in signals_a.values() if abs(s["sentiment_score"]) > 0.05)
    print(f"  Tickers with non-neutral signal: {tickers_with_signal}")
    print(f"  Turnover vs baseline: {turnover_a*100:.1f}%")
    print(f"  Realized: Sharpe={m_a['sharpe']:.4f} (Δ{ds:+.4f})  "
          f"Return={m_a['return']*100:.2f}% (Δ{dr*100:+.2f}%)  "
          f"Vol={m_a['vol']*100:.2f}% (Δ{dv*100:+.2f}%)  "
          f"MaxDD={m_a['max_dd']*100:.2f}% (Δ{dd*100:+.2f}%)")

    rows.append({
        "mode":          "Mode A (FinBERT-Adjusted)",
        "sharpe":        round(m_a["sharpe"],       4),
        "return_pct":    round(m_a["return"] * 100, 2),
        "vol_pct":       round(m_a["vol"]    * 100, 2),
        "max_dd_pct":    round(m_a["max_dd"] * 100, 2),
        "delta_sharpe":  round(ds,             4),
        "delta_return":  round(dr * 100,        2),
        "delta_vol":     round(dv * 100,        2),
        "delta_max_dd":  round(dd * 100,        2),
        "n_constraints": tickers_with_signal,
        "turnover_pct":  round(turnover_a * 100, 2),
    })

    # ── Summary table ─────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("NEWS ABLATION REALIZED PERFORMANCE TABLE")
    print(f"Test period: {TEST_START} → {TEST_END}  |  Universe: {len(tickers)} tickers")
    print(f"{'='*80}")
    hdr = f"{'Mode':<35} {'Sharpe':>8} {'ΔSharpe':>9} {'Return%':>9} {'ΔRet%':>8} {'Vol%':>7} {'MaxDD%':>8} {'Turnover':>10}"
    print(hdr)
    print("-" * 80)
    for r in rows:
        ds_str = f"{r['delta_sharpe']:+.4f}" if r["delta_sharpe"] != 0.0 else "—"
        dr_str = f"{r['delta_return']:+.2f}%" if r["delta_return"] != 0.0 else "—"
        print(f"{r['mode']:<35} "
              f"{r['sharpe']:>8.4f} "
              f"{ds_str:>9} "
              f"{r['return_pct']:>8.2f}% "
              f"{dr_str:>8} "
              f"{r['vol_pct']:>7.2f}% "
              f"{r['max_dd_pct']:>7.2f}% "
              f"{r['turnover_pct']:>9.1f}%")

    # ── Save ──────────────────────────────────────────────────
    df_out   = pd.DataFrame(rows)
    csv_path = OUT_DIR / "news_ablation_realized_study.csv"
    json_path = OUT_DIR / "news_ablation_realized_study.json"

    df_out.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_period": {"start": TEST_START, "end": TEST_END},
            "signal_as_of_date": AS_OF_DATE,
            "parameters": {
                "rf": RF, "w_max": W_MAX, "lambda_l2": LAMBDA_L2,
                "mode_b": {"bullish_threshold": BULLISH_THRESHOLD,
                           "bearish_threshold": BEARISH_THRESHOLD, "delta": DELTA},
                "mode_a": {"alpha": ALPHA, "beta_cov": BETA_COV},
            },
            "baseline": {k: round(v * (100 if k != "sharpe" else 1), 4)
                         for k, v in baseline_m.items()},
            "results": rows,
        }, f, indent=2)

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")

    return {"baseline": baseline_m, "rows": rows}


if __name__ == "__main__":
    run_news_ablation_study()