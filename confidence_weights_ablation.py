"""
Full Mode A Weight Sensitivity — FIXED-DATA VERSION
============================================================================
Same 4 grids as before (composite confidence, blend, uncertainty, richness)
but now sourced from the SAME frozen news CSV used to produce the thesis's
official Mode A realized result (Delta S = +0.087):

    data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv

with the same as-of date and 7-day lookback window as
news_ablation_realized_study.py. This removes run-to-run variability from
live news fetches and makes the results directly comparable to the
official alpha/beta_cov/half_life sensitivity study (mode_a_ablation.py),
which used a different (live) snapshot.

No FinBERT re-inference needed: prob_positive/negative/neutral are already
stored in the CSV.

Usage:
    python full_weight_sensitivity_fixed_data.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mode_a_ablation import (
    load_base_data, run_base_portfolio, run_maxsharpe_only, portfolio_metrics,
    turnover, RF, W_MAX, LAMBDA,
)
import probabilistic_news_integration as pni

OUT_DIR = Path("data/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed data source (same as news_ablation_realized_study.py) ───────────
RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
AS_OF_DATE = "2026-01-15"
LOOKBACK_DAYS = 7

# ── Fixed alpha/beta/half_life (production defaults, same as thesis) ──────
ALPHA_FIXED = 0.08
BETA_FIXED = 0.35
HALF_LIFE_FIXED = 2.0

# ═════════════════════════════════════════════════════════════════════
# GRID A: composite confidence weights
# ═════════════════════════════════════════════════════════════════════
W_MODEL_GRID = [0.35, 0.40, 0.45, 0.50, 0.55]
W_RICHNESS_GRID = [0.05, 0.10, 0.15, 0.20]

def _build_confidence_weight_grid():
    combos = []
    for w_model in W_MODEL_GRID:
        for w_richness in W_RICHNESS_GRID:
            remaining = 1.0 - w_model - w_richness
            if remaining <= 0:
                continue
            w_source = w_recency = remaining / 2.0
            if w_model >= w_source and w_source >= w_richness:
                combos.append((round(w_model, 3), round(w_source, 3),
                                round(w_recency, 3), round(w_richness, 3)))
    return combos

CONFIDENCE_WEIGHT_GRID = _build_confidence_weight_grid()

# ═════════════════════════════════════════════════════════════════════
# GRID B: article blend weights
# ═════════════════════════════════════════════════════════════════════
BLEND_CONF_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]

# ═════════════════════════════════════════════════════════════════════
# GRID C: uncertainty index weights + variance_scale
# ═════════════════════════════════════════════════════════════════════
W_UNC_CONF_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
VARIANCE_SCALE_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]

# ═════════════════════════════════════════════════════════════════════
# GRID D: content richness constants
# ═════════════════════════════════════════════════════════════════════
RICHNESS_BASE_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
RICHNESS_SCALE_GRID = [300.0, 400.0, 500.0, 600.0, 700.0]


# ═════════════════════════════════════════════════════════════════════
# LOAD FIXED NEWS DATA + PRECOMPUTE (no FinBERT inference needed)
# ═════════════════════════════════════════════════════════════════════
def load_and_precompute_from_fixed_csv(
    tickers: List[str],
    raw_path: str = RAW_PATH,
    as_of_date: str = AS_OF_DATE,
    lookback_days: int = LOOKBACK_DAYS,
) -> List[dict]:
    """
    Reads the frozen article-level CSV, filters to the same 7-day
    look-ahead-bias-free window used by the thesis's official Mode A
    realized result, and extracts the raw per-article quantities needed
    to recompute composite confidence under arbitrary weight schemes:
    model_conf (from prob_positive/negative/neutral), source_conf, text_len,
    sentiment, and raw datetime (for recency).
    """
    df = pd.read_csv(raw_path)
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt", "ticker", "prob_positive", "prob_negative",
        "prob_neutral", "source",
    ]).copy()

    as_of = pd.to_datetime(as_of_date)
    lookback_start = as_of - pd.Timedelta(days=lookback_days)
    df = df[(df["news_date_dt"] >= lookback_start) & (df["news_date_dt"] < as_of)].copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    allowed = {pni._normalize_ticker(t) for t in tickers}
    df = df[df["ticker"].isin(allowed)].copy()

    print(f"[FIXED-DATA] {len(df)} articles in window "
          f"[{lookback_start.date()}, {as_of.date()}) matching {len(allowed)} tickers.")

    if df.empty:
        return []

    base_signals = []
    for _, row in df.iterrows():
        positive = float(row["prob_positive"])
        negative = float(row["prob_negative"])
        neutral = float(row["prob_neutral"])
        sentiment = max(-1.0, min(1.0, positive - negative))
        model_conf = max(positive, negative, neutral)

        headline = str(row.get("headline") or "")
        summary = str(row.get("summary") or "")
        text_len = len(f"{headline} {summary}".strip())

        # datetime column preferred (has time-of-day for recency decay);
        # fall back to news_date if datetime is missing/unparseable
        dt_val = row.get("datetime")
        if pd.isna(dt_val):
            dt_val = row["news_date"]

        base_signals.append({
            "ticker": row["ticker"],
            "datetime": dt_val,
            "sentiment": sentiment,
            "model_conf": model_conf,
            "source_conf": pni._source_credibility(str(row.get("source") or "")),
            "text_len": text_len,
        })

    print(f"[FIXED-DATA] {len(base_signals)} article-level base signals prepared "
          f"(no FinBERT re-inference; sourced from existing prob_positive/negative/neutral).")
    return base_signals


def _richness(text_len: int, base_score: float, length_scale: float) -> float:
    return float(min(1.0, base_score + text_len / length_scale))


# Recency weighting relative to as_of_date (NOT "now") since this is a
# frozen historical snapshot, not a live run.
def _recency_weight_as_of(dt_value, half_life_days: float, as_of: pd.Timestamp) -> float:
    dt = pni._to_datetime(dt_value)
    if dt is None:
        return 0.35
    dt_naive = pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt)
    age_days = max(0.0, (as_of - dt_naive).total_seconds() / 86400.0)
    lam = np.log(2.0) / max(half_life_days, 1e-6)
    return float(np.exp(-lam * age_days))


# ═════════════════════════════════════════════════════════════════════
# FAST AGGREGATION — parametrized over all 4 grids, fixed-data version
# ═════════════════════════════════════════════════════════════════════
def fast_adjust_inputs_full(
    mu, cov, tickers, base_signals, *,
    alpha, beta, half_life, as_of: pd.Timestamp,
    w_model, w_source, w_recency_w, w_richness,
    w_blend_conf, w_blend_recency,
    w_unc_conf, w_unc_var, variance_scale,
    richness_base, richness_scale,
):
    grouped: Dict[str, List[tuple]] = {}

    for a in base_signals:
        recency = _recency_weight_as_of(a["datetime"], half_life, as_of)
        richness = _richness(a["text_len"], richness_base, richness_scale)

        confidence = (
            w_model * a["model_conf"]
            + w_source * a["source_conf"]
            + w_recency_w * recency
            + w_richness * richness
        )
        confidence = max(0.0, min(1.0, confidence))
        combined_weight = max(1e-6, w_blend_conf * confidence + w_blend_recency * recency)

        grouped.setdefault(a["ticker"], []).append((a["sentiment"], confidence, combined_weight))

    ticker_signals: Dict[str, pni.TickerNewsSignal] = {}

    for t in tickers:
        tt = pni._normalize_ticker(t)
        items = grouped.get(tt)

        if not items:
            ticker_signals[t] = pni.TickerNewsSignal(
                ticker=t, sentiment_score=0.0, confidence_score=0.0,
                weighted_article_count=0.0, raw_article_count=0, sentiment_variance=0.0,
            )
            continue

        sentiments = np.array([x[0] for x in items], dtype=float)
        confidences = np.array([x[1] for x in items], dtype=float)
        weights = np.array([x[2] for x in items], dtype=float)

        weighted_count = float(weights.sum())
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights_norm = weights / weights.sum()

        sentiment_score = float(np.sum(weights_norm * sentiments))
        confidence_score = float(np.sum(weights_norm * confidences))
        sentiment_variance = float(np.sum(weights_norm * (sentiments - sentiment_score) ** 2))

        ticker_signals[t] = pni.TickerNewsSignal(
            ticker=t,
            sentiment_score=max(-1.0, min(1.0, sentiment_score)),
            confidence_score=max(0.0, min(1.0, confidence_score)),
            weighted_article_count=weighted_count,
            raw_article_count=len(items),
            sentiment_variance=max(0.0, sentiment_variance),
        )

    adjusted_mu = pni.adjust_expected_returns(mu, ticker_signals, alpha=alpha, power=1.5)

    adjusted_cov = cov.copy().astype(float)
    for ticker in adjusted_cov.index:
        sig = ticker_signals.get(ticker)
        if sig is None:
            continue
        original_var = max(float(adjusted_cov.loc[ticker, ticker]), 1e-10)
        disagreement_term = min(1.0, variance_scale * float(sig.sentiment_variance))
        uncertainty = (
            w_unc_conf * (1.0 - float(sig.confidence_score))
            + w_unc_var * disagreement_term
        )
        multiplier = 1.0 + beta * uncertainty
        adjusted_cov.loc[ticker, ticker] = original_var * multiplier
    adjusted_cov = 0.5 * (adjusted_cov + adjusted_cov.T)

    return adjusted_mu, adjusted_cov, ticker_signals


def evaluate_point(mu, cov, tickers, base_signals, base_port, as_of, **kwargs):
    try:
        mu_adj, cov_adj, ticker_signals = fast_adjust_inputs_full(
            mu, cov, tickers, base_signals,
            alpha=ALPHA_FIXED, beta=BETA_FIXED, half_life=HALF_LIFE_FIXED, as_of=as_of,
            **kwargs,
        )
        news_port = run_maxsharpe_only(mu_adj, cov_adj, rf=RF, w_max=W_MAX, lambda_l2=LAMBDA)

        bm = portfolio_metrics(base_port)
        nm = portfolio_metrics(news_port)
        to = turnover(base_port.get("weights") or {}, news_port.get("weights") or {})

        return {
            **kwargs,
            "sharpe_base": round(bm["sharpe"], 5),
            "sharpe_news": round(nm["sharpe"], 5),
            "delta_sharpe": round(nm["sharpe"] - bm["sharpe"], 5),
            "delta_return": round(nm["return"] - bm["return"], 5),
            "delta_vol": round(nm["vol"] - bm["vol"], 5),
            "turnover": round(to, 5),
            "status": "ok",
        }
    except Exception as exc:
        return {**kwargs, "status": f"error: {exc}"}


PROD_CONF = dict(w_model=0.45, w_source=0.20, w_recency_w=0.20, w_richness=0.15)
PROD_BLEND = dict(w_blend_conf=0.50, w_blend_recency=0.50)
PROD_UNC = dict(w_unc_conf=0.55, w_unc_var=0.45, variance_scale=3.0)
PROD_RICH = dict(richness_base=0.25, richness_scale=500.0)


def run_grid_a(mu, cov, tickers, base_signals, base_port, as_of):
    print(f"\n[GRID A] Composite confidence weights ({len(CONFIDENCE_WEIGHT_GRID)} combos) …")
    rows = []
    for i, (w_model, w_source, w_recency_w, w_richness) in enumerate(CONFIDENCE_WEIGHT_GRID, 1):
        row = evaluate_point(
            mu, cov, tickers, base_signals, base_port, as_of,
            w_model=w_model, w_source=w_source, w_recency_w=w_recency_w, w_richness=w_richness,
            **PROD_BLEND, **PROD_UNC, **PROD_RICH,
        )
        rows.append(row)
        print(f"  [{i:2d}/{len(CONFIDENCE_WEIGHT_GRID)}] w_model={w_model:.2f} "
              f"w_src={w_source:.2f} w_rich={w_richness:.2f}  "
              f"ΔSharpe={row.get('delta_sharpe','?')}  status={row.get('status')}")
    return pd.DataFrame(rows)


def run_grid_b(mu, cov, tickers, base_signals, base_port, as_of):
    print(f"\n[GRID B] Blend weights ({len(BLEND_CONF_GRID)} combos) …")
    rows = []
    for i, w_blend_conf in enumerate(BLEND_CONF_GRID, 1):
        w_blend_recency = round(1.0 - w_blend_conf, 3)
        row = evaluate_point(
            mu, cov, tickers, base_signals, base_port, as_of,
            **PROD_CONF, w_blend_conf=w_blend_conf, w_blend_recency=w_blend_recency,
            **PROD_UNC, **PROD_RICH,
        )
        rows.append(row)
        print(f"  [{i}/{len(BLEND_CONF_GRID)}] w_blend_conf={w_blend_conf:.2f}  "
              f"ΔSharpe={row.get('delta_sharpe','?')}  status={row.get('status')}")
    return pd.DataFrame(rows)


def run_grid_c(mu, cov, tickers, base_signals, base_port, as_of):
    combos = [(c, round(1.0 - c, 3)) for c in W_UNC_CONF_GRID]
    total = len(combos) * len(VARIANCE_SCALE_GRID)
    print(f"\n[GRID C] Uncertainty index weights x variance_scale ({total} combos) …")
    rows = []
    i = 0
    for w_unc_conf, w_unc_var in combos:
        for vscale in VARIANCE_SCALE_GRID:
            i += 1
            row = evaluate_point(
                mu, cov, tickers, base_signals, base_port, as_of,
                **PROD_CONF, **PROD_BLEND,
                w_unc_conf=w_unc_conf, w_unc_var=w_unc_var, variance_scale=vscale,
                **PROD_RICH,
            )
            rows.append(row)
            print(f"  [{i:2d}/{total}] w_unc_conf={w_unc_conf:.2f} vscale={vscale:.1f}  "
                  f"ΔSharpe={row.get('delta_sharpe','?')}  status={row.get('status')}")
    return pd.DataFrame(rows)


def run_grid_d(mu, cov, tickers, base_signals, base_port, as_of):
    total = len(RICHNESS_BASE_GRID) * len(RICHNESS_SCALE_GRID)
    print(f"\n[GRID D] Content richness constants ({total} combos) …")
    rows = []
    i = 0
    for base_score in RICHNESS_BASE_GRID:
        for length_scale in RICHNESS_SCALE_GRID:
            i += 1
            row = evaluate_point(
                mu, cov, tickers, base_signals, base_port, as_of,
                **PROD_CONF, **PROD_BLEND, **PROD_UNC,
                richness_base=base_score, richness_scale=length_scale,
            )
            rows.append(row)
            print(f"  [{i:2d}/{total}] base={base_score:.2f} scale={length_scale:.0f}  "
                  f"ΔSharpe={row.get('delta_sharpe','?')}  status={row.get('status')}")
    return pd.DataFrame(rows)


def summarize_grid(df, name, prod_filter, prod_label):
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print(f"\n[{name}] All evaluations failed.")
        return

    print(f"\n{'='*70}\nSUMMARY — {name}\n{'='*70}")
    print(f"ΔSharpe range: [{ok['delta_sharpe'].min():+.4f}, {ok['delta_sharpe'].max():+.4f}]")
    print(f"ΔSharpe std:   {ok['delta_sharpe'].std():.4f}")

    prod = ok
    for k, v in prod_filter.items():
        prod = prod[np.isclose(prod[k], v)]
    if not prod.empty:
        d = prod.iloc[0]
        print(f"Production ({prod_label}): ΔSharpe={d['delta_sharpe']:+.4f}  "
              f"Turnover={d['turnover']*100:.2f}%")
    else:
        print(f"[NOTE] Production combo not found in grid for {name}.")

    best = ok.loc[ok["delta_sharpe"].idxmax()]
    print(f"Best in grid: ΔSharpe={best['delta_sharpe']:+.4f}")


def run_all():
    print("=" * 70)
    print("FULL MODE A WEIGHT SENSITIVITY — FIXED-DATA VERSION")
    print(f"Data source: {RAW_PATH}")
    print(f"As-of date: {AS_OF_DATE}  |  Lookback: {LOOKBACK_DAYS} days")
    print(f"  Grid A (confidence weights): {len(CONFIDENCE_WEIGHT_GRID)} combos")
    print(f"  Grid B (blend weights):      {len(BLEND_CONF_GRID)} combos")
    print(f"  Grid C (uncertainty weights): {len(W_UNC_CONF_GRID) * len(VARIANCE_SCALE_GRID)} combos")
    print(f"  Grid D (richness constants): {len(RICHNESS_BASE_GRID) * len(RICHNESS_SCALE_GRID)} combos")
    print(f"  alpha={ALPHA_FIXED}  beta={BETA_FIXED}  half_life={HALF_LIFE_FIXED} (fixed throughout)")
    print("=" * 70)

    print("\n[1/4] Loading base data …")
    mu, cov, tickers = load_base_data()

    print("[2/4] Loading news from FIXED CSV (no live API call) …")
    as_of = pd.to_datetime(AS_OF_DATE)
    base_signals = load_and_precompute_from_fixed_csv(tickers)

    print("[3/4] Running base portfolio …")
    base_port = run_base_portfolio(mu, cov)
    bm = portfolio_metrics(base_port)
    print(f"      Base Sharpe={bm['sharpe']:.4f}")

    print("[4/4] Running all 4 grids …")
    df_a = run_grid_a(mu, cov, tickers, base_signals, base_port, as_of)
    df_b = run_grid_b(mu, cov, tickers, base_signals, base_port, as_of)
    df_c = run_grid_c(mu, cov, tickers, base_signals, base_port, as_of)
    df_d = run_grid_d(mu, cov, tickers, base_signals, base_port, as_of)

    summarize_grid(df_a, "Grid A: Composite Confidence Weights",
                    {"w_model": 0.45, "w_richness": 0.15}, "0.45/0.20/0.20/0.15")
    summarize_grid(df_b, "Grid B: Blend Weights",
                    {"w_blend_conf": 0.50}, "0.50/0.50")
    summarize_grid(df_c, "Grid C: Uncertainty Index Weights",
                    {"w_unc_conf": 0.55, "variance_scale": 3.0}, "0.55/0.45, scale=3")
    summarize_grid(df_d, "Grid D: Content Richness Constants",
                    {"richness_base": 0.25, "richness_scale": 500.0}, "base=0.25, scale=500")

    all_ok = pd.concat([
        df_a[df_a["status"] == "ok"].assign(grid="A_confidence"),
        df_b[df_b["status"] == "ok"].assign(grid="B_blend"),
        df_c[df_c["status"] == "ok"].assign(grid="C_uncertainty"),
        df_d[df_d["status"] == "ok"].assign(grid="D_richness"),
    ], ignore_index=True)

    print(f"\n{'='*70}\nCROSS-GRID ΔSharpe SPREAD (for comparison with alpha/beta_cov)\n{'='*70}")
    for g, sub in all_ok.groupby("grid"):
        print(f"  {g:16s}: range=[{sub['delta_sharpe'].min():+.4f}, "
              f"{sub['delta_sharpe'].max():+.4f}]  std={sub['delta_sharpe'].std():.4f}")

    return df_a, df_b, df_c, df_d


def save_all(df_a, df_b, df_c, df_d):
    names = ["gridA_confidence", "gridB_blend", "gridC_uncertainty", "gridD_richness"]
    for df, name in zip([df_a, df_b, df_c, df_d], names):
        p = OUT_DIR / f"full_weight_sensitivity_fixed_{name}.csv"
        df.to_csv(p, index=False)
        print(f"[SAVED] {p}")


if __name__ == "__main__":
    df_a, df_b, df_c, df_d = run_all()
    save_all(df_a, df_b, df_c, df_d)
    print("\n[DONE] Full weight sensitivity (fixed data) complete.")