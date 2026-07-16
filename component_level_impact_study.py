# component_level_impact_study.py
# ============================================================
# LAYER 1 ONLY — Full 101-ticker universe
#
#  look-ahead bias fix: latest signals capped at 2026-01-14
#  Both expected AND realized metrics reported
#  5-run mean±std (no fixed seeds)
#  Deterministic for LR (std≈0 expected, confirms stability)
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from realized_eval import compute_realized_metrics, load_test_returns

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
MU_PATH  = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH = Path("data/processed_yahoo/cov_annual.csv")

#  look-ahead bias fix
SIGNAL_CUTOFF = pd.Timestamp("2026-01-14")

N_RUNS            = 5
RF                = 0.02
W_MAX             = 0.30
LAMBDA_L2         = 1e-3
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA             = 0.02

# ============================================================
# FEATURE SET DEFINITIONS
# ============================================================

FEATURE_GROUPS = {
    "price_only": {
        "features": [
            "past_5d_return",
            "past_20d_return",
            "past_20d_volatility",
        ],
        "description": "Only price/momentum — no news. (3 features)",
    },
    "sentiment_only": {
        "features": [
            "article_count",
            "weighted_sentiment",
            "sentiment_std",
            "mean_confidence",
            "positive_ratio",
            "negative_ratio",
            "mean_sentiment_confidence",
        ],
        "description": "Current-day FinBERT sentiment — no price, no flow. (7 features)",
    },
    "news_only": {
        "features": [
            "article_count",
            "weighted_sentiment",
            "sentiment_std",
            "mean_confidence",
            "positive_ratio",
            "negative_ratio",
            "mean_sentiment_confidence",
            "sentiment_flow_5d",
            "confidence_flow_5d",
            "sentiment_flow_20d",
            "confidence_flow_20d",
        ],
        "description": "Full news + rolling flow — no price. (11 features)",
    },
    "sentiment_price": {
        "features": [
            "article_count",
            "weighted_sentiment",
            "sentiment_std",
            "mean_confidence",
            "positive_ratio",
            "negative_ratio",
            "mean_sentiment_confidence",
            "past_5d_return",
            "past_20d_return",
            "past_20d_volatility",
        ],
        "description": "Sentiment + price — no rolling flow. (10 features)",
    },
    "all_features": {
        "features": [
            "article_count",
            "weighted_sentiment",
            "sentiment_std",
            "mean_confidence",
            "positive_ratio",
            "negative_ratio",
            "mean_sentiment_confidence",
            "sentiment_flow_5d",
            "confidence_flow_5d",
            "sentiment_flow_20d",
            "confidence_flow_20d",
            "past_5d_return",
            "past_20d_return",
            "past_20d_volatility",
        ],
        "description": "Full: sentiment + flow + price. Production model. (14 features)",
    },
}

# ============================================================
# Portfolio helpers
# ============================================================

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
    return (summary.loc[common, "mu_annual"].astype(float),
            cov_df.loc[common, common].astype(float))


def _optimize_portfolio(mu, cov, rf, w_max, lambda_l2, extra_constraints=None):
    tickers = list(mu.index)
    n = len(tickers)
    effective_w_max = max(w_max, 1.0 / n + 1e-6)
    cov_np = cov.values.copy()
    if np.linalg.eigvalsh(cov_np).min() < 0:
        cov_np = _near_psd(cov_np)
    cov_f = pd.DataFrame(cov_np, index=tickers, columns=tickers)
    bounds = [(0.0, effective_w_max)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        cons += extra_constraints
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov_f.values @ w))
        return -(r - rf) / v if v > 0 else np.inf

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(neg_sharpe, w0, method="trust-constr",
                       bounds=bounds, constraints=cons)
    w = pd.Series(np.clip(res.x, 0, None), index=tickers)
    w = w / w.sum()
    r = float(w.values @ mu.values)
    v = float(np.sqrt(w.values @ cov_f.values @ w.values))
    return {
        "weights": {t: float(w[t]) for t in tickers},
        "return": r, "vol": v,
        "sharpe": (r - rf) / v if v > 0 else 0.0,
    }


# ============================================================
# Dataset builders
# ============================================================

def _build_dataset(raw_path: str) -> pd.DataFrame:
    """Training dataset with tau=0.02 filter."""
    df = pd.read_csv(raw_path)
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt", "ticker", "future_return",
        "article_sentiment", "article_confidence",
        "prob_positive", "prob_negative", "prob_neutral",
        "combined_weight", "past_5d_return",
        "past_20d_return", "past_20d_volatility",
    ]).copy()
    df = df[df["future_return"].abs() >= 0.02].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["is_positive_article"] = (df["prob_positive"] > df["prob_negative"]).astype(int)
    df["is_negative_article"] = (df["prob_negative"] > df["prob_positive"]).astype(int)
    df["sentiment_confidence"] = df["article_sentiment"] * df["article_confidence"]

    grouped_rows = []
    for (ticker, news_date, news_date_dt), g in df.groupby(
        ["ticker", "news_date", "news_date_dt"]
    ):
        weights = g["article_confidence"].astype(float)
        w_sum = float(weights.sum())

        def wmean(col, _g=g, _w=weights, _ws=w_sum):
            vals = _g[col].astype(float)
            return float(np.average(vals, weights=_w)) if _ws > 0 else float(vals.mean())

        grouped_rows.append({
            "ticker": ticker,
            "news_date": news_date,
            "news_date_dt": news_date_dt,
            "article_count": int(len(g)),
            "weighted_sentiment": wmean("article_sentiment"),
            "sentiment_std": float(g["article_sentiment"].std()) if len(g) > 1 else 0.0,
            "mean_confidence": float(g["article_confidence"].mean()),
            "positive_ratio": float(g["is_positive_article"].mean()),
            "negative_ratio": float(g["is_negative_article"].mean()),
            "mean_sentiment_confidence": float(g["sentiment_confidence"].mean()),
            "past_5d_return": float(g["past_5d_return"].mean()),
            "past_20d_return": float(g["past_20d_return"].mean()),
            "past_20d_volatility": float(g["past_20d_volatility"].mean()),
            "future_return": float(g["future_return"].mean()),
        })

    out = pd.DataFrame(grouped_rows)
    out = out.sort_values(["ticker", "news_date_dt"]).reset_index(drop=True)

    flow_parts = []
    for ticker, g in out.groupby("ticker"):
        g = g.sort_values("news_date_dt").copy()
        for w in [5, 20]:
            g[f"sentiment_flow_{w}d"] = (
                g["weighted_sentiment"].shift(1).rolling(w, min_periods=1).mean()
            )
            g[f"confidence_flow_{w}d"] = (
                g["mean_confidence"].shift(1).rolling(w, min_periods=1).mean()
            )
        flow_parts.append(g)

    out = pd.concat(flow_parts, ignore_index=True)
    out = out.dropna().reset_index(drop=True)
    out["target_direction"] = (out["future_return"] > 0).astype(int)
    return out


def _build_latest_dataset(raw_path: str) -> pd.DataFrame:
    """
    Unfiltered dataset for latest signal generation (tau=None).
     look-ahead bias fix: capped at SIGNAL_CUTOFF (2026-01-14).
    """
    df = pd.read_csv(raw_path)
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt", "ticker", "future_return",
        "article_sentiment", "article_confidence",
        "prob_positive", "prob_negative", "prob_neutral",
        "combined_weight", "past_5d_return",
        "past_20d_return", "past_20d_volatility",
    ]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    #  look-ahead bias fix
    df = df[df["news_date_dt"] <= SIGNAL_CUTOFF].copy()

    df["is_positive_article"] = (df["prob_positive"] > df["prob_negative"]).astype(int)
    df["is_negative_article"] = (df["prob_negative"] > df["prob_positive"]).astype(int)
    df["sentiment_confidence"] = df["article_sentiment"] * df["article_confidence"]

    grouped_rows = []
    for (ticker, news_date, news_date_dt), g in df.groupby(
        ["ticker", "news_date", "news_date_dt"]
    ):
        weights = g["article_confidence"].astype(float)
        w_sum = float(weights.sum())

        def wmean(col, _g=g, _w=weights, _ws=w_sum):
            vals = _g[col].astype(float)
            return float(np.average(vals, weights=_w)) if _ws > 0 else float(vals.mean())

        grouped_rows.append({
            "ticker": ticker,
            "news_date": news_date,
            "news_date_dt": news_date_dt,
            "article_count": int(len(g)),
            "weighted_sentiment": wmean("article_sentiment"),
            "sentiment_std": float(g["article_sentiment"].std()) if len(g) > 1 else 0.0,
            "mean_confidence": float(g["article_confidence"].mean()),
            "positive_ratio": float(g["is_positive_article"].mean()),
            "negative_ratio": float(g["is_negative_article"].mean()),
            "mean_sentiment_confidence": float(g["sentiment_confidence"].mean()),
            "past_5d_return": float(g["past_5d_return"].mean()),
            "past_20d_return": float(g["past_20d_return"].mean()),
            "past_20d_volatility": float(g["past_20d_volatility"].mean()),
            "future_return": float(g["future_return"].mean()),
        })

    out = pd.DataFrame(grouped_rows)
    out = out.sort_values(["ticker", "news_date_dt"]).reset_index(drop=True)

    flow_parts = []
    for ticker, g in out.groupby("ticker"):
        g = g.sort_values("news_date_dt").copy()
        for w in [5, 20]:
            g[f"sentiment_flow_{w}d"] = (
                g["weighted_sentiment"].shift(1).rolling(w, min_periods=1).mean()
            )
            g[f"confidence_flow_{w}d"] = (
                g["mean_confidence"].shift(1).rolling(w, min_periods=1).mean()
            )
        flow_parts.append(g)

    out = pd.concat(flow_parts, ignore_index=True)
    out = out.dropna().reset_index(drop=True)
    out["target_direction"] = (out["future_return"] > 0).astype(int)
    return out


# ============================================================
# Train model + predict (single run)
# ============================================================

def _train_and_predict(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    portfolio_tickers: List[str],
    latest_dataset: pd.DataFrame,
    test_size: float = 0.30,
) -> Optional[Dict[str, float]]:
    available = [f for f in feature_cols if f in dataset.columns]
    if not available:
        return None

    df = dataset.dropna(
        subset=available + ["target_direction", "news_date_dt", "ticker"]
    ).copy()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=float)
    df = pd.concat([df, dummies], axis=1)
    feat_cols = available + list(dummies.columns)

    y = df["target_direction"].astype(int)
    if y.nunique() < 2:
        return None

    split_idx = int(len(df) * (1.0 - test_size))
    if split_idx <= 10:
        return None

    X_train = df.iloc[:split_idx][feat_cols].astype(float)
    y_train = y.iloc[:split_idx]

    if y_train.nunique() < 2:
        return None

    # NO random_state — probabilistic
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.3,
        )),
    ])
    model.fit(X_train, y_train)

    # Latest signals from CUTOFF-capped dataset
    latest_df = latest_dataset.dropna(
        subset=available + ["target_direction", "news_date_dt", "ticker"]
    ).copy()
    latest_df = latest_df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    latest_dummies = pd.get_dummies(latest_df["ticker"], prefix="ticker", dtype=float)
    latest_df = pd.concat([latest_df, latest_dummies], axis=1)
    for col in feat_cols:
        if col not in latest_df.columns:
            latest_df[col] = 0.0

    probs = {}
    for ticker in portfolio_tickers:
        sub = latest_df[latest_df["ticker"] == ticker].copy()
        if sub.empty:
            continue
        latest_row = sub.sort_values("news_date_dt").tail(1)
        p = float(model.predict_proba(latest_row[feat_cols].astype(float))[0, 1])
        probs[ticker] = p

    return probs


def _probs_to_constraints(probs, baseline_weights, min_baseline_weight=1e-3):
    constraints = {}
    for ticker, prob in probs.items():
        if ticker not in baseline_weights:
            continue
        base_w = float(baseline_weights[ticker])
        if base_w < min_baseline_weight:
            continue
        if prob >= BULLISH_THRESHOLD:
            constraints[ticker] = {
                "type": "bullish",
                "min_weight": min(base_w + DELTA, W_MAX - 1e-4),
            }
        elif prob <= BEARISH_THRESHOLD:
            constraints[ticker] = {
                "type": "bearish",
                "max_weight": max(0.0, base_w - DELTA),
            }
    return constraints


def _build_scipy_constraints(news_constraints, ticker_to_idx):
    sci = []
    for ticker, cdict in news_constraints.items():
        if ticker not in ticker_to_idx:
            continue
        idx = ticker_to_idx[ticker]
        if "min_weight" in cdict:
            mw = float(cdict["min_weight"])
            sci.append({"type": "ineq", "fun": lambda w, i=idx, m=mw: w[i] - m})
        if "max_weight" in cdict:
            mw = float(cdict["max_weight"])
            sci.append({"type": "ineq", "fun": lambda w, i=idx, m=mw: m - w[i]})
    return sci


# ============================================================
# Aggregate runs → mean ± std
# ============================================================

def _aggregate(group_name: str, run_results: List[Dict]) -> Dict:
    if not run_results:
        return {"feature_group": group_name, "n_runs": 0}

    metrics = [
        "exp_sharpe_delta", "exp_return_delta", "exp_vol_delta",
        "real_sharpe_delta", "real_sharpe", "real_return_pct", "real_vol_pct",
        "turnover", "n_bullish", "n_bearish",
    ]
    out: Dict[str, Any] = {
        "feature_group": group_name,
        "n_runs": len(run_results),
        "n_features": run_results[0].get("n_features"),
    }
    for m in metrics:
        vals = [r[m] for r in run_results if m in r and r[m] is not None]
        if vals:
            out[f"{m}_mean"] = float(np.mean(vals))
            out[f"{m}_std"]  = float(np.std(vals))
    return out


# ============================================================
# Main study
# ============================================================

def run_component_level_impact_study(
    raw_path: str = RAW_PATH,
    rf: float = RF,
    w_max: float = W_MAX,
    lambda_l2: float = LAMBDA_L2,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("COMPONENT-LEVEL PORTFOLIO IMPACT STUDY")
    print(f"N_RUNS={N_RUNS} | No fixed seeds (probabilistic)")
    print(f" Signal cutoff: {SIGNAL_CUTOFF.date()} (no look-ahead bias)")
    print(f" Both expected AND realized metrics reported")
    print("=" * 70)
    print(f"rf={rf}, w_max={w_max}, lambda_l2={lambda_l2}")
    print(f"bull>={BULLISH_THRESHOLD}, bear<={BEARISH_THRESHOLD}, delta={DELTA}")

    # Load mu/cov
    mu, cov = _load_mu_cov(tickers=None)
    tickers = list(mu.index)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    print(f"\nUniverse: {len(tickers)} tickers")

    # Baseline
    baseline = _optimize_portfolio(mu, cov, rf, w_max, lambda_l2)
    baseline_weights = baseline["weights"]

    # Load test returns
    returns_test = load_test_returns()
    baseline_realized = compute_realized_metrics(baseline_weights, returns_test)
    baseline_realized_sharpe = baseline_realized["realized_sharpe"]

    print(f"Baseline expected : Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | Vol={baseline['vol']*100:.2f}%")
    print(f"Baseline realized : Sharpe={baseline_realized_sharpe:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}% | "
          f"Vol={baseline_realized['realized_vol']*100:.2f}%")

    # Build datasets once
    print("\nBuilding datasets...")
    dataset        = _build_dataset(raw_path)
    latest_dataset = _build_latest_dataset(raw_path)
    print(f"Training dataset : {len(dataset)} rows (tau=0.02)")
    print(f"Latest dataset   : {len(latest_dataset)} rows "
          f"(tau=None, capped at {SIGNAL_CUTOFF.date()})")
    print(f"Latest max date  : {latest_dataset['news_date_dt'].max().date()}")

    # Run each feature group
    summary_rows = []
    all_run_rows  = []

    for group_name, cfg in FEATURE_GROUPS.items():
        print(f"\n{'─'*60}")
        print(f"Feature group: {group_name} ({len(cfg['features'])} features)")
        print(f"{'─'*60}")

        run_results = []

        for run_idx in range(N_RUNS):
            print(f"  Run {run_idx+1}/{N_RUNS} ...", end=" ", flush=True)

            probs = _train_and_predict(
                dataset=dataset,
                feature_cols=cfg["features"],
                portfolio_tickers=tickers,
                latest_dataset=latest_dataset,
                test_size=0.30,
            )
            if not probs:
                print("FAILED")
                continue

            news_constraints = _probs_to_constraints(
                probs, baseline_weights, min_baseline_weight=1e-3
            )
            n_bull = sum(1 for c in news_constraints.values() if c["type"] == "bullish")
            n_bear = sum(1 for c in news_constraints.values() if c["type"] == "bearish")

            sci_cons = _build_scipy_constraints(news_constraints, ticker_to_idx)
            constrained = _optimize_portfolio(mu, cov, rf, w_max, lambda_l2, sci_cons)

            exp_sd = constrained["sharpe"] - baseline["sharpe"]
            exp_rd = constrained["return"] - baseline["return"]
            exp_vd = constrained["vol"] - baseline["vol"]
            turnover = sum(
                abs(constrained["weights"].get(t, 0) - baseline_weights.get(t, 0))
                for t in tickers
            ) / 2.0

            # Realized metrics
            real = compute_realized_metrics(constrained["weights"], returns_test)

            run_row = {
                "feature_group": group_name,
                "run": run_idx + 1,
                "n_features": len(cfg["features"]),
                "n_bullish": n_bull,
                "n_bearish": n_bear,
                "exp_sharpe_delta": exp_sd,
                "exp_return_delta": exp_rd,
                "exp_vol_delta": exp_vd,
                "turnover": turnover,
                "real_sharpe": real["realized_sharpe"],
                "real_sharpe_delta": real["realized_sharpe"] - baseline_realized_sharpe,
                "real_return_pct": real["realized_return"] * 100,
                "real_vol_pct": real["realized_vol"] * 100,
            }
            run_results.append(run_row)
            all_run_rows.append(run_row)

            print(f"Exp ΔS={exp_sd:+.4f} | Real ΔS={real['realized_sharpe'] - baseline_realized_sharpe:+.4f} | "
                  f"#C={n_bull+n_bear} | TO={turnover*100:.1f}%")

        summary = _aggregate(group_name, run_results)
        summary_rows.append(summary)

    # ============================================================
    # Print summary table
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"COMPONENT-LEVEL IMPACT — EXPECTED METRICS (mean±std, {N_RUNS} runs)")
    print(f"Expected baseline: Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}%")
    print(f"{'='*80}")

    col = 18
    print(f"\n{'Feature group':<20} {'#Feat':>6} "
          f"{'Exp ΔSharpe':>{col}} {'Turnover':>{col}} {'#Bull':>8} {'#Bear':>8}")
    print("─" * (20 + 6 + col * 2 + 16 + 4))

    print(f"{'Baseline (no news)':<20} {'—':>6} "
          f"{'+0.000±0.000':>{col}} {'0.0%±0.0%':>{col}} {'—':>8} {'—':>8}")

    for s in summary_rows:
        def ms(key, pct=False, digits=3):
            m = s.get(f"{key}_mean")
            d = s.get(f"{key}_std")
            if m is None:
                return "—"
            if pct:
                return f"{m*100:+.1f}%±{d*100:.1f}%"
            return f"{m:+.{digits}f}±{d:.{digits}f}"

        def fmt_int(key):
            m = s.get(f"{key}_mean")
            d = s.get(f"{key}_std")
            if m is None:
                return "—"
            return f"{m:.1f}±{d:.1f}"

        print(
            f"{s['feature_group']:<20} {s.get('n_features', '?'):>6} "
            f"{ms('exp_sharpe_delta'):>{col}} "
            f"{ms('turnover', pct=True):>{col}} "
            f"{fmt_int('n_bullish'):>8} "
            f"{fmt_int('n_bearish'):>8}"
        )

    print(f"\n\n{'='*80}")
    print(f"COMPONENT-LEVEL IMPACT — REALIZED METRICS (mean±std, {N_RUNS} runs)")
    print(f"Realized baseline: Sharpe={baseline_realized_sharpe:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}%")
    print(f"{'='*80}")

    print(f"\n{'Feature group':<20} {'#Feat':>6} "
          f"{'Real ΔSharpe':>{col}} {'Real Return':>{col}} {'Real Vol':>{col}}")
    print("─" * (20 + 6 + col * 3 + 6))

    _br = f"{baseline_realized['realized_return']*100:.2f}%"
    _bv = f"{baseline_realized['realized_vol']*100:.2f}%"
    print(f"{'Baseline (no news)':<20} {'—':>6} "
          f"{'+0.000±0.000':>{col}} "
          f"{_br:>{col}} "
          f"{_bv:>{col}}")

    for s in summary_rows:
        def ms_r(key, pct=False, digits=3):
            m = s.get(f"{key}_mean")
            d = s.get(f"{key}_std")
            if m is None:
                return "—"
            if pct:
                return f"{m:.2f}%±{d:.2f}%"
            return f"{m:+.{digits}f}±{d:.{digits}f}"

        print(
            f"{s['feature_group']:<20} {s.get('n_features', '?'):>6} "
            f"{ms_r('real_sharpe_delta'):>{col}} "
            f"{ms_r('real_return_pct', pct=True):>{col}} "
            f"{ms_r('real_vol_pct', pct=True):>{col}}"
        )

    # ============================================================
    # Save outputs
    # ============================================================
    if save_outputs:
        if all_run_rows:
            pd.DataFrame(all_run_rows).to_csv(
                OUT_DIR / "component_level_all_runs.csv", index=False)
            print(f"\n[Saved] {OUT_DIR}/component_level_all_runs.csv")

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                OUT_DIR / "component_level_impact.csv", index=False)
            print(f"[Saved] {OUT_DIR}/component_level_impact.csv")

    return {"baseline": baseline, "summary_rows": summary_rows}


if __name__ == "__main__":
    run_component_level_impact_study(
        raw_path=RAW_PATH,
        rf=RF,
        w_max=W_MAX,
        save_outputs=True,
    )