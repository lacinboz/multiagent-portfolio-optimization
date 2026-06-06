# component_level_impact_study.py
# ============================================================
# LAYER 1 ONLY — Full 101-ticker universe
# Layer 2 (case study) removed.
#
# Feature sets (consistent with news_model_feature_ablation.py):
#   price_only      — 3 features  — Does price alone suffice?
#   sentiment_only  — 7 features  — Does current-day sentiment alone work?
#   news_only       — 11 features — Full news + flow, no price
#   sentiment_price — 10 features — News + price, no flow (what does flow add?)
#   all_features    — 14 features — Production model
#
# All experiments use:
#   - Same 101-ticker universe (mu/cov files)
#   - Same 70/30 chronological split
#   - Same LR (C=0.3)
#   - Same constraint parameters (bull=0.60, bear=0.40, delta=0.02)
#   - min_baseline_weight=1e-3 filter for near-zero positions
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

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
MU_PATH = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH = Path("data/processed_yahoo/cov_annual.csv")

RF = 0.02
W_MAX = 0.30
LAMBDA_L2 = 1e-3
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA = 0.02

# ============================================================
# FEATURE SET DEFINITIONS
# Consistent with news_model_feature_ablation.py
# Each set answers a distinct question.
# ============================================================

FEATURE_GROUPS = {
    "price_only": {
        "features": [
            "past_5d_return",
            "past_20d_return",
            "past_20d_volatility",
        ],
        "description": (
            "Only price/momentum features — no news, no sentiment. "
            "Pure technical baseline. (3 features)"
        ),
        "question": "Does price momentum alone provide sufficient directional signal?",
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
        "description": (
            "Current-day FinBERT sentiment features only — "
            "no price, no rolling flow. (7 features)"
        ),
        "question": "What is the standalone value of current-day news sentiment?",
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
        "description": (
            "Full news features including rolling flow — no price features. (11 features)"
        ),
        "question": "How much does the full news signal (with trend) contribute without price?",
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
        "description": (
            "Current-day sentiment + price features — no rolling flow. (10 features)"
        ),
        "question": "What does adding price momentum to sentiment contribute? (flow ablated out)",
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
        "description": (
            "Full feature set: sentiment + flow + price. Production model. (14 features)"
        ),
        "question": "Does adding rolling flow to sentiment+price further improve outcomes?",
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
    cov_df = pd.read_csv(COV_PATH, index_col=0)
    if tickers is None:
        common = [t for t in summary.index if t in cov_df.index]
    else:
        common = [t for t in tickers if t in summary.index and t in cov_df.index]
    mu = summary.loc[common, "mu_annual"].astype(float)
    cov = cov_df.loc[common, common].astype(float)
    return mu, cov


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
    sharpe = (r - rf) / v if v > 0 else 0.0
    return {"weights": {t: float(w[t]) for t in tickers},
            "return": r, "vol": v, "sharpe": sharpe,
            "success": bool(res.success)}


# ============================================================
# Dataset builder
# ============================================================

def _build_dataset(raw_path: str) -> pd.DataFrame:
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

        def wmean(col):
            vals = g[col].astype(float)
            return float(np.average(vals, weights=weights)) if w_sum > 0 else float(vals.mean())

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
# Train model + predict
# ============================================================

def _train_and_predict(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    portfolio_tickers: List[str],
    test_size: float = 0.30,
    C: float = 0.3,
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

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=C, random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    probs = {}
    for ticker in portfolio_tickers:
        ticker_data = df[df["ticker"] == ticker].copy()
        if ticker_data.empty:
            continue
        for col in feat_cols:
            if col not in ticker_data.columns:
                ticker_data[col] = 0.0
        latest_row = ticker_data.sort_values("news_date_dt").tail(1)
        X_pred = latest_row[feat_cols].astype(float)
        p = float(model.predict_proba(X_pred)[0, 1])
        probs[ticker] = p

    return probs


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
                "type": "bullish", "prob": prob,
                "baseline_weight": base_w,
                "min_weight": min(base_w + DELTA, W_MAX - 1e-4),
            }
        elif prob <= BEARISH_THRESHOLD:
            constraints[ticker] = {
                "type": "bearish", "prob": prob,
                "baseline_weight": base_w,
                "max_weight": max(0.0, base_w - DELTA),
            }
    return constraints


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
    print("=" * 70)
    print(f"Universe: full 101-ticker NASDAQ universe")
    print(f"rf={rf}, w_max={w_max}, lambda_l2={lambda_l2}")
    print(f"Constraints: bull>={BULLISH_THRESHOLD}, bear<={BEARISH_THRESHOLD}, delta={DELTA}")
    print(f"Model: LogisticRegression (C=0.3), 70/30 chronological split")
    print(f"Filter: constraints applied only to tickers with baseline weight >= 0.001")

    # Print feature group summary
    print(f"\nFeature groups ({len(FEATURE_GROUPS)}):")
    for name, cfg in FEATURE_GROUPS.items():
        print(f"  {name:<20} {len(cfg['features']):>3} features — {cfg['question']}")

    # Load full universe
    mu, cov = _load_mu_cov(tickers=None)
    tickers = list(mu.index)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    print(f"\nLoaded {len(tickers)} tickers")

    # Baseline
    baseline = _optimize_portfolio(mu, cov, rf, w_max, lambda_l2)
    baseline_weights = baseline["weights"]
    print(f"Baseline: return={baseline['return']*100:.2f}% "
          f"vol={baseline['vol']*100:.2f}% sharpe={baseline['sharpe']:.4f}")

    # Dataset
    print("\nBuilding dataset...")
    dataset = _build_dataset(raw_path)
    print(f"Dataset: {len(dataset)} rows, {dataset['ticker'].nunique()} tickers")

    # Run each feature group
    rows = []

    for group_name, cfg in FEATURE_GROUPS.items():
        print(f"\n{'─'*60}")
        print(f"Feature group: {group_name}")
        print(f"  {cfg['description']}")
        print(f"  Question: {cfg['question']}")

        probs = _train_and_predict(
            dataset=dataset,
            feature_cols=cfg["features"],
            portfolio_tickers=tickers,
            test_size=0.30,
            C=0.3,
        )
        if not probs:
            print("  FAILED: could not generate predictions")
            continue

        news_constraints = _probs_to_constraints(
            probs, baseline_weights, min_baseline_weight=1e-3
        )
        n_bull = sum(1 for c in news_constraints.values() if c["type"] == "bullish")
        n_bear = sum(1 for c in news_constraints.values() if c["type"] == "bearish")
        print(f"  Active constraints: {n_bull} bullish, {n_bear} bearish")

        sci_cons = _build_scipy_constraints(news_constraints, ticker_to_idx)
        constrained = _optimize_portfolio(mu, cov, rf, w_max, lambda_l2, sci_cons)

        sd = constrained["sharpe"] - baseline["sharpe"]
        rd = constrained["return"] - baseline["return"]
        vd = constrained["vol"] - baseline["vol"]
        turnover = sum(
            abs(constrained["weights"].get(t, 0) - baseline_weights.get(t, 0))
            for t in tickers
        ) / 2.0

        print(f"  Sharpe Δ={sd:+.4f} | Return Δ={rd*100:+.2f}% | "
              f"Vol Δ={vd*100:+.2f}% | Turnover={turnover*100:.1f}%")

        rows.append({
            "feature_group": group_name,
            "n_features": len(cfg["features"]),
            "description": cfg["description"],
            "question": cfg["question"],
            "n_bullish_constraints": n_bull,
            "n_bearish_constraints": n_bear,
            "sharpe": constrained["sharpe"],
            "return_pct": constrained["return"] * 100,
            "vol_pct": constrained["vol"] * 100,
            "sharpe_delta": sd,
            "return_delta_pct": rd * 100,
            "vol_delta_pct": vd * 100,
            "turnover_pct": turnover * 100,
        })

    # Print results table
    print(f"\n\n{'='*70}")
    print("COMPONENT-LEVEL IMPACT TABLE")
    print(f"Universe: {len(tickers)} tickers | Baseline Sharpe={baseline['sharpe']:.4f}")
    print(f"{'='*70}")
    print(f"{'Feature Group':<20} {'#Feat':>6} {'Sharpe Δ':>10} "
          f"{'Return Δ':>10} {'Vol Δ':>8} {'Turnover':>10} "
          f"{'#Bull':>6} {'#Bear':>6}")
    print("-" * 78)
    print(f"{'Baseline (no news)':<20} {'—':>6} {0.0:>+10.4f} "
          f"{0.0:>+10.2f}% {0.0:>+8.2f}% {0.0:>9.1f}% {'—':>6} {'—':>6}")
    for row in rows:
        print(
            f"{row['feature_group']:<20}"
            f"{row['n_features']:>6}"
            f"{row['sharpe_delta']:>+10.4f}"
            f"{row['return_delta_pct']:>+10.2f}%"
            f"{row['vol_delta_pct']:>+8.2f}%"
            f"{row['turnover_pct']:>9.1f}%"
            f"{row['n_bullish_constraints']:>6}"
            f"{row['n_bearish_constraints']:>6}"
        )

    # Save
    if save_outputs:
        df_out = pd.DataFrame(rows)
        csv_path = OUT_DIR / "component_level_impact.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        json_path = OUT_DIR / "component_level_impact.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "parameters": {
                    "rf": rf, "w_max": w_max, "lambda_l2": lambda_l2,
                    "bullish_threshold": BULLISH_THRESHOLD,
                    "bearish_threshold": BEARISH_THRESHOLD,
                    "delta": DELTA,
                    "model": "LogisticRegression",
                    "C": 0.3,
                    "train_test_split": "70/30 chronological",
                    "universe_size": len(tickers),
                    "min_baseline_weight_filter": 0.001,
                },
                "feature_groups": {
                    k: {
                        "features": v["features"],
                        "n_features": len(v["features"]),
                        "description": v["description"],
                        "question": v["question"],
                    }
                    for k, v in FEATURE_GROUPS.items()
                },
                "baseline": {
                    "return": baseline["return"],
                    "vol": baseline["vol"],
                    "sharpe": baseline["sharpe"],
                },
                "results": rows,
            }, f, indent=2)
        print(f"[Saved] {json_path}")

    return {"baseline": baseline, "rows": rows, "universe_size": len(tickers)}


if __name__ == "__main__":
    run_component_level_impact_study(
        raw_path=RAW_PATH,
        rf=RF,
        w_max=W_MAX,
        save_outputs=True,
    )