# news_model_feature_ablation.py
# ============================================================
# Feature set ablation — two-table design:
#
# TABLE 1 — Classification comparison: LR vs RF
#   All 5 feature sets, both models
#   Metrics: ROC-AUC, Balanced Accuracy, F1, ΔBal
#   5-run mean±std (no fixed seeds)
#
# TABLE 2 — Portfolio impact: LR only (production model)
#   All 5 feature sets
#   Metrics: Expected ΔSharpe, Realized ΔSharpe, Turnover, #Bull, #Bear
#   5-run mean±std (no fixed seeds)
#
# ✅ look-ahead bias fix: latest_dataset capped at 2026-01-14
# ✅ Both expected AND realized metrics in Table 2
# ✅ No fixed seeds (probabilistic)
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from realized_eval import compute_realized_metrics, load_test_returns

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
MU_PATH  = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH = Path("data/processed_yahoo/cov_annual.csv")

# ✅ look-ahead bias fix
SIGNAL_CUTOFF = pd.Timestamp("2026-01-14")

N_RUNS            = 5
RF_RATE           = 0.02
W_MAX             = 0.30
LAMBDA_L2         = 1e-3
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA             = 0.02
MIN_ABS_RETURN    = 0.02
PRODUCTION_FEATURE_SET = "all_features"
USE_TICKER_FEATURES    = True

# ============================================================
# FEATURE SET DEFINITIONS
# ============================================================

FEATURE_SETS = {
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


def _load_mu_cov():
    summary = pd.read_csv(MU_PATH, index_col=0)
    cov_df  = pd.read_csv(COV_PATH, index_col=0)
    common  = [t for t in summary.index if t in cov_df.index]
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

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
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

def _build_dataset(raw_path: str, min_abs_return: Optional[float] = MIN_ABS_RETURN) -> pd.DataFrame:
    """Training dataset with optional tau filter."""
    df = pd.read_csv(raw_path)
    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt", "ticker", "future_return",
        "article_sentiment", "article_confidence",
        "prob_positive", "prob_negative", "prob_neutral",
        "combined_weight", "past_5d_return",
        "past_20d_return", "past_20d_volatility",
    ]).copy()
    if min_abs_return is not None:
        df = df[df["future_return"].abs() >= float(min_abs_return)].copy()
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
    ✅ look-ahead bias fix: capped at SIGNAL_CUTOFF (2026-01-14).
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

    # ✅ look-ahead bias fix
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
# Train model + get predictions (single run)
# ============================================================

def _train_and_get_probs(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    portfolio_tickers: List[str],
    test_size: float = 0.30,
    latest_dataset: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    available = [f for f in feature_cols if f in dataset.columns]
    if not available:
        return None

    df = dataset.dropna(
        subset=available + ["target_direction", "news_date_dt", "ticker"]
    ).copy()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    feat_cols = available.copy()
    if USE_TICKER_FEATURES:
        dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        feat_cols += list(dummies.columns)

    y = df["target_direction"].astype(int)
    if y.nunique() < 2:
        return None

    split_idx = int(len(df) * (1.0 - test_size))
    if split_idx <= 10:
        return None

    X_train = df.iloc[:split_idx][feat_cols].astype(float)
    y_train = y.iloc[:split_idx]
    X_test  = df.iloc[split_idx:][feat_cols].astype(float)
    y_test  = y.iloc[split_idx:]

    if y_train.nunique() < 2:
        return None

    model = model_factory()
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, proba_test)) if y_test.nunique() == 2 else None
    bal_acc = float(balanced_accuracy_score(y_test, pred_test))
    f1      = float(f1_score(y_test, pred_test, zero_division=0))

    majority_class    = int(y_train.mode().iloc[0])
    baseline_pred     = np.full(len(y_test), majority_class, dtype=int)
    baseline_bal_acc  = float(balanced_accuracy_score(y_test, baseline_pred))
    bal_acc_delta     = bal_acc - baseline_bal_acc

    # Latest signals from UNFILTERED + CUTOFF dataset
    latest_source = latest_dataset if latest_dataset is not None else df
    latest_df = latest_source.dropna(
        subset=available + ["target_direction", "news_date_dt", "ticker"]
    ).copy()
    latest_df = latest_df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    if USE_TICKER_FEATURES:
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

    return {
        "roc_auc": roc_auc,
        "bal_acc": bal_acc,
        "f1": f1,
        "bal_acc_delta": bal_acc_delta,
        "baseline_bal_acc": baseline_bal_acc,
        "n_features": len(available),
        "probs": probs,
    }


def _probs_to_portfolio_impact(
    probs: Dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    baseline_weights: Dict[str, float],
    baseline_sharpe: float,
    baseline_return: float,
    baseline_vol: float,
    returns_test: pd.DataFrame,
    baseline_realized_sharpe: float,
) -> Dict[str, Any]:
    tickers = list(mu.index)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

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

    n_bull = sum(1 for c in news_constraints.values() if c["type"] == "bullish")
    n_bear = sum(1 for c in news_constraints.values() if c["type"] == "bearish")

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

    constrained = _optimize_portfolio(mu, cov, RF_RATE, W_MAX, LAMBDA_L2, sci_cons)

    turnover = sum(
        abs(constrained["weights"].get(t, 0) - baseline_weights.get(t, 0))
        for t in tickers
    ) / 2.0

    # Realized metrics
    real = compute_realized_metrics(constrained["weights"], returns_test)

    return {
        # expected
        "exp_sharpe_delta": constrained["sharpe"] - baseline_sharpe,
        "exp_return_delta": constrained["return"] - baseline_return,
        "exp_vol_delta":    constrained["vol"] - baseline_vol,
        # realized
        "real_sharpe":       real["realized_sharpe"],
        "real_sharpe_delta": real["realized_sharpe"] - baseline_realized_sharpe,
        "real_return_pct":   real["realized_return"] * 100,
        "real_vol_pct":      real["realized_vol"] * 100,
        # common
        "turnover":    turnover,
        "n_bullish":   n_bull,
        "n_bearish":   n_bear,
        "weights":     constrained["weights"],
    }


# ============================================================
# Aggregate runs → mean ± std
# ============================================================

def _aggregate(model_name: str, feat_name: str, run_results: List[Dict]) -> Dict:
    if not run_results:
        return {"model": model_name, "feature_set": feat_name, "n_runs": 0}

    metrics = [
        "roc_auc", "bal_acc", "f1", "bal_acc_delta",
        "exp_sharpe_delta", "exp_return_delta", "exp_vol_delta",
        "real_sharpe_delta", "real_sharpe", "real_return_pct", "real_vol_pct",
        "turnover", "n_bullish", "n_bearish",
    ]
    out: Dict[str, Any] = {
        "model": model_name,
        "feature_set": feat_name,
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

def run_feature_ablation_study(
    raw_path: str = RAW_PATH,
    test_size: float = 0.30,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("NEWS MODEL FEATURE ABLATION STUDY")
    print(f"N_RUNS={N_RUNS} | No fixed seeds (probabilistic)")
    print(f"✅ Signal cutoff: {SIGNAL_CUTOFF.date()} (no look-ahead bias)")
    print(f"✅ Both expected AND realized metrics in Table 2")
    print("=" * 70)
    print(f"Fixed: 101 tickers | 70/30 chronological split")
    print(f"       bull>={BULLISH_THRESHOLD}, bear<={BEARISH_THRESHOLD}, "
          f"delta={DELTA}, w_max={W_MAX}")

    # Load portfolio data
    mu, cov = _load_mu_cov()
    tickers = list(mu.index)
    baseline = _optimize_portfolio(mu, cov, RF_RATE, W_MAX, LAMBDA_L2)
    baseline_weights = baseline["weights"]

    # Load test returns
    returns_test = load_test_returns()
    baseline_realized = compute_realized_metrics(baseline_weights, returns_test)
    baseline_realized_sharpe = baseline_realized["realized_sharpe"]

    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Baseline expected : Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | Vol={baseline['vol']*100:.2f}%")
    print(f"Baseline realized : Sharpe={baseline_realized_sharpe:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}% | "
          f"Vol={baseline_realized['realized_vol']*100:.2f}%")

    # Build datasets once (deterministic)
    print("\nBuilding datasets...")
    dataset        = _build_dataset(raw_path, min_abs_return=MIN_ABS_RETURN)
    latest_dataset = _build_latest_dataset(raw_path)

    print(f"Training dataset : {len(dataset)} rows (tau=0.02)")
    print(f"Latest dataset   : {len(latest_dataset)} rows "
          f"(tau=None, capped at {SIGNAL_CUTOFF.date()})")
    print(f"Latest max date  : {latest_dataset['news_date_dt'].max().date()}")

    # Model factories — NO random_state
    lr_factory = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
    ])
    rf_factory = lambda: RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1,
    )

    # ============================================================
    # Run 5 times per (model, feature_set)
    # ============================================================
    table1_summaries = []  # LR vs RF classification
    table2_summaries = []  # LR portfolio impact

    feat_names = list(FEATURE_SETS.keys())

    for feat_name in feat_names:
        feat_cols = FEATURE_SETS[feat_name]["features"]
        n_feat = len(feat_cols)

        print(f"\n{'─'*60}")
        print(f"Feature set: {feat_name} ({n_feat} features)")
        print(f"{'─'*60}")

        # ── TABLE 1: LR vs RF classification ──
        for model_name, factory in [("LR (C=0.3)", lr_factory), ("RF (d=6)", rf_factory)]:
            run_results_cls = []
            for run_idx in range(N_RUNS):
                res = _train_and_get_probs(
                    dataset=dataset,
                    feature_cols=feat_cols,
                    model_factory=factory,
                    portfolio_tickers=tickers,
                    test_size=test_size,
                    latest_dataset=latest_dataset,
                )
                if res is None:
                    continue
                run_results_cls.append({
                    "roc_auc": res["roc_auc"],
                    "bal_acc": res["bal_acc"],
                    "f1": res["f1"],
                    "bal_acc_delta": res["bal_acc_delta"],
                    "n_features": res["n_features"],
                })

            summary_cls = _aggregate(model_name, feat_name, run_results_cls)
            table1_summaries.append(summary_cls)
            print(f"  {model_name}: ROC-AUC={summary_cls.get('roc_auc_mean', 0):.4f}"
                  f"±{summary_cls.get('roc_auc_std', 0):.4f} | "
                  f"Bal={summary_cls.get('bal_acc_mean', 0):.4f}"
                  f"±{summary_cls.get('bal_acc_std', 0):.4f}")

        # ── TABLE 2: LR portfolio impact ──
        print(f"  LR portfolio impact:")
        run_results_port = []
        for run_idx in range(N_RUNS):
            res = _train_and_get_probs(
                dataset=dataset,
                feature_cols=feat_cols,
                model_factory=lr_factory,
                portfolio_tickers=tickers,
                test_size=test_size,
                latest_dataset=latest_dataset,
            )
            if res is None:
                continue

            impact = _probs_to_portfolio_impact(
                probs=res["probs"],
                mu=mu, cov=cov,
                baseline_weights=baseline_weights,
                baseline_sharpe=baseline["sharpe"],
                baseline_return=baseline["return"],
                baseline_vol=baseline["vol"],
                returns_test=returns_test,
                baseline_realized_sharpe=baseline_realized_sharpe,
            )
            run_results_port.append({
                "roc_auc": res["roc_auc"],
                "bal_acc": res["bal_acc"],
                "f1": res["f1"],
                "bal_acc_delta": res["bal_acc_delta"],
                "n_features": res["n_features"],
                **{k: v for k, v in impact.items() if k != "weights"},
            })

        summary_port = _aggregate("LR (C=0.3)", feat_name, run_results_port)
        table2_summaries.append(summary_port)
        print(f"    Exp ΔS={summary_port.get('exp_sharpe_delta_mean', 0):+.4f}"
              f"±{summary_port.get('exp_sharpe_delta_std', 0):.4f} | "
              f"Real ΔS={summary_port.get('real_sharpe_delta_mean', 0):+.4f}"
              f"±{summary_port.get('real_sharpe_delta_std', 0):.4f} | "
              f"TO={summary_port.get('turnover_mean', 0)*100:.1f}%"
              f"±{summary_port.get('turnover_std', 0)*100:.1f}%")

    # ============================================================
    # Print TABLE 1
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"TABLE 1 — LR vs RF: Classification Metrics (mean±std, {N_RUNS} runs)")
    print(f"{'='*80}")

    col = 16
    print(f"\n{'Feature set':<20} {'Model':<14} {'#Feat':>6} "
          f"{'ROC-AUC':>{col}} {'Bal.Acc':>{col}} {'F1':>{col}} {'ΔBal':>{col}}")
    print("─" * (20 + 14 + 6 + col * 4 + 8))

    for s in table1_summaries:
        def ms(key, digits=4):
            m = s.get(f"{key}_mean")
            d = s.get(f"{key}_std")
            if m is None:
                return "—"
            return f"{m:.{digits}f}±{d:.{digits}f}"

        print(
            f"{s['feature_set']:<20} {s['model']:<14} {s.get('n_features', '?'):>6} "
            f"{ms('roc_auc'):>{col}} {ms('bal_acc'):>{col}} "
            f"{ms('f1'):>{col}} {ms('bal_acc_delta'):>{col}}"
        )

    # ============================================================
    # Print TABLE 2
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"TABLE 2 — LR Portfolio Impact (mean±std, {N_RUNS} runs)")
    print(f"Expected baseline: Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}%")
    print(f"Realized baseline: Sharpe={baseline_realized_sharpe:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}%")
    print(f"{'='*80}")

    col2 = 18
    print(f"\n{'Feature set':<20} {'#Feat':>6} "
          f"{'Exp ΔSharpe':>{col2}} {'Real ΔSharpe':>{col2}} "
          f"{'Turnover':>{col2}} {'#Bull':>8} {'#Bear':>8}")
    print("─" * (20 + 6 + col2 * 3 + 16 + 6))

    # Baseline row
    print(f"{'Baseline (no news)':<20} {'—':>6} "
          f"{'+0.000±0.000':>{col2}} {'+0.000±0.000':>{col2}} "
          f"{'0.0%±0.0%':>{col2}} {'—':>8} {'—':>8}")

    for s in table2_summaries:
        def ms2(key, pct=False, digits=3):
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
            f"{s['feature_set']:<20} {s.get('n_features', '?'):>6} "
            f"{ms2('exp_sharpe_delta'):>{col2}} "
            f"{ms2('real_sharpe_delta'):>{col2}} "
            f"{ms2('turnover', pct=True):>{col2}} "
            f"{fmt_int('n_bullish'):>8} "
            f"{fmt_int('n_bearish'):>8}"
        )

    # ============================================================
    # Save outputs
    # ============================================================
    if save_outputs:
        pd.DataFrame(table1_summaries).to_csv(
            OUT_DIR / "feature_ablation_table1_classification.csv", index=False)
        pd.DataFrame(table2_summaries).to_csv(
            OUT_DIR / "feature_ablation_table2_portfolio.csv", index=False)
        print(f"\n[Saved] {OUT_DIR}/feature_ablation_table1_classification.csv")
        print(f"[Saved] {OUT_DIR}/feature_ablation_table2_portfolio.csv")

    return {"table1": table1_summaries, "table2": table2_summaries}


if __name__ == "__main__":
    run_feature_ablation_study(
        raw_path=RAW_PATH,
        test_size=0.30,
        save_outputs=True,
    )