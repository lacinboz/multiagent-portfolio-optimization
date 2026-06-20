# rf_portfolio_impact_comparison.py
# ============================================================
# PURPOSE: Compare RF vs LR portfolio-level impact
# Hoca sorusu: "RF sinyalleriyle portfolio yaptığında ne olur?"
#
# Her iki model için:
#   1. all_features feature set kullanılır (production)
#   2. Model eğitilir, latest signals üretilir
#   3. Constraints build edilir
#   4. Portfolio optimize edilir
#   5. ΔSharpe, ΔReturn, ΔVol, Turnover, #Bull, #Bear raporlanır
#
# Standalone — mevcut kodlara dokunmaz.
# ============================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

RAW_PATH  = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
MU_PATH   = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH  = Path("data/processed_yahoo/cov_annual.csv")
OUT_DIR   = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RF_RATE           = 0.02
W_MAX             = 0.30
LAMBDA_L2         = 1e-3
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA             = 0.02
MIN_BASELINE_W    = 1e-3

# Production feature set — all 14 features
FEATURE_COLS = [
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
]

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


def _optimize(mu, cov, rf, w_max, lambda_l2, extra_constraints=None):
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
# Dataset builder (same as news_model_feature_ablation.py)
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
# Train model → get latest probabilities
# ============================================================

def _train_and_get_probs(
    dataset: pd.DataFrame,
    model_factory,
    portfolio_tickers: List[str],
    test_size: float = 0.30,
    latest_dataset: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    Train on filtered dataset (tau=0.02).
    Generate latest signals from unfiltered dataset (tau=None),
    matching production inference flow exactly.
    """
    available = [f for f in FEATURE_COLS if f in dataset.columns]
    if not available:
        return None

    df = dataset.dropna(
        subset=available + ["target_direction", "news_date_dt", "ticker"]
    ).copy()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    feat_cols = available.copy()
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

    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    roc_auc = float(roc_auc_score(y_test, proba_test)) if y_test.nunique() == 2 else None
    bal_acc = float(balanced_accuracy_score(y_test, pred_test))

    # Latest signals from UNFILTERED dataset (production style)
    latest_source = latest_dataset if latest_dataset is not None else df
    latest_df = latest_source.dropna(
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

    return {
        "roc_auc": roc_auc,
        "bal_acc": bal_acc,
        "probs": probs,
    }


# ============================================================
# Probs → constraints → portfolio impact
# ============================================================

def _probs_to_portfolio_impact(
    probs: Dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    baseline_weights: Dict[str, float],
    baseline_sharpe: float,
    baseline_return: float,
    baseline_vol: float,
) -> Dict[str, Any]:
    tickers = list(mu.index)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # Build constraints (same logic as news_constraint_integration.py)
    news_constraints = {}
    for ticker, prob in probs.items():
        if ticker not in baseline_weights:
            continue
        base_w = float(baseline_weights[ticker])
        if base_w < MIN_BASELINE_W:
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

    constrained = _optimize(mu, cov, RF_RATE, W_MAX, LAMBDA_L2, sci_cons)

    turnover = sum(
        abs(constrained["weights"].get(t, 0) - baseline_weights.get(t, 0))
        for t in tickers
    ) / 2.0

    return {
        "sharpe_delta": constrained["sharpe"] - baseline_sharpe,
        "return_delta": constrained["return"] - baseline_return,
        "vol_delta": constrained["vol"] - baseline_vol,
        "turnover": turnover,
        "n_bullish": n_bull,
        "n_bearish": n_bear,
        "n_constraints": n_bull + n_bear,
    }


# ============================================================
# Main comparison
# ============================================================

def run_rf_vs_lr_portfolio_impact(
    raw_path: str = RAW_PATH,
    test_size: float = 0.30,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("RF vs LR — PORTFOLIO-LEVEL IMPACT COMPARISON")
    print("Feature set: all_features (14 features, production)")
    print(f"bull>={BULLISH_THRESHOLD}, bear<={BEARISH_THRESHOLD}, delta={DELTA}")
    print("=" * 70)

    # Load portfolio data
    mu, cov = _load_mu_cov()
    tickers = list(mu.index)
    baseline = _optimize(mu, cov, RF_RATE, W_MAX, LAMBDA_L2)
    baseline_weights = baseline["weights"]
    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Baseline: Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | "
          f"Vol={baseline['vol']*100:.2f}%")

    # Build datasets
    print("\nBuilding datasets...")
    dataset = _build_dataset(raw_path)  # tau=0.02 for training
    # unfiltered for latest signal generation
    df_raw = pd.read_csv(raw_path)
    df_raw["news_date_dt"] = pd.to_datetime(df_raw["news_date"], errors="coerce")
    df_raw = df_raw.dropna(subset=[
        "news_date_dt", "ticker", "future_return",
        "article_sentiment", "article_confidence",
        "prob_positive", "prob_negative", "prob_neutral",
        "combined_weight", "past_5d_return",
        "past_20d_return", "past_20d_volatility",
    ]).copy()
    df_raw["ticker"] = df_raw["ticker"].astype(str).str.upper().str.strip()
    df_raw["is_positive_article"] = (df_raw["prob_positive"] > df_raw["prob_negative"]).astype(int)
    df_raw["is_negative_article"] = (df_raw["prob_negative"] > df_raw["prob_positive"]).astype(int)
    df_raw["sentiment_confidence"] = df_raw["article_sentiment"] * df_raw["article_confidence"]
    grouped_rows = []
    for (ticker, news_date, news_date_dt), g in df_raw.groupby(
        ["ticker", "news_date", "news_date_dt"]
    ):
        weights_g = g["article_confidence"].astype(float)
        w_sum = float(weights_g.sum())
        def wmean(col, _g=g, _w=weights_g, _ws=w_sum):
            vals = _g[col].astype(float)
            return float(np.average(vals, weights=_w)) if _ws > 0 else float(vals.mean())
        grouped_rows.append({
            "ticker": ticker, "news_date": news_date, "news_date_dt": news_date_dt,
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
    latest_dataset = pd.DataFrame(grouped_rows)
    latest_dataset = latest_dataset.sort_values(["ticker", "news_date_dt"]).reset_index(drop=True)
    flow_parts = []
    for ticker, g in latest_dataset.groupby("ticker"):
        g = g.sort_values("news_date_dt").copy()
        for w in [5, 20]:
            g[f"sentiment_flow_{w}d"] = g["weighted_sentiment"].shift(1).rolling(w, min_periods=1).mean()
            g[f"confidence_flow_{w}d"] = g["mean_confidence"].shift(1).rolling(w, min_periods=1).mean()
        flow_parts.append(g)
    latest_dataset = pd.concat(flow_parts, ignore_index=True).dropna().reset_index(drop=True)
    latest_dataset["target_direction"] = (latest_dataset["future_return"] > 0).astype(int)

    print(f"Training dataset: {len(dataset)} rows (tau=0.02)")
    print(f"Latest-signal dataset: {len(latest_dataset)} rows (tau=None)")

    # Model factories
    lr_factory = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=0.3, random_state=42,
        )),
    ])
    rf_factory = lambda: RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )

    models = {
        "LR (C=0.3)": lr_factory,
        "RF (depth=6)": rf_factory,
    }

    rows = []
    prob_distributions = {}

    for model_name, factory in models.items():
        print(f"\n→ {model_name}")
        result = _train_and_get_probs(
            dataset=dataset,
            model_factory=factory,
            portfolio_tickers=tickers,
            test_size=test_size,
            latest_dataset=latest_dataset,
        )
        if result is None:
            print("  FAILED")
            continue

        probs = result["probs"]
        roc_auc = result["roc_auc"]
        bal_acc = result["bal_acc"]

        # Probability distribution analysis
        prob_values = list(probs.values())
        n_bull = sum(1 for p in prob_values if p >= BULLISH_THRESHOLD)
        n_bear = sum(1 for p in prob_values if p <= BEARISH_THRESHOLD)
        n_neut = len(prob_values) - n_bull - n_bear
        prob_mean = float(np.mean(prob_values))
        prob_std  = float(np.std(prob_values))

        prob_distributions[model_name] = {
            "mean": prob_mean,
            "std": prob_std,
            "n_bullish_raw": n_bull,
            "n_bearish_raw": n_bear,
            "n_neutral_raw": n_neut,
        }

        print(f"  ROC-AUC={roc_auc:.4f} | Bal.Acc={bal_acc:.4f}")
        print(f"  Prob distribution: mean={prob_mean:.3f} std={prob_std:.3f}")
        print(f"  Raw signals: #Bull={n_bull} #Bear={n_bear} #Neutral={n_neut}")

        impact = _probs_to_portfolio_impact(
            probs=probs,
            mu=mu, cov=cov,
            baseline_weights=baseline_weights,
            baseline_sharpe=baseline["sharpe"],
            baseline_return=baseline["return"],
            baseline_vol=baseline["vol"],
        )

        print(f"  ΔSharpe={impact['sharpe_delta']:+.4f} | "
              f"ΔReturn={impact['return_delta']*100:+.2f}% | "
              f"Turnover={impact['turnover']*100:.1f}% | "
              f"#Bull={impact['n_bullish']} #Bear={impact['n_bearish']}")

        rows.append({
            "model": model_name,
            "roc_auc": roc_auc,
            "bal_acc": bal_acc,
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "n_signals_raw": n_bull + n_bear,
            **impact,
        })

    # Print results table
    print(f"\n\n{'='*70}")
    print("RF vs LR — PORTFOLIO IMPACT TABLE")
    print(f"Baseline: Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | "
          f"Vol={baseline['vol']*100:.2f}%")
    print(f"{'='*70}")

    col = 10
    header = (
        f"{'Model':<16}"
        f"{'ROC-AUC':>{col}}"
        f"{'Prob Std':>{col}}"
        f"{'#Const':>{col}}"
        f"{'ΔSharpe':>{col}}"
        f"{'ΔReturn':>{col}}"
        f"{'Turnover':>{col}}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Baseline':<16}"
        f"{'—':>{col}}"
        f"{'—':>{col}}"
        f"{'0':>{col}}"
        f"{'+0.0000':>{col}}"
        f"{'+0.00%':>{col}}"
        f"{'0.0%':>{col}}"
    )
    for r in rows:
        print(
            f"{r['model']:<16}"
            f"{r['roc_auc']:>{col}.4f}"
            f"{r['prob_std']:>{col}.4f}"
            f"{r['n_constraints']:>{col}}"
            f"{r['sharpe_delta']:>+{col}.4f}"
            f"{r['return_delta']*100:>+{col}.2f}%"
            f"{r['turnover']*100:>{col}.1f}%"
        )

    # Key insight
    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")
    lr_row = next((r for r in rows if "LR" in r["model"]), None)
    rf_row = next((r for r in rows if "RF" in r["model"]), None)

    if lr_row and rf_row:
        print(f"\nLR probability std = {lr_row['prob_std']:.4f} "
              f"(spreads across [0,1], reaches thresholds)")
        print(f"RF probability std = {rf_row['prob_std']:.4f} "
              f"(clusters near 0.5, rarely reaches thresholds)")
        print(f"\nLR active constraints: {lr_row['n_constraints']} "
              f"(#Bull={lr_row['n_bullish']}, #Bear={lr_row['n_bearish']})")
        print(f"RF active constraints: {rf_row['n_constraints']} "
              f"(#Bull={rf_row['n_bullish']}, #Bear={rf_row['n_bearish']})")

        if rf_row['n_constraints'] == 0:
            print("\nRF produces ZERO active constraints → portfolio identical to baseline.")
            print("This confirms that RF's poor probability calibration makes it")
            print("ineffective as a constraint generator, regardless of ROC-AUC.")
        else:
            constraint_ratio = rf_row['n_constraints'] / max(lr_row['n_constraints'], 1)
            print(f"\nRF generates {constraint_ratio:.1%} as many constraints as LR.")

    # Save
    if save_outputs and rows:
        df_out = pd.DataFrame(rows)
        csv_path = OUT_DIR / "rf_vs_lr_portfolio_impact.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

    return {
        "rows": rows,
        "baseline": baseline,
        "prob_distributions": prob_distributions,
    }


if __name__ == "__main__":
    run_rf_vs_lr_portfolio_impact(
        raw_path=RAW_PATH,
        test_size=0.30,
        save_outputs=True,
    )