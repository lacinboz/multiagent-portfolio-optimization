from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"

N_RUNS = 5  #  consistent with feature_ablation script

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
# Model configs to compare
#  Now use FACTORIES (callables) instead of pre-built instances,
#    so each of the N_RUNS gets a fresh model with fresh randomness.
#  RF configs no longer pass random_state (matches feature_ablation.py).
#  LR configs keep random_state=42: LR with the default 'lbfgs'
#    solver is deterministic given fixed data, so repeated runs
#    are expected to produce identical results (std ≈ 0).
# ============================================================

MODEL_CONFIGS = {
    "logistic_C03": {
        "type": "logistic",
        "description": "Logistic Regression L2, C=0.3 (production model)",
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                C=0.3, random_state=42,
            )),
        ]),
    },
    "logistic_C01": {
        "type": "logistic",
        "description": "Logistic Regression L2, C=0.1 (stronger regularization)",
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                C=0.1, random_state=42,
            )),
        ]),
    },
    "logistic_C10": {
        "type": "logistic",
        "description": "Logistic Regression L2, C=1.0 (weaker regularization)",
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                C=1.0, random_state=42,
            )),
        ]),
    },
    "random_forest_d6": {
        "type": "random_forest",
        "description": "Random Forest, max_depth=6 (production config)",
        "factory": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,  #  no random_state
        ),
    },
    "random_forest_d4": {
        "type": "random_forest",
        "description": "Random Forest, max_depth=4 (more regularized)",
        "factory": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,  #  no random_state
        ),
    },
    "random_forest_d8": {
        "type": "random_forest",
        "description": "Random Forest, max_depth=8 (less regularized)",
        "factory": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,  #  no random_state
        ),
    },
}

# ============================================================
# Dataset builder (unchanged)
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
# Single model evaluation (single run)
# ============================================================

def _evaluate_model_once(
    model_factory,
    dataset: pd.DataFrame,
    test_size: float = 0.30,
    use_ticker_features: bool = True,
) -> Dict[str, Any]:
    available = [f for f in FEATURE_COLS if f in dataset.columns]
    df = dataset.dropna(
        subset=available + ["target_direction", "news_date_dt", "ticker"]
    ).copy()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    feat_cols = available.copy()
    if use_ticker_features:
        dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        feat_cols += list(dummies.columns)

    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feat_cols].astype(float)
    y_train = train_df["target_direction"].astype(int)
    X_test = test_df[feat_cols].astype(float)
    y_test = test_df["target_direction"].astype(int)

    model = model_factory()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    majority = int(y_train.mode().iloc[0])
    baseline_pred = np.full(len(y_test), majority)

    roc_auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() == 2 else None
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    baseline_bal = float(balanced_accuracy_score(y_test, baseline_pred))

    return {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "baseline_balanced_accuracy": baseline_bal,
        "lift": float(bal_acc - baseline_bal),
        "use_as_signal": bool(
            roc_auc is not None and roc_auc >= 0.55
            and (bal_acc - baseline_bal) > 0.0
        ),
    }


def _aggregate_runs(name: str, description: str, model_type: str,
                     run_results: List[Dict]) -> Dict[str, Any]:
    """✅ Aggregate N_RUNS results into mean±std, matching
    news_model_feature_ablation.py's _aggregate()."""
    out: Dict[str, Any] = {
        "model": name,
        "type": model_type,
        "description": description,
        "n_runs": len(run_results),
    }
    metrics = [
        "roc_auc", "balanced_accuracy", "accuracy", "precision",
        "recall", "f1", "lift",
    ]
    for m in metrics:
        vals = [r[m] for r in run_results if r.get(m) is not None]
        if vals:
            out[f"{m}_mean"] = float(np.mean(vals))
            out[f"{m}_std"] = float(np.std(vals))
    out["use_as_signal"] = bool(
        out.get("roc_auc_mean", 0) >= 0.55 and out.get("lift_mean", 0) > 0.0
    )
    return out


# ============================================================
# Main comparison
# ============================================================

def run_rf_vs_lr_comparison(
    raw_path: str = RAW_PATH,
    test_size: float = 0.30,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("RANDOM FOREST vs LOGISTIC REGRESSION COMPARISON")
    print(f"N_RUNS={N_RUNS} | No fixed seed for RF (matches feature_ablation.py)")
    print("=" * 70)

    print("\nBuilding dataset...")
    dataset = _build_dataset(raw_path)
    print(f"Rows: {len(dataset)}, tickers: {dataset['ticker'].nunique()}")
    print(f"Class balance: {dataset['target_direction'].value_counts().to_dict()}")

    results = {}
    rows = []

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n→ {name}: {cfg['description']}")
        try:
            run_results = []
            for run_idx in range(N_RUNS):
                m = _evaluate_model_once(
                    model_factory=cfg["factory"],
                    dataset=dataset,
                    test_size=test_size,
                )
                run_results.append(m)

            agg = _aggregate_runs(name, cfg["description"], cfg["type"], run_results)
            results[name] = agg

            print(f"  ROC-AUC={agg.get('roc_auc_mean', 0):.4f}"
                  f"±{agg.get('roc_auc_std', 0):.4f} | "
                  f"Bal.Acc={agg.get('balanced_accuracy_mean', 0):.4f}"
                  f"±{agg.get('balanced_accuracy_std', 0):.4f} | "
                  f"Lift={agg.get('lift_mean', 0):+.4f}"
                  f"±{agg.get('lift_std', 0):.4f} | "
                  f"Signal={agg['use_as_signal']}")

            rows.append(agg)

        except Exception as e:
            print(f"  FAILED: {e}")

    # ── Comparison table ───────────────────────────────────────
    print("\n\n" + "=" * 70)
    print(f"COMPARISON TABLE (mean±std, {N_RUNS} runs)")
    print("=" * 70)
    col = 18
    print(
        f"{'Model':<20}"
        f"{'ROC-AUC':>{col}}"
        f"{'Bal.Acc':>{col}}"
        f"{'F1':>{col}}"
        f"{'Lift':>{col}}"
        f"{'Signal?':>10}"
    )
    print("-" * (20 + col * 4 + 10))

    rows_sorted = sorted(rows, key=lambda x: x.get("roc_auc_mean", 0), reverse=True)
    for row in rows_sorted:
        roc = f"{row.get('roc_auc_mean', 0):.4f}±{row.get('roc_auc_std', 0):.4f}"
        bal = f"{row.get('balanced_accuracy_mean', 0):.4f}±{row.get('balanced_accuracy_std', 0):.4f}"
        f1v = f"{row.get('f1_mean', 0):.4f}±{row.get('f1_std', 0):.4f}"
        lift = f"{row.get('lift_mean', 0):+.4f}±{row.get('lift_std', 0):.4f}"
        print(
            f"{row['model']:<20}"
            f"{roc:>{col}}"
            f"{bal:>{col}}"
            f"{f1v:>{col}}"
            f"{lift:>{col}}"
            f"{'YES' if row['use_as_signal'] else 'NO':>10}"
        )

    # ── Why LR over RF? ────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("WHY LOGISTIC REGRESSION OVER RANDOM FOREST?")
    print("=" * 70)

    lr_results = {k: v for k, v in results.items() if v["type"] == "logistic"}
    rf_results = {k: v for k, v in results.items() if v["type"] == "random_forest"}

    best_lr = max(lr_results.items(), key=lambda x: x[1].get("roc_auc_mean", 0))
    best_rf = max(rf_results.items(), key=lambda x: x[1].get("roc_auc_mean", 0))

    print(f"\nBest LR: {best_lr[0]} → ROC-AUC={best_lr[1]['roc_auc_mean']:.4f}"
          f"±{best_lr[1]['roc_auc_std']:.4f}")
    print(f"Best RF: {best_rf[0]} → ROC-AUC={best_rf[1]['roc_auc_mean']:.4f}"
          f"±{best_rf[1]['roc_auc_std']:.4f}")

    roc_diff = best_lr[1]["roc_auc_mean"] - best_rf[1]["roc_auc_mean"]
    lift_diff = best_lr[1]["lift_mean"] - best_rf[1]["lift_mean"]

    print(f"\nROC-AUC difference (LR - RF): {roc_diff:+.4f}")
    print(f"Lift difference (LR - RF): {lift_diff:+.4f}")

    print("\nConclusion:")
    if roc_diff >= 0:
        print("  LR achieves equal or better ROC-AUC than RF on this dataset.")
    else:
        print(f"  RF achieves higher ROC-AUC by {abs(roc_diff):.4f}.")
    print("  LR is preferred for interpretability and constraint generation:")
    print("  1. LR produces calibrated probabilities → better threshold behavior")
    print("  2. LR coefficients are directly interpretable")
    print("  3. LR with L2 regularization (C=0.3) is less prone to overfitting")
    print("     on financial time-series data with high noise")
    print("  4. LR is faster to retrain when new data arrives")

    # ── Save outputs ───────────────────────────────────────────
    if save_outputs:
        df = pd.DataFrame(rows_sorted)
        csv_path = OUT_DIR / "rf_vs_lr_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        json_path = OUT_DIR / "rf_vs_lr_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[Saved] {json_path}")

    return results


if __name__ == "__main__":
    run_rf_vs_lr_comparison(
        raw_path=RAW_PATH,
        test_size=0.30,
        save_outputs=True,
    )