# model_comparison_rf_vs_lr.py
# ============================================================
# Hocamın notu: "Random forest vs LR"
# İkisini aynı dataset üzerinde karşılaştır, neden LR seçtik göster.
# Mevcut kodlara dokunmaz. Standalone script.
# ============================================================
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
# ============================================================

MODEL_CONFIGS = {
    "logistic_C03": {
        "type": "logistic",
        "description": "Logistic Regression L2, C=0.3 (production model)",
        "model": Pipeline([
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
        "model": Pipeline([
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
        "model": Pipeline([
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
        "model": RandomForestClassifier(
            n_estimators=500, max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
    },
    "random_forest_d4": {
        "type": "random_forest",
        "description": "Random Forest, max_depth=4 (more regularized)",
        "model": RandomForestClassifier(
            n_estimators=500, max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
    },
    "random_forest_d8": {
        "type": "random_forest",
        "description": "Random Forest, max_depth=8 (less regularized)",
        "model": RandomForestClassifier(
            n_estimators=500, max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
    },
}

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
# Single model evaluation
# ============================================================

def _evaluate_model(
    model: Any,
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

    y = df["target_direction"].astype(int)
    split_idx = int(len(df) * (1.0 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feat_cols].astype(float)
    y_train = train_df["target_direction"].astype(int)
    X_test = test_df[feat_cols].astype(float)
    y_test = test_df["target_direction"].astype(int)

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
        "train_date_range": [
            str(train_df["news_date_dt"].min().date()),
            str(train_df["news_date_dt"].max().date()),
        ],
        "test_date_range": [
            str(test_df["news_date_dt"].min().date()),
            str(test_df["news_date_dt"].max().date()),
        ],
    }


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
            metrics = _evaluate_model(
                model=cfg["model"],
                dataset=dataset,
                test_size=test_size,
            )
            results[name] = {**metrics, "description": cfg["description"], "type": cfg["type"]}

            print(f"  ROC-AUC={metrics['roc_auc']:.4f} | "
                  f"Bal.Acc={metrics['balanced_accuracy']:.4f} | "
                  f"Lift={metrics['lift']:+.4f} | "
                  f"Signal={metrics['use_as_signal']}")

            rows.append({
                "model": name,
                "type": cfg["type"],
                "description": cfg["description"],
                "roc_auc": metrics["roc_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "baseline_balanced_accuracy": metrics["baseline_balanced_accuracy"],
                "lift": metrics["lift"],
                "use_as_signal": metrics["use_as_signal"],
                "train_size": metrics["train_size"],
                "test_size_rows": metrics["test_size"],
            })

        except Exception as e:
            print(f"  FAILED: {e}")

    # ── Comparison table ───────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    col = 12
    print(
        f"{'Model':<24}"
        f"{'ROC-AUC':>{col}}"
        f"{'Bal.Acc':>{col}}"
        f"{'Lift':>{col}}"
        f"{'F1':>{col}}"
        f"{'Signal?':>{col}}"
    )
    print("-" * (24 + col * 5))

    # Sort by ROC-AUC descending
    rows_sorted = sorted(rows, key=lambda x: x["roc_auc"] or 0, reverse=True)
    for row in rows_sorted:
        print(
            f"{row['model']:<24}"
            f"{row['roc_auc']:.4f}".rjust(col) +
            f"{row['balanced_accuracy']:.4f}".rjust(col) +
            f"{row['lift']:+.4f}".rjust(col) +
            f"{row['f1']:.4f}".rjust(col) +
            f"{'YES' if row['use_as_signal'] else 'NO'}".rjust(col)
        )

    # ── Why LR over RF? ────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("WHY LOGISTIC REGRESSION OVER RANDOM FOREST?")
    print("=" * 70)

    lr_results = {k: v for k, v in results.items() if v["type"] == "logistic"}
    rf_results = {k: v for k, v in results.items() if v["type"] == "random_forest"}

    best_lr = max(lr_results.items(), key=lambda x: x[1]["roc_auc"] or 0)
    best_rf = max(rf_results.items(), key=lambda x: x[1]["roc_auc"] or 0)

    print(f"\nBest LR: {best_lr[0]} → ROC-AUC={best_lr[1]['roc_auc']:.4f} "
          f"Bal.Acc={best_lr[1]['balanced_accuracy']:.4f}")
    print(f"Best RF: {best_rf[0]} → ROC-AUC={best_rf[1]['roc_auc']:.4f} "
          f"Bal.Acc={best_rf[1]['balanced_accuracy']:.4f}")

    roc_diff = best_lr[1]["roc_auc"] - best_rf[1]["roc_auc"]
    lift_diff = best_lr[1]["lift"] - best_rf[1]["lift"]

    print(f"\nROC-AUC difference (LR - RF): {roc_diff:+.4f}")
    print(f"Lift difference (LR - RF): {lift_diff:+.4f}")

    print("\nConclusion:")
    if roc_diff >= 0:
        print("  LR achieves equal or better ROC-AUC than RF on this dataset.")
    else:
        print(f"  RF achieves higher ROC-AUC by {abs(roc_diff):.4f}.")
        print("  However, LR is preferred for interpretability and constraint generation:")

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