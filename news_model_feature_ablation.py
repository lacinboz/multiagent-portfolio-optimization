# news_model_feature_ablation.py
# ============================================================
# Hocamın istediği: "news features without/with + different feature sets"
# Mevcut news_return_predictor.py'e dokunmaz.
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"

# ============================================================
# Feature set definitions
# ============================================================

# Tüm mevcut feature'lar (news_return_predictor.py'deki get_feature_cols ile aynı)
ALL_FEATURES = [
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

# Sadece price features (news yok)
PRICE_ONLY_FEATURES = [
    "past_5d_return",
    "past_20d_return",
    "past_20d_volatility",
]

# Sadece news features (price yok)
NEWS_ONLY_FEATURES = [
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
]

# News features ama flow yok (sadece anlık)
NEWS_NO_FLOW_FEATURES = [
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
]

# FinBERT probability features + price (sentiment score yok)
FINBERT_PROB_FEATURES = [
    "positive_ratio",
    "negative_ratio",
    "mean_confidence",
    "sentiment_flow_5d",
    "sentiment_flow_20d",
    "past_5d_return",
    "past_20d_return",
    "past_20d_volatility",
]

FEATURE_SETS = {
    "price_only": {
        "features": PRICE_ONLY_FEATURES,
        "description": "Only price features (no news) — pure technical baseline",
    },
    "news_only": {
        "features": NEWS_ONLY_FEATURES,
        "description": "Only news/sentiment features (no price data)",
    },
    "news_no_flow": {
        "features": NEWS_NO_FLOW_FEATURES,
        "description": "News + price, but no rolling flow features",
    },
    "finbert_prob_price": {
        "features": FINBERT_PROB_FEATURES,
        "description": "FinBERT probability features + price (no raw sentiment score)",
    },
    "all_features": {
        "features": ALL_FEATURES,
        "description": "Full feature set — all news + price + flow (production model)",
    },
}

# ============================================================
# Dataset builder (same logic as news_return_predictor.py
# build_ticker_date_prediction_dataset_v2, but standalone)
# ============================================================

def _build_dataset(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError("Raw dataset is empty.")

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

    # rolling flow features
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
# Single feature set experiment
# ============================================================

def _run_one_feature_set(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    *,
    test_size: float = 0.30,
    C: float = 0.3,
    random_state: int = 42,
    use_ticker_features: bool = True,
) -> Dict[str, Any]:

    # keep only rows where ALL requested features are available
    available = [f for f in feature_cols if f in dataset.columns]
    missing = [f for f in feature_cols if f not in dataset.columns]

    df = dataset.dropna(subset=available + ["target_direction", "news_date_dt", "ticker"]).copy()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    feat_cols = available.copy()

    if use_ticker_features:
        dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        feat_cols += list(dummies.columns)

    y = df["target_direction"].astype(int)
    if y.nunique() < 2:
        return {"ok": False, "reason": "Only one class in dataset."}

    split_idx = int(len(df) * (1.0 - test_size))
    if split_idx <= 10 or split_idx >= len(df) - 10:
        return {"ok": False, "reason": "Dataset too small."}

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feat_cols].astype(float)
    X_test = test_df[feat_cols].astype(float)
    y_train = train_df["target_direction"].astype(int)
    y_test = test_df["target_direction"].astype(int)

    if y_train.nunique() < 2:
        return {"ok": False, "reason": "Training set has only one class."}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=C, random_state=random_state,
        )),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    majority = int(y_train.mode().iloc[0])
    baseline_pred = np.full(len(y_test), majority, dtype=int)

    acc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    baseline_acc = float(accuracy_score(y_test, baseline_pred))
    baseline_bal = float(balanced_accuracy_score(y_test, baseline_pred))

    roc_auc = None
    if y_test.nunique() == 2:
        roc_auc = float(roc_auc_score(y_test, proba))

    return {
        "ok": True,
        "n_features": len(available),
        "missing_features": missing,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "baseline_accuracy": baseline_acc,
        "baseline_balanced_accuracy": baseline_bal,
        "lift_accuracy": float(acc - baseline_acc),
        "lift_balanced_accuracy": float(bal_acc - baseline_bal),
        "roc_auc": roc_auc,
        "use_as_signal": bool(
            roc_auc is not None and roc_auc >= 0.55
            and (bal_acc - baseline_bal) > 0.0
        ),
    }


# ============================================================
# Full feature ablation runner
# ============================================================

def run_feature_ablation_study(
    raw_path: str = RAW_PATH,
    test_size: float = 0.30,
    C: float = 0.3,
    use_ticker_features: bool = True,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("NEWS MODEL FEATURE ABLATION STUDY")
    print("=" * 70)
    print(f"Raw path: {raw_path}")
    print(f"test_size={test_size}, C={C}, use_ticker_features={use_ticker_features}")

    print("\n[Step 1] Building dataset...")
    dataset = _build_dataset(raw_path)
    print(f"Rows: {len(dataset)}, tickers: {dataset['ticker'].nunique()}")
    print(f"Class balance: {dataset['target_direction'].value_counts().to_dict()}")

    results: Dict[str, Any] = {}

    print("\n[Step 2] Running each feature set...")
    for name, cfg in FEATURE_SETS.items():
        print(f"\n  → {name}: {cfg['description']}")
        res = _run_one_feature_set(
            dataset,
            cfg["features"],
            test_size=test_size,
            C=C,
            use_ticker_features=use_ticker_features,
        )
        results[name] = {**res, "description": cfg["description"]}
        if res.get("ok"):
            print(f"     ROC-AUC={res.get('roc_auc', 0):.4f} | "
                  f"Balanced acc={res.get('balanced_accuracy', 0):.4f} | "
                  f"Lift={res.get('lift_balanced_accuracy', 0):+.4f} | "
                  f"Use as signal: {res.get('use_as_signal')}")
        else:
            print(f"     FAILED: {res.get('reason')}")

    # ── Print comparison table ─────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FEATURE ABLATION COMPARISON TABLE")
    print("=" * 70)

    col = 14
    header = (
        f"{'Feature set':<22}"
        f"{'ROC-AUC':>{col}}"
        f"{'Bal. acc':>{col}}"
        f"{'Lift':>{col}}"
        f"{'#Features':>{col}}"
        f"{'Signal?':>{col}}"
    )
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        if not res.get("ok"):
            print(f"{name:<22}{'FAILED':>{col * 5}}")
            continue
        print(
            f"{name:<22}"
            f"{res.get('roc_auc', 0):.4f}".rjust(col) +
            f"{res.get('balanced_accuracy', 0):.4f}".rjust(col) +
            f"{res.get('lift_balanced_accuracy', 0):+.4f}".rjust(col) +
            f"{res.get('n_features', 0)}".rjust(col) +
            f"{'YES' if res.get('use_as_signal') else 'NO'}".rjust(col)
        )

    # ── Save outputs ───────────────────────────────────────────
    if save_outputs:
        rows = []
        for name, res in results.items():
            rows.append({
                "feature_set": name,
                "description": res.get("description", ""),
                "n_features": res.get("n_features"),
                "roc_auc": res.get("roc_auc"),
                "balanced_accuracy": res.get("balanced_accuracy"),
                "baseline_balanced_accuracy": res.get("baseline_balanced_accuracy"),
                "lift_balanced_accuracy": res.get("lift_balanced_accuracy"),
                "accuracy": res.get("accuracy"),
                "lift_accuracy": res.get("lift_accuracy"),
                "use_as_signal": res.get("use_as_signal"),
                "train_size": res.get("train_size"),
                "test_size_rows": res.get("test_size"),
                "missing_features": str(res.get("missing_features", [])),
                "ok": res.get("ok"),
            })

        csv_path = OUT_DIR / "feature_ablation_results.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        json_path = OUT_DIR / "feature_ablation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[Saved] {json_path}")

    return results


if __name__ == "__main__":
    run_feature_ablation_study(
        raw_path=RAW_PATH,
        test_size=0.30,
        C=0.3,
        use_ticker_features=True,
        save_outputs=True,
    )