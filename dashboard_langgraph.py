# dashboard_langgraph_app.py
# ✅ Updated for:
# - Option A candidate selection (LLM chooses maxsharpe/minvar)
# - SAFE metric rendering (no crashes when sharpe/metrics missing)
# - Insight panel supports BOTH:
#     - structured JSON insight in state["insight"]
#     - narrative insight in state["insight_raw_text"]
# - Streamlit-safe container widths (use_container_width=True)
# - Safe delta computations (no None - None crashes)
# - ✅ NEW: News Snapshot / News Risk Check UI rendering
#     - supports BOTH placeholder news_signals and your new llm_client.generate_news_snapshot() output
#     - shows a short snapshot + per-ticker risk flags if present
# - ✅ NEW: News is OPTIONAL in refine (gated by UI checkbox)
#     - sends refined_answers["use_news"] = "yes"/"no"
#     - passes use_news=... into run_graph
#
# Notes:
# - Base run: mode="base" use_llm=False use_news=False
# - Refine run: mode="refine" use_llm=True/False use_news=True/False
# - Prefer finalized outputs:
#     state["optimized_weights"], state["optimized_metrics"], state["insight"/"insight_raw_text"]
#   fallback to optimization_result[chosen] only if optimized_* absent.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from portfolio_langgraph_withllm import (
    run_graph,
    run_graph_prob_news,
    run_graph_prediction_constraint,
)
from llm_client import LLMClient
NEWS_PRED_DIR = Path("data/news_prediction")
BEST_NEWS_PREDICTIONS_PATH = NEWS_PRED_DIR / "best_news_prediction_predictions.csv"
BEST_NEWS_METRICS_PATH = NEWS_PRED_DIR / "best_news_prediction_metrics.json"
BEST_NEWS_MODEL_PATH = NEWS_PRED_DIR / "best_news_prediction_model.joblib"
LATEST_NEWS_SIGNALS_PATH = NEWS_PRED_DIR / "latest_news_prediction_signals.csv"
TICKER_ALIASES_FOR_NEWS = {
    "GOOGL": ["GOOGL", "GOOG"],
    "GOOG": ["GOOG", "GOOGL"],
}
DATA_DIR = Path("data/processed_yahoo")

from dotenv import load_dotenv
load_dotenv()

@st.cache_data
def load_available_tickers() -> list[str]:
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    return list(map(str, summary.index))

def _expand_tickers_for_news(selected_tickers: List[str]) -> List[str]:
    expanded = []

    for t in selected_tickers or []:
        tt = str(t).upper().strip()
        aliases = TICKER_ALIASES_FOR_NEWS.get(tt, [tt])

        for a in aliases:
            if a not in expanded:
                expanded.append(a)

    return expanded
def _safe_normalize_current_inputs(df: pd.DataFrame, mode: str) -> Optional[dict[str, float]]:
    if df is None or df.empty or "Value" not in df.columns:
        return None

    x = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if mode == "Percent (%)":
        x = x / 100.0

    s = float(x.sum())
    if s <= 0:
        return None

    w = x / s
    return {str(t): float(v) for t, v in w.items()}


def _get_chosen_candidate(state: Dict[str, Any]) -> str:
    chosen = state.get("chosen_candidate") or state.get("objective_key") or "maxsharpe"
    chosen = str(chosen).lower().strip()
    return chosen or "maxsharpe"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    aa = _safe_float(a)
    bb = _safe_float(b)
    if aa is None or bb is None:
        return None
    return float(aa - bb)


def _extract_weights_and_metrics(state: Dict[str, Any]):
    """
     Option A compatible extraction.
    Prefer FINAL outputs:
      - optimized_weights (final selection)
      - optimized_metrics (final selection risk metrics)
    Fallback:
      - optimization_result[chosen] if optimized_* absent.
    """
    optimization_result = state.get("optimization_result") or {}
    chosen = _get_chosen_candidate(state)

    weights_series = None
    portfolio_metrics = None

    # 1) Prefer finalized weights/metrics
    opt_w = state.get("optimized_weights") or {}
    opt_m = state.get("optimized_metrics") or {}

    if opt_w:
        w = pd.Series(opt_w, dtype=float)
        w = w[w.abs() > 1e-6].sort_values(ascending=False)
        weights_series = w

        sharpe = _safe_float(opt_m.get("sharpe"))
        ret = _safe_float(opt_m.get("return"))
        vol = _safe_float(opt_m.get("vol"))

        ret_pct = _safe_float(opt_m.get("return_pct"))
        vol_pct = _safe_float(opt_m.get("vol_pct"))

        active_assets = opt_m.get("active_assets", None)
        try:
            active_assets = int(active_assets) if active_assets is not None else int(len(w))
        except Exception:
            active_assets = int(len(w))

        portfolio_metrics = {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "return_pct": ret_pct,
            "vol_pct": vol_pct,
            "sharpe": sharpe,
            "used_assets": int(len(w)),
            "universe_assets": int(len(state.get("selected_tickers", []))),
            "active_assets": active_assets,
        }
        return optimization_result, chosen, weights_series, portfolio_metrics

    # 2) Fallback: read directly from optimization_result[chosen]
    if optimization_result and chosen in optimization_result:
        port = optimization_result[chosen]
        w = pd.Series(port.get("weights", {}), dtype=float)
        w = w[w.abs() > 1e-6].sort_values(ascending=False)
        weights_series = w

        sharpe = _safe_float(port.get("sharpe"))
        ret = _safe_float(port.get("return"))
        vol = _safe_float(port.get("vol"))

        portfolio_metrics = {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "return_pct": None,
            "vol_pct": None,
            "sharpe": sharpe,
            "used_assets": int(len(w)),
            "universe_assets": int(len(state.get("selected_tickers", []))),
            "active_assets": int(len(w)),
        }

    return optimization_result, chosen, weights_series, portfolio_metrics

@st.cache_data
def load_saved_news_prediction_outputs() -> Dict[str, Any]:
    pred_path = BEST_NEWS_PREDICTIONS_PATH
    metrics_path = BEST_NEWS_METRICS_PATH
    model_path = BEST_NEWS_MODEL_PATH

    if not pred_path.exists():
        return {
            "ok": False,
            "reason": f"Best prediction file not found: {pred_path}",
        }

    pred_df = pd.read_csv(pred_path)
    latest_signals = pd.DataFrame()
    

    if LATEST_NEWS_SIGNALS_PATH.exists():
        latest_signals = pd.read_csv(LATEST_NEWS_SIGNALS_PATH)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    model_bundle = {}
    if model_path.exists():
        try:
            model_bundle = joblib.load(model_path)
        except Exception as e:
            model_bundle = {"load_error": str(e)}

    explanation = {}
    if isinstance(model_bundle, dict):
        explanation = model_bundle.get("model_explanation") or {}

    return {
        "ok": True,
        "predictions": pred_df,
        "metrics": metrics,
        "explanation": explanation,
        "model_bundle": model_bundle,
        "pred_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "latest_signals": latest_signals,
        "latest_signals_path": str(LATEST_NEWS_SIGNALS_PATH),
    }

def _prediction_direction_label(x: Any) -> str:
    try:
        return "Positive" if int(x) == 1 else "Negative"
    except Exception:
        return "–"

def _correct_icon(x: Any) -> str:
    try:
        return "Matched realized direction" if bool(x) else "Did not match realized direction"
    except Exception:
        return "–"

def _prepare_prediction_display_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["news_date_dt"] = pd.to_datetime(df["news_date_dt"], errors="coerce")
    df["predicted_positive_probability"] = pd.to_numeric(
        df["predicted_positive_probability"], errors="coerce"
    )
    df["future_return"] = pd.to_numeric(df["future_return"], errors="coerce")

    if "predicted_direction" in df.columns:
        df["Predicted direction"] = df["predicted_direction"].apply(_prediction_direction_label)
    else:
        df["Predicted direction"] = np.where(
            df["predicted_positive_probability"] >= 0.5,
            "Positive",
            "Negative",
        )

    if "target_direction" in df.columns:
        df["Actual direction"] = df["target_direction"].apply(_prediction_direction_label)
    else:
        df["Actual direction"] = np.where(df["future_return"] > 0, "Positive", "Negative")

    if "correct" in df.columns:
        df["Correct?"] = df["correct"].apply(_correct_icon)
    else:
        df["Correct?"] = np.where(
            df["Predicted direction"] == df["Actual direction"],
            "Matched realized direction",
            "Did not match realized direction",
        )

    df["confidence_distance"] = (df["predicted_positive_probability"] - 0.5).abs()

    df["Signal strength"] = pd.cut(
        df["confidence_distance"],
        bins=[-0.001, 0.05, 0.20, 0.50],
        labels=["Weak signal", "Medium signal", "Strong signal"],
    )

    return df

def _coefficient_interpretation(feature: str, coef: float) -> str:
    f = str(feature).lower()
    sign = "positive" if coef > 0 else "negative"

    if "positive_prob" in f and coef < 0:
        return "High positive-news flow may indicate overreaction or already-priced-in optimism."
    if "positive_prob" in f and coef > 0:
        return "Positive news flow is associated with higher probability of positive future returns."
    if "negative_prob" in f and coef < 0:
        return "Negative-news flow lowers the predicted probability of positive future returns."
    if "negative_prob" in f and coef > 0:
        return "Negative-news flow may behave as a contrarian/rebound signal in this dataset."
    if "volatility" in f:
        return f"Past volatility has a {sign} relationship with predicted positive future return."
    if "momentum" in f or "return" in f:
        return f"Recent price movement has a {sign} relationship with predicted future direction."
    if "confidence" in f:
        return f"Model/news confidence has a {sign} relationship with predicted future direction."
    if "sentiment" in f:
        return f"News sentiment flow has a {sign} relationship with predicted future direction."

    return f"This feature has a {sign} coefficient in the logistic model."


def _render_confusion_matrix_heatmap(metrics: Dict[str, Any]):
    cm = metrics.get("confusion_matrix")
    if not cm:
        return

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        title="Confusion Matrix",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_probability_distribution(df: pd.DataFrame):
    if "predicted_positive_probability" not in df.columns:
        return

    plot_df = df.copy()
    plot_df["predicted_positive_probability"] = pd.to_numeric(
        plot_df["predicted_positive_probability"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["predicted_positive_probability"])

    if plot_df.empty:
        return

    fig = px.histogram(
        plot_df,
        x="predicted_positive_probability",
        nbins=20,
        title="Prediction Confidence Distribution",
    )

    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        xaxis_title="Predicted probability of positive return",
        yaxis_title="Number of predictions",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_xaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)


def _render_calibration_table(df: pd.DataFrame):
    required = {"predicted_positive_probability", "target_direction"}
    if not required.issubset(set(df.columns)):
        return

    cal = df.copy()
    cal["predicted_positive_probability"] = pd.to_numeric(
        cal["predicted_positive_probability"], errors="coerce"
    )
    cal["target_direction"] = pd.to_numeric(cal["target_direction"], errors="coerce")
    cal = cal.dropna(subset=["predicted_positive_probability", "target_direction"])

    if cal.empty:
        return

    cal["Probability bucket"] = pd.cut(
        cal["predicted_positive_probability"],
        bins=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        include_lowest=True,
    )

    bucket_df = (
        cal.groupby("Probability bucket", observed=False)
        .agg(
            predictions=("target_direction", "count"),
            avg_predicted_probability=("predicted_positive_probability", "mean"),
            realized_positive_rate=("target_direction", "mean"),
            directional_accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    bucket_labels = ["0–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80–100%"]

    bucket_df["Probability bucket"] = bucket_labels[: len(bucket_df)]

    st.markdown("**Probability calibration by bucket**")
    st.caption(
        "This checks whether higher predicted probabilities actually correspond to more positive realized returns."
    )

    st.dataframe(
        bucket_df.style.format(
            {
                "avg_predicted_probability": "{:.1%}",
                "realized_positive_rate": "{:.1%}",
                "directional_accuracy": "{:.1%}",
            },
            na_rep="–",
        ),
        use_container_width=True,
    )
def _render_ticker_level_prediction_summary(df_selected: pd.DataFrame):
    if df_selected is None or df_selected.empty:
        return

    latest_signal = (
        df_selected.sort_values(
            ["ticker", "confidence_distance", "news_date_dt"],
            ascending=[True, False, False],
        )
        .groupby("ticker", as_index=False)
        .head(1)
        .copy()
    )

    latest_signal["Model signal"] = latest_signal["Predicted direction"]
    latest_signal["Confidence"] = np.where(
        latest_signal["predicted_positive_probability"] >= 0.5,
        latest_signal["predicted_positive_probability"],
        1.0 - latest_signal["predicted_positive_probability"],
    )

    latest_signal["Interpretation"] = latest_signal.apply(
        lambda r: (
            f"The model leans {r['Model signal'].lower()} for this ticker "
            f"with {r['Confidence']:.1%} directional confidence."
        ),
        axis=1,
    )

    display_cols = [
        "ticker",
        "Model signal",
        "Confidence",
        "predicted_positive_probability",
        "Signal strength",
        "news_date",
        "Interpretation",
    ]

    if "headline" in latest_signal.columns:
        display_cols.insert(-1, "headline")

    display = latest_signal[display_cols]
    display = display.rename(
    columns={
        "ticker": "Ticker",
        "predicted_positive_probability": "Positive probability",
        "news_date": "News date",
        "headline": "Supporting headline",
    }
)

    st.markdown("**Ticker-level prediction summary**")
    st.caption(
        "This converts the saved article-level prediction results into one representative signal per selected ticker. "
        "It is still based on historical out-of-sample prediction rows, not a live trading recommendation."
    )

    st.dataframe(
        display.style.format(
            {
                "Confidence": "{:.1%}",
                "Positive probability": "{:.1%}",
                "Signal strength": "{}",
            },
            na_rep="–",
        ),
        use_container_width=True,
        height=300,
    )
def _render_latest_prediction_signals(latest_df: pd.DataFrame, selected_tickers: List[str]):
    if latest_df is None or latest_df.empty:
        st.info("No latest ticker-level prediction signals available yet.")
        return

    df = latest_df.copy()

    if selected_tickers:
        selected_upper = _expand_tickers_for_news(selected_tickers)
        df = df[df["ticker"].astype(str).str.upper().isin(selected_upper)]
        df["news_date_dt"] = pd.to_datetime(df["news_date_dt"], errors="coerce")


        df = (
            df.sort_values(["news_date_dt", "ticker"], ascending=[False, True])
            .groupby(df["ticker"].replace({"GOOG": "GOOGL"}), as_index=False)
            .head(1)
        )

    if df.empty:
        st.info("No latest prediction signals available for the selected tickers.")
        return

    df["predicted_positive_probability"] = pd.to_numeric(
        df["predicted_positive_probability"], errors="coerce"
    )
    df["prediction_confidence"] = pd.to_numeric(
        df["prediction_confidence"], errors="coerce"
    )

    display = df[
        [
            "ticker",
            "news_date",
            "predicted_positive_probability",
            "prediction_confidence",
            "signal_label",
            "signal_strength",
            "article_count",
            "weighted_sentiment",
            "mean_confidence",
            "positive_ratio",
            "negative_ratio",
        ]
    ].rename(
        columns={
            "ticker": "Ticker",
            "news_date": "Latest news date",
            "predicted_positive_probability": "Predicted positive probability",
            "prediction_confidence": "Prediction confidence",
            "signal_label": "Signal",
            "signal_strength": "Strength",
            "article_count": "Articles",
            "weighted_sentiment": "Weighted sentiment",
            "mean_confidence": "Mean FinBERT confidence",
            "positive_ratio": "Positive news ratio",
            "negative_ratio": "Negative news ratio",
        }
    )

    st.markdown("**Latest ticker-level prediction probabilities**")
    st.caption(
        "These are the latest per-ticker model probabilities generated from `latest_news_prediction_signals.csv`. "
        "They summarize the model's current directional signal for each selected asset."
    )

    st.dataframe(
        display.style.format(
            {
                "Predicted positive probability": "{:.1%}",
                "Prediction confidence": "{:.1%}",
                "Weighted sentiment": "{:.3f}",
                "Mean FinBERT confidence": "{:.1%}",
                "Positive news ratio": "{:.1%}",
                "Negative news ratio": "{:.1%}",
            },
            na_rep="–",
        ),
        use_container_width=True,
        height=360,
    )
def _render_prediction_weight_shift_chart(
    refined_state: Dict[str, Any],
    *,
    chart_key: str = "prediction_shift_chart",
):
    if not refined_state:
        return

    base_w = refined_state.get("baseline_candidate_weights") or {}
    final_w = refined_state.get("optimized_weights") or {}

    if not base_w or not final_w:
        return

    rows = []

    tickers = sorted(set(base_w.keys()) | set(final_w.keys()))

    for t in tickers:
        rows.append(
            {
                "Ticker": t,
                "Baseline": float(base_w.get(t, 0.0)),
                "Prediction-Constrained": float(final_w.get(t, 0.0)),
            }
        )

    df = pd.DataFrame(rows)

    long_df = df.melt(
        id_vars="Ticker",
        value_vars=["Baseline", "Prediction-Constrained"],
        var_name="Portfolio",
        value_name="Weight",
    )

    fig = px.bar(
        long_df,
        x="Ticker",
        y="Weight",
        color="Portfolio",
        barmode="group",
        text="Weight",
        title="Prediction Constraint Effect on Portfolio Weights",
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key,
    )
def _render_prediction_constraint_table(
    refined_state: Dict[str, Any],
):
    if not refined_state:
        return

    probs = refined_state.get("prediction_probs") or {}
    base_w = refined_state.get("baseline_candidate_weights") or {}
    final_w = refined_state.get("optimized_weights") or {}

    if not probs:
        return

    rows = []

    tickers = sorted(probs.keys())

    for t in tickers:
        p = probs.get(t)

        if p >= 0.60:
            signal = "Bullish"
        elif p <= 0.40:
            signal = "Bearish"
        else:
            signal = "Neutral"

        rows.append(
            {
                "Ticker": t,
                "Probability": p,
                "Signal": signal,
                "Old Weight": float(base_w.get(t, 0.0)),
                "New Weight": float(final_w.get(t, 0.0)),
                "Shift": float(final_w.get(t, 0.0))
                - float(base_w.get(t, 0.0)),
            }
        )

    df = pd.DataFrame(rows)

    st.markdown("### Constraint-Level Changes")

    st.dataframe(
        df.style.format(
            {
                "Probability": "{:.1%}",
                "Old Weight": "{:.1%}",
                "New Weight": "{:.1%}",
                "Shift": "{:+.1%}",
            }
        ),
        use_container_width=True,
    )
def _render_saved_news_prediction_model(
    selected_tickers: List[str],
    active_state: Optional[Dict[str, Any]] = None,
):
    out = load_saved_news_prediction_outputs()

    if not out.get("ok"):
        st.warning(out.get("reason", "Saved prediction output could not be loaded."))
        return

    df_all = _prepare_prediction_display_df(out["predictions"].copy())

    metrics = out.get("metrics") or {}
    explanation = out.get("explanation") or {}
    latest_signals = out.get("latest_signals")

    df_selected = df_all.copy()

    if selected_tickers:
        selected_upper = _expand_tickers_for_news(selected_tickers)
        df_selected = df_selected[
            df_selected["ticker"].astype(str).str.upper().isin(selected_upper)
        ]

    if df_selected.empty:
        st.info("No saved prediction rows available for the selected tickers.")
        return
    _render_latest_prediction_signals(latest_signals, selected_tickers)
    with st.expander("DEBUG latest signals raw rows"):
        googl_rows = latest_signals[
            latest_signals["ticker"].astype(str).str.upper().isin(["GOOGL", "GOOG"])
        ].copy()

        googl_rows["news_date_dt"] = pd.to_datetime(
            googl_rows["news_date_dt"],
            errors="coerce"
        )

        googl_rows = googl_rows.sort_values(
            "news_date_dt",
            ascending=False
        )

        st.dataframe(googl_rows, use_container_width=True)
        st.markdown("---")
    _render_ticker_level_prediction_summary(df_selected)

    st.markdown("---")

    latest = (
        df_selected.sort_values(
            ["ticker", "confidence_distance", "news_date_dt"],
            ascending=[True, False, False],
        )
        .groupby("ticker", as_index=False)
        .head(1)
        .copy()
    )

    # ============================================================
    # Portfolio integration
    # ============================================================
    prediction_probs = {}
    prediction_caps = {}
    prediction_used = False
    optimized_weights = {}

    if active_state:
                    prediction_probs = active_state.get("prediction_probs") or {}
                    prediction_caps = active_state.get("prediction_adjusted_caps") or {}
                    prediction_used = active_state.get("prediction_model_used")
                    optimized_weights = active_state.get("optimized_weights") or {}
                    constraint_debug = active_state.get("constraint_debug")

                    if optimized_weights:
                        st.markdown("---")
                        st.markdown("## Portfolio Integration")

                        st.caption(
                            "This section connects the trained news prediction model "
                            "with the currently optimized portfolio."
                        )

                        exposure_rows = []

                        for _, row in latest.iterrows():
                            ticker = str(row.get("ticker")).upper()

                            portfolio_weight = float(
                                optimized_weights.get(ticker, 0.0)
                            )

                            exposure_rows.append({
                                "Ticker": ticker,
                                "Portfolio weight": portfolio_weight,
                                "Predicted direction": row.get("Predicted direction"),
                                "Positive probability": row.get("predicted_positive_probability"),
                                "Signal strength": row.get("Signal strength"),
                            })

                        exposure_df = pd.DataFrame(exposure_rows)

                        exposure_df = exposure_df.sort_values(
                            "Portfolio weight",
                            ascending=False,
                        )

                        st.dataframe(
                            exposure_df.style.format({
                                "Portfolio weight": "{:.2%}",
                                "Positive probability": "{:.1%}",
                            }),
                            use_container_width=True,
                        )
    if prediction_used and prediction_probs:

        st.markdown("### Prediction-Constrained Optimization")

        pred_rows = []

        for ticker in selected_tickers:

            pred_rows.append({
                "Ticker": ticker,
                "Prediction probability":
                    prediction_probs.get(ticker),

                "Adjusted max allocation":
                    prediction_caps.get(ticker),

                "Final portfolio weight":
                    optimized_weights.get(ticker, 0.0),
                "Cap delta":
                    prediction_caps.get(ticker, 0.0)
                    - float(active_state.get("w_max", 0.30)),
            })

        pred_df = pd.DataFrame(pred_rows)

        st.dataframe(
            pred_df.style.format({
                "Prediction probability": "{:.1%}",
                "Adjusted max allocation": "{:.1%}",
                "Final portfolio weight": "{:.1%}",
            }),
            use_container_width=True,
    )
        # ============================================================
        # Compare baseline vs prediction-constrained portfolio
        # ============================================================

        baseline_weights = active_state.get("baseline_candidate_weights") or {}

        if baseline_weights:

            st.markdown("### Portfolio Weight Shift")

            shift_rows = []

            for ticker in selected_tickers:

                baseline_w = float(baseline_weights.get(ticker, 0.0))
                final_w = float(optimized_weights.get(ticker, 0.0))

                shift_rows.append({
                    "Ticker": ticker,
                    "Baseline weight": baseline_w,
                    "Prediction-constrained weight": final_w,
                    "Weight change": final_w - baseline_w,
                })

            shift_df = pd.DataFrame(shift_rows)

            st.dataframe(
                shift_df.style.format({
                    "Baseline weight": "{:.1%}",
                    "Prediction-constrained weight": "{:.1%}",
                    "Weight change": "{:+.1%}",
                }),
                use_container_width=True,
            )
        st.write("DEBUG constraint_debug:", constraint_debug)
        st.write("DEBUG active_state keys:", list(active_state.keys()) if active_state else None)
        if constraint_debug:
            st.markdown("### Constraint Debug Information")

            debug_rows = []

            for ticker, info in constraint_debug.items():
                constraint_type = info.get("constraint_type", "None")
                
                # Bullish → min floor göster, Bearish → max cap göster
                if constraint_type == "Bullish Min Floor":
                    constraint_value_label = "Min weight (floor)"
                    constraint_value = info.get("adjusted_min_weight") or info.get("adjusted_cap")
                elif constraint_type == "Bearish Max Cap":
                    constraint_value_label = "Max weight (cap)"
                    constraint_value = info.get("adjusted_max_weight") or info.get("adjusted_cap")
                else:
                    constraint_value_label = "No constraint"
                    constraint_value = None

                debug_rows.append({
                    "Ticker": ticker,
                    "Constraint type": constraint_type,
                    "Baseline max weight": info.get("baseline_cap"),
                    "Prediction probability": info.get("prediction_probability"),
                    constraint_value_label: constraint_value,
                    "Final weight": info.get("final_weight"),
                    "Bullish": info.get("is_bullish"),
                    "Bearish": info.get("is_bearish"),
                    "Binding": info.get("constraint_binding"),
                })

            debug_df = pd.DataFrame(debug_rows)

            # Format: sadece sayısal kolonları format et
            format_dict = {
                "Baseline max weight": "{:.1%}",
                "Prediction probability": "{:.1%}",
                "Final weight": "{:.1%}",
            }
            
            # Dinamik kolon adı için
            if "Min weight (floor)" in debug_df.columns:
                format_dict["Min weight (floor)"] = "{:.1%}"
            if "Max weight (cap)" in debug_df.columns:
                format_dict["Max weight (cap)"] = "{:.1%}"

            st.dataframe(
                debug_df.style.format(format_dict, na_rep="–"),
                use_container_width=True,
            )
                    
    latest_cols = [
        "ticker",
        "news_date",
        "predicted_positive_probability",
        "Signal strength",
        "Predicted direction",
        "Actual direction",
        "future_return",
        "Correct?",
    ]

    if "headline" in latest.columns:
        latest_cols.insert(2, "headline")

    latest_display = latest[latest_cols]
    latest_display = latest_display.rename(
        columns={
            "ticker": "Ticker",
            "news_date": "News date",
            "headline": "Headline",
            "predicted_positive_probability": "Positive probability",
            "future_return": "Realized future return",
        }
    )

    st.markdown("**Saved News Return Prediction Model**")
    st.caption(
        "This uses the selected best trained news prediction model saved as `best_news_prediction_model.joblib`. "
        "It predicts whether the 7-day future return after a news item is positive."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", _fmt_pct_from_decimal(_safe_float(metrics.get("accuracy"))))
    with c2:
        st.metric("Balanced accuracy", _fmt_pct_from_decimal(_safe_float(metrics.get("balanced_accuracy"))))
    with c3:
        st.metric("ROC-AUC", _fmt_num(_safe_float(metrics.get("roc_auc"))))
    with c4:
        baseline = _safe_float(metrics.get("majority_baseline_accuracy"))
        st.metric("Baseline accuracy", _fmt_pct_from_decimal(baseline))
    with c5:
        lift = _safe_float(metrics.get("model_minus_baseline_balanced_accuracy"))
        st.metric("Balanced lift vs balanced baseline", _fmt_signed_pct_from_decimal(lift, 1))

    reliability_warning = metrics.get("reliability_warning")
    if reliability_warning:
        st.warning(reliability_warning)
    else:
        st.success(
            "The model shows out-of-sample predictive signal above the majority baseline "
            "on the saved time-based test split."
        )

    with st.expander("Representative article-level prediction examples", expanded=False):
        st.caption(
            "These are the highest-confidence out-of-sample examples for the selected tickers. "
            "The table illustrates how the model's predicted direction compared with the realized future return."
        )

        st.dataframe(
            latest_display.style.format(
                {
                    "Positive probability": "{:.1%}",
                    "Realized future return": "{:.2%}",
                    "Signal strength": "{}",
                },
                na_rep="–",
            ),
            use_container_width=True,
            height=280,
        )

    st.markdown("---")

    c_left, c_right = st.columns(2)

    with c_left:
        _render_confusion_matrix_heatmap(metrics)

    with c_right:
        _render_probability_distribution(df_all)

    _render_calibration_table(df_all)

    with st.expander("Show full saved prediction results"):
            keep_cols = [
                "ticker",
                "news_date",
                "predicted_positive_probability",
                "Predicted direction",
                "Actual direction",
                "future_return",
                "Correct?",
            ]

            if "headline" in df_selected.columns:
                keep_cols.insert(2, "headline")

            show_cols = [c for c in keep_cols if c in df_selected.columns]

            st.dataframe(
                df_selected[show_cols].style.format(
                    {
                        "predicted_positive_probability": "{:.1%}",
                        "future_return": "{:.2%}",
                    },
                    na_rep="–",
                ),
                use_container_width=True,
                height=400,
            )

    items = explanation.get("items") or []
    if items:
        exp_df = pd.DataFrame(items[:15]).copy()

        value_col = "coefficient" if "coefficient" in exp_df.columns else "importance"
        if value_col in exp_df.columns:
            exp_df[value_col] = pd.to_numeric(exp_df[value_col], errors="coerce")
            exp_df["Interpretation"] = exp_df.apply(
                lambda r: _coefficient_interpretation(
                    r.get("feature"),
                    float(r.get(value_col) or 0.0),
                )
                if value_col == "coefficient"
                else "Higher importance means this feature contributed more to the model's decisions.",
                axis=1,
            )

        with st.expander("Top model explanation features", expanded=True):
            st.dataframe(exp_df, use_container_width=True)

    with st.expander("Saved model metrics JSON"):
        st.json(metrics)
        
# not used function 
def _active_portfolio_label(is_refined: bool) -> str:
    return "Refined Portfolio" if is_refined else "Base Portfolio"
def _render_news_adjustment_evaluation(evaluation: Optional[Dict[str, Any]]):
    if not isinstance(evaluation, dict) or not evaluation:
        st.info("No news adjustment evaluation available.")
        return

    st.markdown("**Portfolio-level News/FinBERT effect**")
    st.caption(evaluation.get("thesis_framing", ""))

    base = evaluation.get("base") or {}
    news = evaluation.get("news_adjusted") or {}
    deltas = evaluation.get("deltas") or {}
    effects = evaluation.get("effects") or {}
    weight_changes = evaluation.get("weight_changes") or {}

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Base**")
        st.write(f"Return: {_fmt_pct_from_decimal(_safe_float(base.get('return')))}")
        st.write(f"Volatility: {_fmt_pct_from_decimal(_safe_float(base.get('vol')))}")
        st.write(f"Sharpe: {_fmt_num(_safe_float(base.get('sharpe')))}")
        st.write(f"Max weight: {_fmt_pct_from_decimal(_safe_float(base.get('max_weight')))}")
        st.write(f"Effective N: {_fmt_num(_safe_float(base.get('effective_n')))}")

    with c2:
        st.markdown("**News-adjusted**")
        st.write(f"Return: {_fmt_pct_from_decimal(_safe_float(news.get('return')))}")
        st.write(f"Volatility: {_fmt_pct_from_decimal(_safe_float(news.get('vol')))}")
        st.write(f"Sharpe: {_fmt_num(_safe_float(news.get('sharpe')))}")
        st.write(f"Max weight: {_fmt_pct_from_decimal(_safe_float(news.get('max_weight')))}")
        st.write(f"Effective N: {_fmt_num(_safe_float(news.get('effective_n')))}")

    with c3:
        st.markdown("**Delta**")
        st.write(f"Δ Return: {_fmt_pct_from_decimal(_safe_float(deltas.get('return')))}")
        st.write(f"Δ Volatility: {_fmt_pct_from_decimal(_safe_float(deltas.get('vol')))}")
        st.write(f"Δ Sharpe: {_fmt_num(_safe_float(deltas.get('sharpe')))}")
        st.write(f"Δ Max weight: {_fmt_pct_from_decimal(_safe_float(deltas.get('max_weight')))}")
        st.write(f"Δ Effective N: {_fmt_num(_safe_float(deltas.get('effective_n')))}")
        st.write(f"Turnover: {_fmt_pct_from_decimal(_safe_float(deltas.get('turnover')))}")

    st.markdown("**Interpretation labels**")
    st.write(f"- Risk effect: `{effects.get('risk_effect', 'unknown')}`")
    st.write(f"- Efficiency effect: `{effects.get('efficiency_effect', 'unknown')}`")
    st.write(f"- Concentration effect: `{effects.get('concentration_effect', 'unknown')}`")

    if isinstance(weight_changes, dict) and weight_changes:
        rows = []
        for ticker, values in weight_changes.items():
            if not isinstance(values, dict):
                continue
            rows.append(
                {
                    "Ticker": ticker,
                    "Base weight": values.get("base_weight"),
                    "News-adjusted weight": values.get("news_weight"),
                    "Δ weight": values.get("delta_weight"),
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            for col in ["Base weight", "News-adjusted weight", "Δ weight"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("Δ weight", key=lambda s: s.abs(), ascending=False)

            with st.expander("Weight changes per ticker"):
                st.dataframe(
                    df.style.format(
                        {
                            "Base weight": "{:.2%}",
                            "News-adjusted weight": "{:.2%}",
                            "Δ weight": "{:+.2%}",
                        }
                    ),
                    use_container_width=True,
                )

def _render_prediction_evaluation(
    evaluation: Optional[Dict[str, Any]],
    *,
    title: str = "News signal predictive evaluation",
    caption: Optional[str] = None,
):
    if not isinstance(evaluation, dict) or not evaluation:
        st.info("No prediction evaluation available.")
        return

    if not evaluation.get("ok"):
        st.warning(evaluation.get("reason", "Prediction evaluation could not be computed."))
        return

    st.markdown(f"**{title}**")

    if caption:
        st.caption(caption)
    else:
        st.caption(
            "This checks whether FinBERT-based news directions matched subsequent realized stock returns."
        )
    from_date = evaluation.get("from_date")
    to_date = evaluation.get("to_date")
    raw_news_count = evaluation.get("raw_news_count")
    article_signal_count = evaluation.get("article_signal_count")
    tickers_used = evaluation.get("tickers") or []

    st.caption(
        f"Window: {from_date or '–'} → {to_date or '–'} | "
        f"Tickers: {', '.join(tickers_used) if tickers_used else '–'} | "
        f"Raw news: {raw_news_count or 0} | "
        f"FinBERT signals: {article_signal_count or 0}"
    )

    summary = evaluation.get("summary") or {}
    rows = evaluation.get("rows") or []

    if summary:
        summary_rows = []
        for horizon, vals in summary.items():
            if not isinstance(vals, dict):
                continue
            summary_rows.append(
                {
                    "Horizon": f"{horizon} trading day(s)",
                    "N": vals.get("n"),
                    "Valid N": vals.get("valid_n"),
                    "Direction accuracy": vals.get("direction_accuracy"),
                    "Avg future return": vals.get("avg_future_return"),
                    "Sentiment-return corr": vals.get("sentiment_future_return_corr"),
                }
            )

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            st.dataframe(
                df_sum.style.format(
                    {
                        "Direction accuracy": "{:.1%}",
                        "Avg future return": "{:.2%}",
                        "Sentiment-return corr": "{:.3f}",
                    },
                    na_rep="–",
                ),
                use_container_width=True,
            )

    if rows:
        df = pd.DataFrame(rows)
        keep_cols = [
            "ticker",
            "news_date",
            "horizon_days",
            "predicted_direction",
            "future_return",
            "actual_direction",
            "correct",
            "sentiment",
            "confidence",
            "start_price_date",
            "future_price_date",
            "start_close",
            "future_close",
            "headline",
            "source",
            
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        with st.expander("Article-level prediction checks"):
            st.dataframe(
                df.style.format(
                    {
                        "future_return": "{:.2%}",
                        "sentiment": "{:.3f}",
                        "confidence": "{:.3f}",
                    },
                    na_rep="–",
                ),
                use_container_width=True,
                height=350,
            )

def _render_news_impact_heatmap(
    df: pd.DataFrame,
    chart_key: str = "news_impact_heatmap",
):
    if df is None or df.empty:
        st.info("No news impact data available for heatmap.")
        return

    heatmap_cols = [
        "FinBERT sentiment",
        "Confidence",
        "Expected return adjustment",
        "Risk adjustment",
        "Δ μ",
        "Δ variance",
    ]

    available_cols = [c for c in heatmap_cols if c in df.columns]

    if not available_cols or "Ticker" not in df.columns:
        st.info("News impact heatmap cannot be rendered because required columns are missing.")
        return

    heat_df = df[["Ticker"] + available_cols].copy()

    for col in available_cols:
        heat_df[col] = pd.to_numeric(heat_df[col], errors="coerce")

    heat_df = heat_df.set_index("Ticker")

    if heat_df.dropna(how="all").empty:
        st.info("No numeric news impact values available for heatmap.")
        return


    max_abs = np.nanmax(np.abs(heat_df.values))

    fig = px.imshow(
        heat_df,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        range_color=[-max_abs, max_abs],
        origin="lower",
    )

    fig.update_xaxes(side="top")

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    st.caption(
        "This heatmap shows how FinBERT/news signals affected each asset before optimization. "
        "Positive expected return adjustments increase the return input, while positive risk adjustments or Δ variance increase the risk input."
    )
    st.caption(
    "Green indicates larger positive magnitudes, while values near zero appear neutral."
    )
def _render_prob_news_trace(trace: Optional[Dict[str, Any]]):
    if not isinstance(trace, dict) or not trace:
        st.info("No mathematical news trace available.")
        return

    ticker_signals = trace.get("ticker_signals") or {}
    prediction_signals = trace.get("prediction_signals") or {}
    mu_before = trace.get("mu_before") or {}
    mu_after = trace.get("mu_after") or {}
    mu_delta = trace.get("mu_delta") or {}
    var_before = trace.get("variance_before") or {}
    var_after = trace.get("variance_after") or {}
    var_delta = trace.get("variance_delta") or {}

    rows = []

    for ticker, sig in ticker_signals.items():
        if not isinstance(sig, dict):
            continue

        pred = prediction_signals.get(ticker) or {}

        rows.append(
            {
                "Ticker": ticker,
                "FinBERT sentiment": sig.get("sentiment_score"),
                "Confidence": sig.get("confidence_score"),
                "Articles": sig.get("raw_article_count"),
                "Weighted articles": sig.get("weighted_article_count"),
                "Sentiment variance": sig.get("sentiment_variance"),
                "Predicted direction": pred.get("predicted_direction"),
                "Prediction confidence": pred.get("prediction_confidence"),
                "Expected return adjustment": pred.get("expected_return_adjustment"),
                "Risk adjustment": pred.get("risk_adjustment"),
                "μ before": mu_before.get(ticker),
                "μ after": mu_after.get(ticker),
                "Δ μ": mu_delta.get(ticker),
                "Variance before": var_before.get(ticker),
                "Variance after": var_after.get(ticker),
                "Δ variance": var_delta.get(ticker),
            }
        )
        

    if rows:
        df = pd.DataFrame(rows)



        numeric_cols = [
            "FinBERT sentiment",
            "Confidence",
            "Weighted articles",
            "Sentiment variance",
            "Prediction confidence",
            "Expected return adjustment",
            "Risk adjustment",
            "μ before",
            "μ after",
            "Δ μ",
            "Variance before",
            "Variance after",
            "Δ variance",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.markdown("**Predictive news signals and their effect on the optimization inputs**")
        st.markdown(
            """
            **Direction thresholds**
            - Positive: sentiment > 0.10
            - Negative: sentiment < -0.10
            - Neutral: between -0.10 and 0.10
            """
        )
        st.dataframe(df, use_container_width=True)
        st.markdown("**News Impact Heatmap**")
        _render_news_impact_heatmap(
            df,
            chart_key="news_impact_heatmap_main",
        )


    params = trace.get("parameters") or {}
    if params:
        with st.expander("Model parameters"):
            st.json(params)

    article_signals = trace.get("article_signals") or []
    if isinstance(article_signals, list) and article_signals:
        with st.expander("Article-level FinBERT signals"):
            article_rows = []
            for a in article_signals:
                if not isinstance(a, dict):
                    continue
                probs = a.get("probs") or {}
                article_rows.append(
                    {
                        "Ticker": a.get("ticker"),
                        "Headline": a.get("headline"),
                        "Source": a.get("source"),
                        "Positive": probs.get("positive"),
                        "Negative": probs.get("negative"),
                        "Neutral": probs.get("neutral"),
                        "Article sentiment": a.get("article_sentiment"),
                        "Confidence": a.get("article_confidence"),
                        "Recency weight": a.get("recency_weight"),
                        "Combined weight": a.get("combined_weight"),
                    }
                )

            if article_rows:
                st.dataframe(pd.DataFrame(article_rows), use_container_width=True, height=320)
def _portfolio_summary_from_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    ✅ For Evaluation section.
    Prefer optimized_* (final) to avoid mismatch.
    """
    if not state:
        return None

    chosen = _get_chosen_candidate(state)
    opt_m = state.get("optimized_metrics") or {}
    opt_w = state.get("optimized_weights") or {}

    if opt_w:
        w = pd.Series(opt_w, dtype=float)
        w = w[w.abs() > 1e-6]
        if w.empty:
            return None

        w = w.sort_values(ascending=False)
        eff_n = float(1.0 / np.sum(np.square(w.values)))
        max_w = float(w.max())

        sharpe = _safe_float(opt_m.get("sharpe"))
        ret = _safe_float(opt_m.get("return"))
        vol = _safe_float(opt_m.get("vol"))

        active_assets = opt_m.get("active_assets", None)
        try:
            active_assets = int(active_assets) if active_assets is not None else int(len(w))
        except Exception:
            active_assets = int(len(w))

        return {
            "candidate": chosen,
            "return": ret if ret is not None else float(np.nan),
            "vol": vol if vol is not None else float(np.nan),
            "sharpe": sharpe,
            "active_assets": active_assets,
            "max_weight": max_w,
            "effective_n": eff_n,
        }

    # fallback older-style
    opt_res = state.get("optimization_result") or {}
    if chosen not in opt_res:
        return None

    port = opt_res[chosen]
    w = pd.Series(port.get("weights", {}), dtype=float)
    w = w[w.abs() > 1e-6]
    if w.empty:
        return None

    w = w.sort_values(ascending=False)
    eff_n = float(1.0 / np.sum(np.square(w.values)))
    max_w = float(w.max())

    sharpe = _safe_float(port.get("sharpe"))
    ret = _safe_float(port.get("return"))
    vol = _safe_float(port.get("vol"))

    return {
        "candidate": chosen,
        "return": ret if ret is not None else float(np.nan),
        "vol": vol if vol is not None else float(np.nan),
        "sharpe": sharpe,
        "active_assets": int(len(w)),
        "max_weight": max_w,
        "effective_n": eff_n,
    }
def _render_base_vs_refined_metric_chart(
    base_sum: Optional[Dict[str, Any]],
    ref_sum: Optional[Dict[str, Any]],
    chart_key: str = "base_vs_refined_metric_chart",
):
    if base_sum is None:
        return

    if ref_sum is None:
        st.info("Run refinement to see the Before / After metric comparison chart.")
        return

    rows = [
        {
            "Metric": "Return",
            "Base": _safe_float(base_sum.get("return")),
            "Refined": _safe_float(ref_sum.get("return")),
            "Display unit": "%",
        },
        {
            "Metric": "Volatility",
            "Base": _safe_float(base_sum.get("vol")),
            "Refined": _safe_float(ref_sum.get("vol")),
            "Display unit": "%",
        },
        {
            "Metric": "Sharpe",
            "Base": _safe_float(base_sum.get("sharpe")),
            "Refined": _safe_float(ref_sum.get("sharpe")),
            "Display unit": "number",
        },
        {
            "Metric": "Effective N",
            "Base": _safe_float(base_sum.get("effective_n")),
            "Refined": _safe_float(ref_sum.get("effective_n")),
            "Display unit": "number",
        },
        {
            "Metric": "Max Weight",
            "Base": _safe_float(base_sum.get("max_weight")),
            "Refined": _safe_float(ref_sum.get("max_weight")),
            "Display unit": "%",
        },
    ]

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Base", "Refined"], how="all")

    if df.empty:
        st.info("No comparable metrics available.")
        return

    # Convert only percent metrics to actual percentage scale for display
    pct_mask = df["Display unit"] == "%"
    df.loc[pct_mask, ["Base", "Refined"]] = df.loc[pct_mask, ["Base", "Refined"]] * 100

    df_long = df.melt(
        id_vars=["Metric", "Display unit"],
        value_vars=["Base", "Refined"],
        var_name="Portfolio",
        value_name="Value",
    )

    df_long["Label"] = df_long.apply(
        lambda r: f"{r['Value']:.1f}%" if r["Display unit"] == "%" else f"{r['Value']:.2f}",
        axis=1,
    )

    fig = px.bar(
        df_long,
        x="Metric",
        y="Value",
        color="Portfolio",
        barmode="group",
        text="Label",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Value",
        xaxis_title="",
        legend_title="",
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    st.caption(
        "Before / After comparison of the base portfolio and the refined portfolio. "
        "Return, volatility, and max weight are shown as percentages; Sharpe and Effective N are shown as numeric values."
    )
def _render_double_frontier_chart(
    base_state: Optional[Dict[str, Any]],
    refined_state: Optional[Dict[str, Any]],
    chart_key: str = "double_frontier_chart",
):
    if not base_state or not refined_state:
        st.info("Run base and news-integrated refinement to compare frontiers.")
        return

    base_opt = base_state.get("optimization_result") or {}
    ref_opt = refined_state.get("optimization_result") or {}

    base_frontier = base_opt.get("frontier")
    ref_frontier = ref_opt.get("frontier")

    if not base_frontier or not ref_frontier:
        st.info("Frontier data is missing for base or refined portfolio.")
        return

    base_df = pd.DataFrame(base_frontier)
    ref_df = pd.DataFrame(ref_frontier)

    def _frontier_y_col(df):
        if "realized_return" in df.columns:
            return "realized_return"
        if "return" in df.columns:
            return "return"
        return df.columns[-1]

    base_y = _frontier_y_col(base_df)
    ref_y = _frontier_y_col(ref_df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=base_df["vol"],
            y=base_df[base_y],
            mode="lines+markers",
            name="Base frontier",
            line=dict(dash="dash"),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ref_df["vol"],
            y=ref_df[ref_y],
            mode="lines+markers",
            name="News-adjusted frontier",
            marker=dict(size=5),
        )
    )

    base_m = base_state.get("optimized_metrics") or {}
    ref_m = refined_state.get("optimized_metrics") or {}

    base_vol = _safe_float(base_m.get("vol"))
    base_ret = _safe_float(base_m.get("return"))
    ref_vol = _safe_float(ref_m.get("vol"))
    ref_ret = _safe_float(ref_m.get("return"))

    if base_vol is not None and base_ret is not None:
        fig.add_trace(
            go.Scatter(
                x=[base_vol],
                y=[base_ret],
                mode="markers+text",
                name="Base portfolio",
                text=["Base portfolio"],
                textposition="bottom right",
                marker=dict(size=12),
            )
        )

    if ref_vol is not None and ref_ret is not None:
        fig.add_trace(
            go.Scatter(
                x=[ref_vol],
                y=[ref_ret],
                mode="markers+text",
                name="News-adjusted portfolio",
                text=["News-adjusted portfolio"],
                textposition="top right",
                marker=dict(size=12),
            )
        )

    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Risk (Volatility, σ)",
        yaxis_title="Expected Return (µ)",
        legend_title="",
        height=520,
    )

    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    st.caption(
        "This chart compares the original base efficient frontier with the news-adjusted frontier. "
        "If the curves differ, it means the news module changed the optimization inputs before portfolio selection."
    )
def _weights_from_state(state: Optional[Dict[str, Any]]) -> Optional[pd.Series]:
    if not state:
        return None

    opt_w = state.get("optimized_weights") or {}
    if opt_w:
        w = pd.Series(opt_w, dtype=float)
        return w[w.abs() > 1e-6].sort_values(ascending=False)

    chosen = _get_chosen_candidate(state)
    opt_res = state.get("optimization_result") or {}

    if chosen in opt_res:
        w = pd.Series((opt_res[chosen] or {}).get("weights", {}), dtype=float)
        return w[w.abs() > 1e-6].sort_values(ascending=False)

    return None


def _render_weight_change_chart(
    base_state: Optional[Dict[str, Any]],
    refined_state: Optional[Dict[str, Any]],
    chart_key: str = "weight_change_chart",
):
    base_w = _weights_from_state(base_state)
    ref_w = _weights_from_state(refined_state)

    if base_w is None or ref_w is None:
        st.info("Run refinement to see weight changes.")
        return

    all_tickers = sorted(set(base_w.index).union(set(ref_w.index)))

    df = pd.DataFrame(
        {
            "Ticker": all_tickers,
            "Base weight": [float(base_w.get(t, 0.0)) for t in all_tickers],
            "Refined weight": [float(ref_w.get(t, 0.0)) for t in all_tickers],
        }
    )

    df["Δ weight"] = df["Refined weight"] - df["Base weight"]
    df = df.sort_values("Δ weight", key=lambda s: s.abs(), ascending=True)
    df["Direction"] = np.select(
    [
        df["Δ weight"] > 1e-6,
        df["Δ weight"] < -1e-6,
    ],
    [
        "Increase",
        "Decrease",
    ],
    default="No change",
    )

    if df["Δ weight"].abs().sum() <= 1e-9:
        st.info("No meaningful weight changes detected.")
        return

    fig = px.bar(
        df,
        x="Δ weight",
        y="Ticker",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Increase": "#4ade80",   # green
            "Decrease": "#f87171",   # red
            "No change": "#9ca3af",  # gray
        },
        text="Δ weight",
        hover_data={
            "Base weight": ":.2%",
            "Refined weight": ":.2%",
            "Δ weight": ":.2%",
            "Direction": False,
        },
    )

    fig.update_traces(
        texttemplate="%{text:+.2%}",
        textposition="outside",
        cliponaxis=False,
    )

    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Change in portfolio weight",
        yaxis_title="",
        height=max(320, 55 * len(df)),
        showlegend=True,
        legend_title="",
    )

    fig.update_xaxes(tickformat=".1%")

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    st.caption(
        "Positive values mean the refined portfolio increased exposure to that asset; "
        "negative values mean exposure was reduced."
    )
def _fmt_pct_from_decimal(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x*100:.1f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.2f}"

def _fmt_signed_pct_from_decimal(x: Optional[float], digits: int = 2) -> str:
    v = _safe_float(x)
    if v is None:
        return "–"
    return f"{v * 100:+.{digits}f}%"


def _fmt_small_num(x: Optional[float], digits: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return "–"
    return f"{v:.{digits}f}"
def _fmt_pct_from_pct_field(x_pct: Optional[float]) -> str:
    # for fields already in percent units (e.g., 17.7)
    if x_pct is None or not np.isfinite(x_pct):
        return "–"
    return f"{float(x_pct):.1f}%"


# ============================================================
# Pain point labels (must match backend constants)
# ============================================================
PP_TOO_RISKY = "It feels too risky"
PP_TOO_CONSERVATIVE = "It feels too conservative"
PP_TOO_CONCENTRATED = "It’s too concentrated in a few assets"
PP_DISLIKE_ASSETS = "I don’t like some of the assets"
PP_NOT_SURE = "I’m not sure — I just want something safer/smoother"


def _sanitize_pain_points(raw: list[str]) -> list[str]:
    if not raw:
        return []
    s = set(raw)
    if PP_NOT_SURE in s:
        return [PP_NOT_SURE]
    if (PP_TOO_RISKY in s) and (PP_TOO_CONSERVATIVE in s):
        s.remove(PP_TOO_CONSERVATIVE)
    return list(s)


def _rc_series_aligned_to(tickers_target: list[str], metrics: dict):
    """
    metrics: {'tickers': [...], 'rc_pct': [...]}
    returns: rc_pct aligned to tickers_target order (np.ndarray)
    """
    if not metrics:
        return None

    src_t = metrics.get("tickers")
    src_rc = metrics.get("rc_pct")
    if src_t is None or src_rc is None:
        return None

    src_t = list(map(str, src_t))
    src_rc = np.array(src_rc, dtype=float)

    if len(src_t) != len(src_rc):
        return None

    m = {t: float(v) for t, v in zip(src_t, src_rc)}
    return np.array([m.get(str(t), np.nan) for t in tickers_target], dtype=float)


def _extract_tickers_from_notes(extra_notes: str, universe: list[str], max_n: int = 10) -> list[str]:
    if not extra_notes:
        return []
    candidates = re.findall(r"\b[A-Z]{1,5}\b", extra_notes.upper())
    if not candidates:
        return []
    universe_set = set(map(str, universe))
    found = []
    for t in candidates:
        if t in universe_set and t not in found:
            found.append(t)
        if len(found) >= max_n:
            break
    return found

def _detect_mentioned_tickers(text: str, universe: list[str], max_n: int = 6) -> list[str]:
    if not text:
        return []
    text_u = text.upper()

    # en basit ve güvenli yöntem: universe içindeki ticker'ları text içinde ara
    found = []
    for t in universe:
        tt = str(t).upper().strip()
        if not tt:
            continue
        if tt in text_u and tt not in found:
            found.append(tt)
        if len(found) >= max_n:
            break
    return found

# ✅ NEW: News rendering helpers (supports placeholder + real LLM news snapshot output)
def _extract_news_snapshot_and_risk(state: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not state:
        return None, None

    # ✅ NEW: prefer explicit fields from backend
    snapshot_text = None
    # ✅ 0) UI-friendly raw snapshot (if backend provides it)
    if isinstance(state.get("news_snapshot_text_raw"), str) and state["news_snapshot_text_raw"].strip():
        snapshot_text = state["news_snapshot_text_raw"].strip()
    if isinstance(state.get("news_snapshot_text"), str) and state["news_snapshot_text"].strip():
        snapshot_text = state["news_snapshot_text"].strip()
    elif isinstance(state.get("news_snapshot"), str) and state["news_snapshot"].strip():  # future-proof
        snapshot_text = state["news_snapshot"].strip()

   
    risk_json = state.get("news_risk_json") if isinstance(state.get("news_risk_json"), dict) else None
    if risk_json is None:
        risk_json = state.get("news_signals") if isinstance(state.get("news_signals"), dict) else None

    # If risk_json has summary, use it as snapshot if snapshot_text missing
    if (not snapshot_text) and isinstance(risk_json, dict):
        if isinstance(risk_json.get("summary"), str) and risk_json["summary"].strip():
            snapshot_text = risk_json["summary"].strip()

    # Placeholder schema fallback: global risk_flags/vol_regime -> build small text if still none
    if (not snapshot_text) and isinstance(risk_json, dict):
        glob = risk_json.get("global")
        if isinstance(glob, dict):
            flags = glob.get("risk_flags")
            vr = glob.get("vol_regime")
            if isinstance(flags, list) and flags:
                snapshot_text = f"Detected {len(flags)} potential event-risk flag(s). Vol regime: {vr or 'normal'}."

    return snapshot_text, risk_json



def _news_section(state: Dict[str, Any]):
    st.markdown('<div class="section-title">📰 News Snapshot & Risk Check</div>', unsafe_allow_html=True)

    snapshot_text, risk_json = _extract_news_snapshot_and_risk(state)

    if (snapshot_text is None) and (not isinstance(risk_json, dict) or not risk_json):
        st.caption("No news snapshot available (yet).")
        return

    if snapshot_text:
        st.write(snapshot_text)

    if isinstance(risk_json, dict) and risk_json:
        glob = risk_json.get("global") if isinstance(risk_json.get("global"), dict) else {}
        by_ticker = risk_json.get("by_ticker") if isinstance(risk_json.get("by_ticker"), dict) else {}

        # global
        if isinstance(glob, dict) and glob:
            vol_regime = str(glob.get("vol_regime") or "normal")
            st.caption(f"Global regime: **{vol_regime}**")

            flags = glob.get("risk_flags")
            if isinstance(flags, list) and flags:
                with st.expander("Global risk flags"):
                    st.json(flags)

        # per ticker
        if isinstance(by_ticker, dict) and by_ticker:
            rows = []
            for t, v in by_ticker.items():
                if not isinstance(v, dict):
                    continue
                rows.append(
                    {
                        "Ticker": str(t),
                        "Risk flag": str(v.get("risk_flag") or "none"),
                        "Confidence": v.get("confidence"),
                    }
                )
            if rows:
                df = pd.DataFrame(rows)
                # sort: highest confidence first
                if "Confidence" in df.columns:
                    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
                    df = df.sort_values(by="Confidence", ascending=False, na_position="last")
                st.dataframe(df, use_container_width=True, height=220)

def _get_news_items_by_id(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build lookup map for BOTH:
      - evidence_id (preferred; e.g. APP_9d1b573b)
      - raw id (fallback; e.g. cea6dfaf1ff9)

    This fixes: "(not found in news_items_by_id)" when actions reference evidence_ids.
    """
    m = state.get("news_items_by_id") or state.get("items_by_id")
    if isinstance(m, dict) and m:
        out = {}
        for k, v in m.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out

    raw = (
        state.get("news_raw_items")
        or state.get("news_snapshot_raw_items")
        or state.get("news_items")
        or state.get("news_raw")
        or state.get("news")
        or []
    )

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue

            eid = it.get("evidence_id")
            rid = it.get("id")

            
            if eid is not None and str(eid).strip():
                out[str(eid)] = it


    return out



def _render_action_evidence(
    *,
    action: Dict[str, Any],
    news_items_by_id: Dict[str, Dict[str, Any]],
    universe_tickers: List[str],
):
    #  1) Prefer structured evidence list (your backend output)
    ev_list = action.get("evidence")
    eids: List[str] = []

    if isinstance(ev_list, list) and ev_list:
        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("evidence_id") or ev.get("id") or "").strip()
            if eid:
                eids.append(eid)

    #  2) Fallback legacy field if present
    if not eids:
        raw_eids = action.get("evidence_ids") or []
        if isinstance(raw_eids, list):
            eids = [str(x).strip() for x in raw_eids if str(x).strip()]

    if not eids:
        st.caption("No evidence attached.")
        return

    lines: List[str] = []
    for eid in eids[:6]:  # show a few
        item = news_items_by_id.get(eid)

        # If we can't find the full item, still show the id (debug)
        if not isinstance(item, dict):
            lines.append(f"- `{eid}` (not found in news_items_by_id)")
            continue

        ticker = str(item.get("ticker") or "UNK").strip()
        date = str(item.get("date") or "unknown").strip()
        source = str(item.get("source") or item.get("provider") or "unknown").strip()
        headline = str(item.get("headline") or "headline").strip()
        summary = str(item.get("summary") or "").strip()
        url = str(item.get("url") or "").strip()

        #  Cross-mention detection (headline + summary)
        mentioned = _detect_mentioned_tickers(f"{headline} {summary}", universe_tickers)
        # item’in kendi ticker’ını “mentions”tan çıkar
        also_mentions = [t for t in mentioned if t != ticker.upper()]

        if url:
            lines.append(f"- {ticker} ({date} | {source}) [{headline}]({url})")
        else:
            lines.append(f"- {ticker} ({date} | {source}) {headline} (no url)")

        if also_mentions:
            lines.append(f"  - _Mentions:_ {', '.join(also_mentions)}")
        else:
            lines.append("  - _Mentions: 0")




    st.markdown("**Evidence**")
    st.markdown("\n".join(lines))


#  NEW: Insight rendering helpers (supports narrative raw_text)
def _insight_section(state: Dict[str, Any]):
    insight = state.get("insight")
    ok = state.get("insight_ok")
    issues = state.get("insight_issues") or []
    parse_mode = state.get("insight_parse_mode")
    raw_text = state.get("insight_raw_text")

    st.markdown('<div class="section-title">✨ Insights (LLM)</div>', unsafe_allow_html=True)

    has_any = (insight is not None) or (isinstance(raw_text, str) and raw_text.strip())
    if not has_any:
        st.info("No insights generated yet. Run **Refine** with LLM enabled to produce insights.")
        return

    if ok is True:
        st.success(f"Insight generated ({parse_mode or 'unknown parse'}).")
    elif ok is False:
        st.warning("Insight generation had issues (showing best-effort output).")
    else:
        st.caption("Insight status unknown.")

    if issues:
        with st.expander("⚠️ Insight issues"):
            for it in issues:
                st.write(f"- {it}")

    #  Prefer narrative text if present
    if isinstance(raw_text, str) and raw_text.strip():
        st.markdown(raw_text)
        return

    # Otherwise render structured JSON insight
    headline = (insight or {}).get("headline")
    if isinstance(headline, str) and headline.strip():
        st.markdown(f"**{headline.strip()}**")
    else:
        st.markdown("**Portfolio insights**")

    story = (insight or {}).get("portfolio_story") or []
    if isinstance(story, list) and story:
        st.markdown("**What changed / what it means**")
        for s in story[:8]:
            if isinstance(s, str) and s.strip():
                st.write(f"- {s.strip()}")

    drivers = (insight or {}).get("risk_drivers") or []
    if isinstance(drivers, list) and drivers:
        st.markdown("**Main risk drivers**")
        for d in drivers[:8]:
            if isinstance(d, str) and d.strip():
                st.write(f"- {d.strip()}")

    bvr = (insight or {}).get("base_vs_refine") or {}
    metric_deltas = (bvr.get("metric_deltas") or {}) if isinstance(bvr, dict) else {}
    key_changes = (bvr.get("key_changes") or []) if isinstance(bvr, dict) else []

    if key_changes:
        st.markdown("**Key changes**")
        for k in key_changes[:8]:
            if isinstance(k, str) and k.strip():
                st.write(f"- {k.strip()}")

    if metric_deltas:
        st.markdown("**Metric deltas (Base → Refine)**")
        try:
            st.json(metric_deltas)
        except Exception:
            st.write(metric_deltas)

    news_overlay = (insight or {}).get("news_overlay") or []
    if isinstance(news_overlay, list) and news_overlay:
        st.markdown("**News overlay**")
        for n in news_overlay[:8]:
            if isinstance(n, str) and n.strip():
                st.write(f"- {n.strip()}")

    actions = (insight or {}).get("action_suggestions_optional") or []
    if isinstance(actions, list) and actions:
        st.markdown("**Optional actions**")
        for a in actions[:8]:
            if isinstance(a, str) and a.strip():
                st.write(f"- {a.strip()}")

def _render_why_this_portfolio_cards(state: Optional[Dict[str, Any]]):
    if not isinstance(state, dict) or not state:
        st.info("No portfolio explanation available yet.")
        return

    active_state = (
    st.session_state.get("refined_state")
    or st.session_state.get("base_state")
    or {}
    )

    is_refined = active_state is not st.session_state.get("base_state")

    insight_text = str(state.get("insight_raw_text") or "").strip()
    explanation_text = str(state.get("explanation") or "").strip()
    text = insight_text or explanation_text

    opt_m = state.get("optimized_metrics") or {}
    if state.get("prediction_model_used"):
        chosen = _get_chosen_candidate(
            st.session_state.get("base_state") or state
        )
    else:
        chosen = _get_chosen_candidate(state)

    ret = _fmt_pct_from_decimal(_safe_float(opt_m.get("return")))
    vol = _fmt_pct_from_decimal(_safe_float(opt_m.get("vol")))
    sharpe = _fmt_num(_safe_float(opt_m.get("sharpe")))

    objective_label = "Max Sharpe" if chosen == "maxsharpe" else "Min Variance"
    news_used = bool(state.get("prob_news_trace") or state.get("news_adjustment_evaluation"))
    if state.get("prediction_model_used"):
        news_label = "Prediction-constrained optimization"
    elif news_used:
        news_label = "News-adjusted (FinBERT/probabilistic)"
    else:
        news_label = "No mathematical news adjustment"

    st.markdown("### 🧠 Why this portfolio?")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Main selection logic</div>
                <div class="metric-value" style="font-size:1.1rem;">{objective_label}</div>
                <div class="metric-sub">Chosen candidate: {chosen}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk / return profile</div>
                <div class="metric-value" style="font-size:1.1rem;">Sharpe {sharpe}</div>
                <div class="metric-sub">Return {ret} | Volatility {vol}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">News effect</div>
                <div class="metric-value" style="font-size:1.1rem;">{news_label}</div>
                <div class="metric-sub">Based on available graph state</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sentences = []
    if text:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    def pick_sentence(keywords: list[str], fallback: str = "") -> str:
        for s in sentences:
            low = s.lower()
            if any(k in low for k in keywords):
                return s
        return fallback

    if chosen == "maxsharpe":
        default_main = (
            "The optimizer selected the portfolio with the strongest risk-adjusted performance, "
            "meaning it tries to maximize return per unit of risk."
        )
    else:
        default_main = (
            "The optimizer selected the lower-risk portfolio, meaning it prioritizes volatility reduction "
            "over maximum expected return."
        )

    main_reason = pick_sentence(["chosen", "selected", "portfolio"], default_main)

    risk_driver = pick_sentence(
        ["risk", "volatility"],
        f"The selected portfolio has an annualized volatility of {vol}, which summarizes its expected risk level."
    )

    return_driver = pick_sentence(
        ["return", "sharpe"],
        f"The portfolio has an expected annualized return of {ret} and a Sharpe ratio of {sharpe}."
    )

    if is_refined:
        change_title = "What changed vs base"
        change_driver = pick_sentence(
            ["base", "refined", "changed", "increase", "decrease"],
            "The refined portfolio updates the base allocation according to the selected preference or news-adjusted optimization."
        )
    else:
        change_title = "Baseline comparison"
        max_w = _fmt_pct_from_decimal(_safe_float(_portfolio_summary_from_state(state).get("max_weight")))
        eff_n = _fmt_num(_safe_float(_portfolio_summary_from_state(state).get("effective_n")))
        change_driver = (
            f"This is the first optimized base portfolio. It can be compared against an equal-weight baseline "
            f"or later against a refined portfolio. Current max weight is {max_w}, effective holdings are {eff_n}."
        )

    card_rows = [
        ("Main reason", main_reason),
        ("Risk driver", risk_driver),
        ("Return / efficiency driver", return_driver),
        (change_title, change_driver),
    ]

    seen = set()
    for title, content in card_rows:
        content = str(content or "").strip()
        normalized = content.lower()

        if not content or normalized in seen:
            continue

        seen.add(normalized)

        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:0.6rem;">
                <div class="metric-label">{title}</div>
                <div style="color:#f7f9ff; font-size:0.95rem; line-height:1.45;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def _render_mode_c_diversification_chart(
    base_sum: Optional[Dict[str, Any]],
    ref_sum: Optional[Dict[str, Any]],
    chart_key: str = "mode_c_diversification_chart",
):
    if base_sum is None or ref_sum is None:
        return

    rows = [
        {
            "Metric": "Max Weight (%)",
            "Base": _safe_float(base_sum.get("max_weight")),
            "Mode C Refined": _safe_float(ref_sum.get("max_weight")),
            "lower_is_better": True,
        },
        {
            "Metric": "Effective N",
            "Base": _safe_float(base_sum.get("effective_n")),
            "Mode C Refined": _safe_float(ref_sum.get("effective_n")),
            "lower_is_better": False,
        },
        {
            "Metric": "Active Assets",
            "Base": _safe_float(base_sum.get("active_assets")),
            "Mode C Refined": _safe_float(ref_sum.get("active_assets")),
            "lower_is_better": False,
        },
    ]

    df = pd.DataFrame(rows)
    pct_mask = df["Metric"] == "Max Weight (%)"
    df.loc[pct_mask, ["Base", "Mode C Refined"]] = (
        df.loc[pct_mask, ["Base", "Mode C Refined"]] * 100
    )

    df_long = df.melt(
        id_vars=["Metric", "lower_is_better"],
        value_vars=["Base", "Mode C Refined"],
        var_name="Portfolio",
        value_name="Value",
    )

    df_long["Label"] = df_long.apply(
        lambda r: f"{r['Value']:.1f}%" if r["Metric"] == "Max Weight (%)" else f"{r['Value']:.1f}",
        axis=1,
    )

    fig = px.bar(
        df_long,
        x="Metric",
        y="Value",
        color="Portfolio",
        barmode="group",
        text="Label",
        color_discrete_map={
            "Base": "#6366f1",
            "Mode C Refined": "#4ade80",
        },
        title="Mode C: Diversification Impact",
    )

    fig.update_traces(textposition="outside", cliponaxis=False)

    fig.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
        legend_title="",
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    st.caption(
        "Mode C LLM news actions reduced concentration: max weight dropped from "
        f"{_fmt_pct_from_decimal(_safe_float(base_sum.get('max_weight')))} to "
        f"{_fmt_pct_from_decimal(_safe_float(ref_sum.get('max_weight')))}, "
        f"effective holdings increased from {base_sum.get('effective_n', 0):.1f} to "
        f"{ref_sum.get('effective_n', 0):.1f}, "
        f"and active assets from {base_sum.get('active_assets', 0)} to "
        f"{ref_sum.get('active_assets', 0)}. "
        "Return and Sharpe declined marginally (−0.5%, −0.02) as a trade-off for better diversification."
    )
def _render_portfolio_story_timeline(
    base_state: Optional[Dict[str, Any]],
    refined_state: Optional[Dict[str, Any]],
):
    if not base_state:
        st.info("Run the base portfolio first to build the portfolio story.")
        return

    active_state = refined_state or base_state
    story_steps = []

    base_obj = _get_chosen_candidate(base_state)
    base_m = base_state.get("optimized_metrics") or {}

    story_steps.append(
        {
            "title": "Base portfolio generated",
            "subtitle": f"Initial optimized portfolio selected with `{base_obj}` objective.",
            "detail": (
                f"Return {_fmt_pct_from_decimal(_safe_float(base_m.get('return')))} | "
                f"Volatility {_fmt_pct_from_decimal(_safe_float(base_m.get('vol')))} | "
                f"Sharpe {_fmt_num(_safe_float(base_m.get('sharpe')))}"
            ),
            "status": "done",
        }
    )

    if refined_state:
        refined_answers = refined_state.get("clarification_answers") or {}
        pain_points = refined_answers.get("pain_points") or []
        excluded_assets = refined_answers.get("excluded_assets") or []
        extra_notes = str(refined_answers.get("extra_notes") or "").strip()

        preference_parts = []

        if pain_points:
            preference_parts.append("Pain points: " + ", ".join(map(str, pain_points)))
        if excluded_assets:
            preference_parts.append("Excluded assets: " + ", ".join(map(str, excluded_assets)))
        if extra_notes:
            preference_parts.append(extra_notes)

        story_steps.append(
            {
                "title": "User preference received",
                "subtitle": "The portfolio was refined based on the user request.",
                "detail": " | ".join(preference_parts) if preference_parts else "No explicit preference text was stored.",
                "status": "done",
            }
        )

        news_used = bool(
            refined_state.get("prob_news_trace")
            or refined_state.get("news_adjustment_evaluation")
        )

        if refined_state.get("prediction_model_used"):
            # prediction constraint için özel timeline adımı
            summary = refined_state.get("prediction_constraint_summary") or {}
            bullish = summary.get("bullish") or []
            bearish = summary.get("bearish") or []
            parts = []
            if bullish:
                parts.append(f"Bullish min-floor: {', '.join(bullish)}")
            if bearish:
                parts.append(f"Bearish max-cap: {', '.join(bearish)}")
            story_steps.append({
                "title": "News prediction constraints applied",
                "subtitle": "Trained LogisticRegression model constrained feasible portfolio allocations.",
                "detail": " | ".join(parts) if parts else "Side constraints from prediction model applied.",
                "status": "done",
            })
        elif news_used:
            trace = refined_state.get("prob_news_trace") or {}
            ticker_signals = trace.get("ticker_signals") or {}
            prediction_signals = trace.get("prediction_signals") or {}

            strongest = None
            for ticker, sig in ticker_signals.items():
                pred = prediction_signals.get(ticker) or {}
                risk_adj = abs(_safe_float(pred.get("risk_adjustment")) or 0.0)
                ret_adj = abs(_safe_float(pred.get("expected_return_adjustment")) or 0.0)
                score = risk_adj + ret_adj

                if strongest is None or score > strongest["score"]:
                    strongest = {
                        "ticker": ticker,
                        "score": score,
                        "sentiment": sig.get("sentiment_score"),
                        "confidence": sig.get("confidence_score"),
                        "direction": pred.get("predicted_direction"),
                        "risk_adj": pred.get("risk_adjustment"),
                        "ret_adj": pred.get("expected_return_adjustment"),
                    }

            if strongest:
                detail = (
                    f"{strongest['ticker']} had the strongest news-model effect. "
                    f"Direction: `{strongest.get('direction') or '–'}` | "
                    f"Return adjustment {_fmt_signed_pct_from_decimal(strongest.get('ret_adj'), 2)} | "
                    f"Risk adjustment {_fmt_signed_pct_from_decimal(strongest.get('risk_adj'), 2)}"
                )
            else:
                detail = "News signals were used to adjust expected return and risk inputs."

            story_steps.append(
                {
                    "title": "News integrated into model",
                    "subtitle": "FinBERT/news signals adjusted the optimization inputs before re-optimization.",
                    "detail": detail,
                    "status": "done",
                }
            )
        else:
            story_steps.append(
                {
                    "title": "No mathematical news integration",
                    "subtitle": "This refinement did not adjust return/risk inputs using FinBERT signals.",
                    "detail": "The refined portfolio was driven by user preferences or candidate selection logic.",
                    "status": "neutral",
                }
            )

        base_w = _weights_from_state(base_state)
        ref_w = _weights_from_state(refined_state)

        if base_w is not None and ref_w is not None:
            all_tickers = sorted(set(base_w.index).union(set(ref_w.index)))
            changes = []
            for t in all_tickers:
                delta = float(ref_w.get(t, 0.0) - base_w.get(t, 0.0))
                if abs(delta) > 1e-6:
                    changes.append((t, delta))

            changes = sorted(changes, key=lambda x: abs(x[1]), reverse=True)

            if changes:
                top_changes = ", ".join([f"{t} {d:+.1%}" for t, d in changes[:4]])
                detail = f"Largest allocation changes: {top_changes}."
            else:
                detail = "Portfolio weights stayed almost unchanged."


            if refined_state.get("prediction_model_used"):
                weight_subtitle = (
                    "Portfolio weights were constrained using predicted news probabilities."
                )
            else:
                weight_subtitle = (
                    "The optimizer selected the final allocation after applying the refinement inputs."
                )

            story_steps.append(
                {
                    "title": "Portfolio weights updated",
                    "subtitle": weight_subtitle,                     
                    "detail": detail,
                                "status": "done",
                            }
                        )
        if refined_state.get("prediction_model_used"):
            ref_obj = _get_chosen_candidate(base_state)
        else:
            ref_obj = _get_chosen_candidate(refined_state)
        ref_m = refined_state.get("optimized_metrics") or {}

        story_steps.append(
            {
                "title": "Final portfolio selected",
                "subtitle": f"Final candidate selected with `{ref_obj}` objective.",
                "detail": (
                    f"Return {_fmt_pct_from_decimal(_safe_float(ref_m.get('return')))} | "
                    f"Volatility {_fmt_pct_from_decimal(_safe_float(ref_m.get('vol')))} | "
                    f"Sharpe {_fmt_num(_safe_float(ref_m.get('sharpe')))}"
                ),
                "status": "done",
            }
        )
    else:
        story_steps.append(
            {
                "title": "Waiting for refinement",
                "subtitle": "Run a refinement step to extend the story.",
                "detail": "Examples: make it safer, exclude NVDA, use mathematical news integration.",
                "status": "neutral",
            }
        )

    cards_html = ""

    for i, step in enumerate(story_steps, start=1):
        status = step.get("status", "neutral")

        if status == "done":
            glow = "#4ade80"
            border = "rgba(74,222,128,0.35)"
            bg = "rgba(74,222,128,0.08)"
        else:
            glow = "#94a3b8"
            border = "rgba(148,163,184,0.25)"
            bg = "rgba(148,163,184,0.05)"

        connector = ""
        if i < len(story_steps):
            connector = f"""
            <div class="connector" style="background:linear-gradient(to bottom, {glow}, rgba(148,163,184,0.05));"></div>
            """

        cards_html += f"""
        <div class="timeline-row">
            <div class="timeline-left">
                <div class="timeline-dot" style="background:{bg}; border-color:{border}; color:{glow}; box-shadow:0 0 22px {glow};">
                    {i}
                </div>
                {connector}
            </div>

            <div class="timeline-card" style="border-color:{border};">
                <div class="step-label" style="color:{glow};">STEP {i}</div>
                <div class="step-title">{step.get("title", "")}</div>
                <div class="step-subtitle">{step.get("subtitle", "")}</div>
                <div class="step-detail">{step.get("detail", "")}</div>
            </div>
        </div>
        """

    timeline_html = f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background: transparent;
            font-family: Inter, Arial, sans-serif;
            color: #e2e8f0;
        }}

        .timeline-wrap {{
            width: 100%;
            padding: 8px 4px 18px 4px;
        }}

        .timeline-row {{
            display: flex;
            align-items: flex-start;
            gap: 18px;
        }}

        .timeline-left {{
            width: 56px;
            display: flex;
            flex-direction: column;
            align-items: center;
            flex-shrink: 0;
        }}

        .timeline-dot {{
            width: 44px;
            height: 44px;
            border-radius: 999px;
            border: 2px solid;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1rem;
        }}

        .connector {{
            width: 3px;
            height: 42px;
            border-radius: 999px;
            margin: 7px 0;
        }}

        .timeline-card {{
            flex: 1;
            background: linear-gradient(135deg, rgba(11,16,32,0.98), rgba(15,23,42,0.94));
            border: 1px solid;
            border-radius: 20px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 14px 35px rgba(0,0,0,0.35);
        }}

        .step-label {{
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: 0.04em;
        }}

        .step-title {{
            color: #f8fafc;
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 8px;
        }}

        .step-subtitle {{
            color: #cbd5e1;
            font-size: 0.94rem;
            line-height: 1.45;
            margin-bottom: 9px;
        }}

        .step-detail {{
            color: #94a3b8;
            font-size: 0.84rem;
            line-height: 1.45;
        }}
    </style>
    </head>
    <body>
        <div class="timeline-wrap">
            {cards_html}
        </div>
    </body>
    </html>
    """

    components.html(timeline_html, height=170 * len(story_steps), scrolling=False)
# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Financial Risk & Portfolio Optimizer", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #050816; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .card {
        background: #0b1020;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #20263a;
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #0b1020;
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid #20263a;
        text-align: left;
    }
    .metric-label { font-size: 0.85rem; color: #8b9ac5; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f7f9ff; }
    .metric-sub { font-size: 0.8rem; color: #9aa6d4; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e6ff;
        margin-bottom: 0.5rem;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #f7f9ff; margin-bottom: 0.2rem; }
    .header-sub { font-size: 0.95rem; color: #9aa6d4; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "base_state" not in st.session_state:
    st.session_state["base_state"] = None
if "refined_state" not in st.session_state:
    st.session_state["refined_state"] = None
if "current_input_df" not in st.session_state:
    st.session_state["current_input_df"] = None
if "pain_points" not in st.session_state:
    st.session_state["pain_points"] = []
if "news_actions_state" not in st.session_state:
    st.session_state["news_actions_state"] = None
if "selected_news_actions" not in st.session_state:
    st.session_state["selected_news_actions"] = []
if "temp_selected_actions" not in st.session_state:
    st.session_state["temp_selected_actions"] = []
if "news_overview_state" not in st.session_state:
    st.session_state["news_overview_state"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_last_command" not in st.session_state:
    st.session_state["chat_last_command"] = None

if "chat_pending_clarification" not in st.session_state:
    st.session_state["chat_pending_clarification"] = None

if "chat_selected_news_mode" not in st.session_state:
    st.session_state["chat_selected_news_mode"] = None

if "news_prediction_state" not in st.session_state:
    st.session_state["news_prediction_state"] = None



# ---------------- LAYOUT ----------------
col_left, col_mid, col_right = st.columns([1.25, 1.05, 1.05])

# ---------------- LEFT: CONTROLS ----------------
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Setup</div>', unsafe_allow_html=True)

    all_tickers = load_available_tickers()

    selected_tickers = st.multiselect(
        "Universe (stocks to include)",
        options=all_tickers,
        default=all_tickers,
    )

    st.markdown(
        '<div class="section-title" style="margin-top:0.8rem;"> Current Portfolio (optional)</div>',
        unsafe_allow_html=True,
    )
    use_current = st.checkbox("I have an existing portfolio (compare vs optimized)", value=False)

    current_weights_dict = None
    current_mode = None

    if use_current:
        current_mode = st.selectbox(
            "How do you want to enter your current portfolio?",
            ["Percent (%)", "Amount (EUR)", "Weight (0-1)"],
            index=0,
        )

    if use_current and selected_tickers:
        if st.session_state["current_input_df"] is None:
            st.session_state["current_input_df"] = (
                pd.DataFrame({"Ticker": selected_tickers, "Value": [0.0] * len(selected_tickers)})
                .set_index("Ticker")
            )
        else:
            st.session_state["current_input_df"] = (
                st.session_state["current_input_df"].reindex(selected_tickers).fillna(0.0)
            )

        if current_mode == "Percent (%)":
            col_label, step, fmt = "Portfolio share (%)", 1.0, "%.2f"
        elif current_mode == "Amount (EUR)":
            col_label, step, fmt = "Invested amount (EUR)", 50.0, "%.2f"
        else:
            col_label, step, fmt = "Weight (0–1)", 0.01, "%.4f"

        edited_df = st.data_editor(
            st.session_state["current_input_df"],
            num_rows="fixed",
            column_config={"Value": st.column_config.NumberColumn(col_label, min_value=0.0, step=step, format=fmt)},
            use_container_width=True,
        )
        st.session_state["current_input_df"] = edited_df.copy()
        current_weights_dict = _safe_normalize_current_inputs(st.session_state["current_input_df"], current_mode)

    st.markdown(
        '<div class="section-title" style="margin-top:0.8rem;"> Optimization settings</div>',
        unsafe_allow_html=True,
    )

    rf = st.number_input(
        "Risk-free rate (annual)",
        value=0.02,
        min_value=-0.05,
        max_value=0.20,
        step=0.005,
        format="%.3f",
    )
    w_max = st.slider("Max weight per asset (hard cap)", min_value=0.05, max_value=1.00, value=0.30, step=0.05)

    run_base = st.button(" Run Base Portfolio", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def _run_base_flow(
    *,
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    current_weights_dict: Optional[Dict[str, float]],
):
    base_state = run_graph(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        preferences={},
        current_weights=current_weights_dict,
        clarification_answers=None,
        mode="base",
        use_llm=True,
        use_news=False,
    )
    st.session_state["base_state"] = base_state
    st.session_state["refined_state"] = None
    st.session_state["pain_points"] = []
    return base_state


def _run_news_overview_flow(
    *,
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    current_weights_dict: Optional[Dict[str, float]],
):
    base_state = st.session_state.get("base_state") or {}

    news_overview_state = run_graph(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        preferences={},
        current_weights=current_weights_dict,
        mode="refine",
        stage="news_overview",
        use_llm=True,
        use_news=True,
        clarification_answers={"satisfaction": "yes", "use_news": "yes"},
        base_portfolio_metrics=base_state.get("optimized_metrics"),
        base_portfolio_weights=base_state.get("optimized_weights"),
        base_portfolio_objective=base_state.get("objective_key"),
    )

    st.session_state["news_overview_state"] = news_overview_state
    return news_overview_state


def _run_news_actions_flow(
    *,
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    current_weights_dict: Optional[Dict[str, float]],
):
    base_state = st.session_state.get("base_state") or {}

    news_actions_state = run_graph(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        preferences={},
        current_weights=current_weights_dict,
        mode="refine",
        stage="news_actions",
        use_llm=True,
        use_news=True,
        clarification_answers={"satisfaction": "yes"},
        base_portfolio_metrics=base_state.get("optimized_metrics"),
        base_portfolio_weights=base_state.get("optimized_weights"),
        base_portfolio_objective=base_state.get("objective_key"),
    )
    print("\n===== NEWS ACTIONS DEBUG NOTES =====")

    for note in (news_actions_state.get("debug_notes") or []):
        if (
            "NewsActions" in note
            or "Evidence" in note
            or "Drop" in note
            or "[CHECK]" in note
        ):
            print(note)

    print("===== /NEWS ACTIONS DEBUG NOTES =====\n")

    st.session_state["news_actions_state"] = news_actions_state
    st.session_state["selected_news_actions"] = []
    return news_actions_state


def _run_refine_flow(
    *,
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    current_weights_dict: Optional[Dict[str, float]],
    pain_points: List[str],
    excluded_assets: List[str],
    extra_notes: str,
    use_llm_refine: bool,
):
    base_state = st.session_state.get("base_state") or {}

    refined_answers = {
        "satisfaction": "no",
        "pain_points": pain_points,
        "excluded_assets": excluded_assets,
        "use_news": "no",
        "extra_notes": extra_notes,
        "notes_tickers": _extract_tickers_from_notes(extra_notes, selected_tickers),
        "selected_news_actions": st.session_state.get("selected_news_actions", []),
    }
    print("\n===== FRONTEND DEBUG: refined_answers sent to run_graph =====")
    print(json.dumps(refined_answers, indent=2, default=str))
    print("=============================================================\n")

    refined_state = run_graph(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        preferences={},
        current_weights=current_weights_dict,
        clarification_answers=refined_answers,
        mode="refine",
        use_llm=bool(use_llm_refine),
        use_news=False,
        base_portfolio_metrics=base_state.get("optimized_metrics"),
        base_portfolio_weights=base_state.get("optimized_weights"),
        base_portfolio_objective=base_state.get("objective_key"),
    )
    print("\n=== NORMAL REFINE DEBUG ===")
    print("optimized_weights exists:",
        refined_state.get("optimized_weights") is not None)

    print("chosen_candidate:",
        refined_state.get("chosen_candidate"))

    print("objective_key:",
        refined_state.get("objective_key"))

    print("===========================\n")

    st.session_state["refined_state"] = refined_state
    return refined_state

def _run_prob_news_refine_flow(
    *,
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    current_weights_dict: Optional[Dict[str, float]],
    pain_points: List[str],
    excluded_assets: List[str],
    extra_notes: str,
    use_llm_refine: bool,
):
    base_state = st.session_state.get("base_state") or {}

    refined_answers = {
        "satisfaction": "no",
        "pain_points": pain_points,
        "excluded_assets": excluded_assets,
        "use_news": "yes",
        "extra_notes": extra_notes,
        "notes_tickers": _extract_tickers_from_notes(extra_notes, selected_tickers),
        "selected_news_actions": [],
    }

    print("\n===== FRONTEND DEBUG: refined_answers sent to run_graph_prob_news =====")
    print(json.dumps(refined_answers, indent=2, default=str))
    print("========================================================================\n")

    print("\n" + "="*60)
    print("DEBUG _run_prob_news_refine_flow CALLED")
    print(f"  selected_tickers: {selected_tickers}")
    print(f"  use_news=True, use_llm={use_llm_refine}")
    print(f"  base_state keys: {list(base_state.keys()) if base_state else 'EMPTY'}")
    print(f"  base_portfolio_objective: {base_state.get('objective_key')}")
    print(f"  refined_answers: {refined_answers}")
    print("="*60 + "\n")

    refined_state = run_graph_prob_news(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        preferences={},
        current_weights=current_weights_dict,
        clarification_answers=refined_answers,
        mode="refine",
        stage="main",
        use_llm=bool(use_llm_refine),
        use_news=True,
        base_portfolio_metrics=base_state.get("optimized_metrics"),
        base_portfolio_weights=base_state.get("optimized_weights"),
        base_portfolio_objective=base_state.get("objective_key"),
        prob_alpha=0.15,
        prob_beta=0.08,
    )
    print("\n" + "="*60)
    print("DEBUG _run_prob_news_refine_flow RESULT")
    print(f"  refined_state keys: {list(refined_state.keys()) if refined_state else 'NONE'}")
    print(f"  news_raw length: {len(refined_state.get('news_raw') or [])}")
    print(f"  use_news in state: {refined_state.get('use_news')}")
    print(f"  probabilistic_news_used: {refined_state.get('probabilistic_news_used')}")
    print(f"  prob_news_trace: {'SET' if refined_state.get('prob_news_trace') else 'NONE'}")
    print(f"  prob_news_signals: {'SET' if refined_state.get('prob_news_signals') else 'NONE'}")
    print(f"  debug_notes relevant:")
    for note in (refined_state.get('debug_notes') or []):
        if any(x in note for x in ['NewsFetch', 'PROB_NEWS', 'use_news', 'Perception', 'news_raw']):
            print(f"    → {note}")
    print("="*60 + "\n")

    st.session_state["refined_state"] = refined_state
    return refined_state

def run_prediction_constrained_optimization(
    selected_tickers,
    rf,
    w_max,
    current_weights_dict,
):
    """
    Runs prediction-constrained portfolio optimization
    after trained prediction model visualization.
    """

    base_state = st.session_state.get("base_state") or {}

    prediction_state = run_graph_prediction_constraint(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),

        preferences={},

        current_weights=current_weights_dict,

        clarification_answers={
            "satisfaction": "no",
            "use_news": "yes",
            "extra_notes": "Prediction constrained optimization",
        },

        mode="refine",
        stage="main",

        use_llm=True,
        use_news=True,

        base_portfolio_metrics=base_state.get("optimized_metrics"),
        base_portfolio_weights=base_state.get("optimized_weights"),
        base_portfolio_objective=base_state.get("objective_key"),
    )
    print("DEBUG prediction_model_used:",
      prediction_state.get("prediction_model_used"))

    print("DEBUG has_prob_news_trace:",
        prediction_state.get("prob_news_trace") is not None)

    print("DEBUG has_constraint_debug:",
        prediction_state.get("constraint_debug") is not None)

    print("DEBUG optimized_weights exists:",
        prediction_state.get("optimized_weights") is not None)

    print("DEBUG prediction_probs exists:",
        prediction_state.get("prediction_probs") is not None)

    print("DEBUG prediction_adjusted_caps exists:",
        prediction_state.get("prediction_adjusted_caps") is not None)

    st.session_state["refined_state"] = prediction_state

    return prediction_state
# ---------------- RUN BASE ----------------
if run_base and selected_tickers:
    _run_base_flow(
        selected_tickers=selected_tickers,
        rf=float(rf),
        w_max=float(w_max),
        current_weights_dict=current_weights_dict,
    )
    st.rerun()


def _append_chat_message(role: str, content: str, kind: str = "text", payload: Optional[Dict[str, Any]] = None):
    st.session_state["chat_history"].append(
        {
            "role": role,
            "content": content,
            "kind": kind,
            "payload": payload or {},
        }
    )


def _handle_chat_command(
    *,
    user_msg: str,
    selected_tickers: List[str],
    rf: float,
    w_max: float,
    current_weights_dict: Optional[Dict[str, float]],
):
    if not selected_tickers:
        _append_chat_message("assistant", "Please select at least one ticker in the universe first.")
        return
    # ✅ FIX: news keyword'lerini LLM'den önce yakala
    _msg_lower = user_msg.lower().strip()
    _news_phrases = [
        "use news", "include news", "apply news", "consider news",
        "add news", "news integration", "news model", "prediction model",
        "show news prediction", "news prediction model", "use recent news",
        "news in the portfolio", "integrate news",
    ]
    if any(phrase in _msg_lower for phrase in _news_phrases):
        st.session_state["chat_pending_clarification"] = {
            "type": "news_mode_selection",
            "original_user_msg": user_msg,
        }
        _append_chat_message(
            "assistant",
            "I can use news in three different ways. Please choose one option below.",
            kind="news_mode_selection",
            payload={
                "question": "Which one do you want?",
                "options": [
                    {"label": "Mathematical news integration", "value": "probabilistic",
                     "description": "Use news signals to adjust return/risk inputs before optimization."},
                    {"label": "LLM news actions", "value": "llm_actions",
                     "description": "Use news to generate qualitative actions, explanations, and suggestions."},
                    {"label": "Trained news prediction model", "value": "prediction_model",
                     "description": "Show saved model probabilities for future return direction."},
                ],
            },
        )
        return  # ← LLM'e hiç gitme


    client = LLMClient()
    cmd = client.interpret_dashboard_chat_command(
        message=user_msg,
        selected_tickers=selected_tickers,
    )
    st.session_state["chat_last_command"] = cmd

    intent = str(cmd.get("intent") or "unsupported").strip()
    params = cmd.get("parameters") if isinstance(cmd.get("parameters"), dict) else {}
    # ← GEÇİCİ DEBUG: terminalde göreceksin
    print(f"\n===== CHAT INTENT DEBUG =====")
    print(f"user_msg: {user_msg!r}")
    print(f"intent: {intent!r}")
    print(f"cmd: {cmd}")
    print(f"=============================\n")

    if intent == "run_base_portfolio":
        _run_base_flow(
            selected_tickers=selected_tickers,
            rf=rf,
            w_max=w_max,
            current_weights_dict=current_weights_dict,
        )
        _append_chat_message("assistant", "Base portfolio generated.")

    elif intent == "run_news_overview":
        if st.session_state.get("base_state") is None:
            _append_chat_message("assistant", "Please run the base portfolio first, then I can generate a news overview.")
            return

        out = _run_news_overview_flow(
            selected_tickers=selected_tickers,
            rf=rf,
            w_max=w_max,
            current_weights_dict=current_weights_dict,
        )

        snapshot_text, risk_json = _extract_news_snapshot_and_risk(out or {})
        _append_chat_message(
            "assistant",
            "News overview generated.",
            kind="news_overview",
            payload={
                "snapshot_text": snapshot_text,
                "risk_json": risk_json,
            },
        )

    elif intent == "clarify_news_usage_mode":
        st.session_state["chat_pending_clarification"] = {
            "type": "news_mode_selection",
            "original_user_msg": user_msg,
        }

        _append_chat_message(
            "assistant",
            "I can use news in three different ways. Please choose one option below.",
            kind="news_mode_selection",
            payload={
                "question": "Which one do you want?",
                "options": [
                    {
                        "label": "Mathematical news integration",
                        "value": "probabilistic",
                        "description": "Use news signals to adjust return/risk inputs before optimization."
                    },
                    {
                        "label": "LLM news actions",
                        "value": "llm_actions",
                        "description": "Use news to generate qualitative actions, explanations, and suggestions."
                    },
                    {
                        "label": "Trained news prediction model",
                        "value": "prediction_model",
                        "description": "Show saved model probabilities for future return direction."
                    },
                ],
            },
        )


    elif intent == "generate_news_actions":
        if st.session_state.get("base_state") is None:
            _append_chat_message("assistant", "Please run the base portfolio first, then I can generate actions from the news.")
            return

        out = _run_news_actions_flow(
            selected_tickers=selected_tickers,
            rf=rf,
            w_max=w_max,
            current_weights_dict=current_weights_dict,
        )
        actions = (out or {}).get("news_actions") or []
        n_actions = len(actions)

        _append_chat_message(
            "assistant",
            f"I generated {n_actions} news-based action(s).",
            kind="news_actions",
            payload={
                "state": out,
                "actions": actions,
                "evidence_snapshot": (out or {}).get("news_evidence_snapshot_text"),
            },
        )
    elif intent == "run_refine_candidate_selection":
        if st.session_state.get("base_state") is None:
            _append_chat_message("assistant", "Please run the base portfolio first, then I can refine it.")
            return

        pain_points = params.get("pain_points") if isinstance(params.get("pain_points"), list) else []
        pain_points = [str(x) for x in pain_points if str(x).strip()]
        excluded_assets = params.get("excluded_assets") if isinstance(params.get("excluded_assets"), list) else []
        excluded_assets = [str(x).upper().strip() for x in excluded_assets if str(x).strip()]
        extra_notes = str(params.get("extra_notes") or user_msg).strip()

        out = _run_refine_flow(
            selected_tickers=selected_tickers,
            rf=rf,
            w_max=w_max,
            current_weights_dict=current_weights_dict,
            pain_points=pain_points,
            excluded_assets=excluded_assets,
            extra_notes=extra_notes,
            use_llm_refine=True
        )

        _append_chat_message(
            "assistant",
            "Portfolio refined based on your message.",
            kind="refine_result",
            payload={
                "chosen_candidate": _get_chosen_candidate(out or {}),
                "explanation": (out or {}).get("explanation"),
                "insight_raw_text": (out or {}).get("insight_raw_text"),
            },
        )

    elif intent == "compare_base_refined_metrics":
        base_sum = _portfolio_summary_from_state(st.session_state.get("base_state"))
        ref_sum = _portfolio_summary_from_state(st.session_state.get("refined_state"))

        if base_sum is None:
            _append_chat_message(
                "assistant",
                "Please run the base portfolio first, then I can compare it with the refined portfolio."
            )
            return

        if ref_sum is None:
            _append_chat_message(
                "assistant",
                "Base portfolio is available, but no refined portfolio exists yet. Please refine the portfolio first."
            )
            return

        _append_chat_message(
            "assistant",
            "Here is the Before / After comparison between the base and refined portfolio, including weight changes.",
            kind="base_refined_comparison_bundle",
            payload={},
        )


    elif intent == "show_final_portfolio_insight":
        active_state = st.session_state.get("refined_state") or st.session_state.get("base_state")

        if active_state is None:
            _append_chat_message(
                "assistant",
                "Please run the base portfolio first, then I can explain the active portfolio."
            )
            return

        insight_text = (active_state.get("insight_raw_text") or "").strip()
        explanation_text = str(active_state.get("explanation") or "").strip()
        chosen_candidate = _get_chosen_candidate(active_state)

        # Prefer insight if available, otherwise fallback to explanation
        final_text = insight_text or explanation_text

        if not final_text and not explanation_text:
            _append_chat_message(
                "assistant",
                "No final portfolio insight is available yet.",
                kind="final_portfolio_insight",
                payload={
                    "chosen_candidate": chosen_candidate,
                    "insight_text": "",
                    "explanation_text": "",
                },
            )
            return

        _append_chat_message(
            "assistant",
            "Here is the insight for the active portfolio.",
            kind="final_portfolio_insight",
            payload={
                "chosen_candidate": chosen_candidate,
                "insight_text": final_text,
                "explanation_text": explanation_text,
            },
        )
            
    else:
        reply = str(cmd.get("reply") or "").strip()
        if not reply:
            reply = "This chatbot can help with base portfolio generation, news overview, news actions, and portfolio refinement."
        _append_chat_message("assistant", reply)
# ---------------- HEADER ----------------
is_refined_active = st.session_state["refined_state"] is not None
active_label = "Refined Portfolio" if is_refined_active else "Base Portfolio"

st.markdown(
    f"""
    <div class="card" style="margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="header-title"> Financial Risk & Portfolio Optimizer</div>
          <div class="header-sub">Two-step UX: Run Base → then Refine with candidate selection + insights</div>
          <div class="header-sub" style="margin-top:0.35rem;">Currently showing: <b>{active_label}</b></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _get_compare_state_for_charts():
    """
    Chart comparison should follow Insight story:
    - If refined exists: compare Base Portfolio -> Refined Portfolio
    - Else if user provided Current portfolio: compare Current -> Active (base)
    - Else: fallback to baseline (equal weight) only when nothing else exists
    """
    base_state = st.session_state.get("base_state")
    refined_state = st.session_state.get("refined_state")
    active_state = refined_state or base_state

    if refined_state is not None and base_state is not None:
        prev_metrics = (base_state.get("optimized_metrics") or {})
        prev_label = "Base Portfolio"
        return prev_metrics, prev_label

    if active_state is not None:
        cm = active_state.get("current_metrics")
        if cm:
            return cm, "Current"
        bm = active_state.get("baseline_metrics")
        if bm:
            return bm, "Baseline (Equal Weight)"

    return None, None


# ---------------- ACTIVE STATE ----------------
graph_state = (
    st.session_state["refined_state"]
    or st.session_state["base_state"]
)

is_refined_active = (
    st.session_state["refined_state"] is not None
)
if graph_state and graph_state.get("prediction_model_used"):
    active_label = "Prediction-Constrained Portfolio"

elif graph_state and graph_state.get("probabilistic_news_used"):
    active_label = "News-Integrated Portfolio"

elif is_refined_active:
    active_label = "Refined Portfolio"

else:
    active_label = "Base Portfolio"

optimization_result = None
portfolio_weights = None
portfolio_metrics = None
baseline_metrics = None
optimized_metrics = None
current_metrics = None
chosen_candidate = "maxsharpe"

if graph_state is not None:
    optimization_result, chosen_candidate, portfolio_weights, portfolio_metrics = _extract_weights_and_metrics(graph_state)
    baseline_metrics = graph_state.get("baseline_metrics")
    optimized_metrics = graph_state.get("optimized_metrics")
    current_metrics = graph_state.get("current_metrics")

# ---------------- MID: COMPOSITION ----------------
with col_mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">🧱 Portfolio Composition — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("Click **Run Base Portfolio** to generate the first portfolio.")
    else:
        pie_df = portfolio_weights.reset_index()
        pie_df.columns = ["Ticker", "Weight"]
        fig = px.pie(pie_df, names="Ticker", values="Weight", hole=0.6)
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0b1020",
            plot_bgcolor="#0b1020",
            font=dict(color="#E2E6FF"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT: METRICS ----------------
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📌 Risk & Performance — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_metrics is None:
        st.info("Metrics will appear after base portfolio runs.")
    else:
        opt = portfolio_metrics
        cand = opt["candidate"]
        obj_label = "Max Sharpe" if cand == "maxsharpe" else "Min Variance"
        st.caption(f"Selected candidate: **{obj_label}** (`{cand}`)")

        ret_str = (
            _fmt_pct_from_pct_field(opt.get("return_pct"))
            if opt.get("return_pct") is not None
            else _fmt_pct_from_decimal(_safe_float(opt.get("return")))
        )
        vol_str = (
            _fmt_pct_from_pct_field(opt.get("vol_pct"))
            if opt.get("vol_pct") is not None
            else _fmt_pct_from_decimal(_safe_float(opt.get("vol")))
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{_fmt_num(opt.get("sharpe"))}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Risk-adjusted return</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Return</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{ret_str}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{vol_str}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized std dev</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Assets</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{opt["used_assets"]} / {opt["universe_assets"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="metric-sub">Active / Universe</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Refinement UI (Candidate selection)
# ============================================================
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔁 Refine (after base portfolio)</div>', unsafe_allow_html=True)

if st.session_state["base_state"] is None:
    st.info("Run **Base Portfolio** first. Then you can refine using candidate selection + insights.")
    st.markdown("</div>", unsafe_allow_html=True)  # close card
else:
    st.caption("News snapshot & risk check is optional. Chat-based refine does not use news unless explicitly requested.")

    st.markdown("---")
    st.markdown('<div class="section-title">💬 Portfolio Chatbot</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ What can I ask this chatbot?", expanded=False):
        st.markdown(
            """
            You can ask me to:

            - **Build the base portfolio**  
            Example: `run base portfolio`

            - **Refine the portfolio**  
            Example: `make it safer`  
            Example: `exclude NVDA`

            - **Compare base and refined portfolios**  
            Example: `show me what changed`  
            Example: `compare base and refined`

            - **Explain the active portfolio**  
            Example: `explain this portfolio`

            - **Use recent news**  
            Example: `use news in the portfolio`

            - **Show a news overview**  
            Example: `show news overview`

            - **Generate actions from news**  
            Example: `generate actions from news`

            - **Show trained news prediction model results**
            Example: `show news prediction model`
            """
        )
    for msg_idx, msg in enumerate(st.session_state.get("chat_history", [])):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            kind = msg.get("kind", "text")
            payload = msg.get("payload", {})

            if kind == "news_mode_selection":
                question = payload.get("question") or "Which one do you want?"
                options = payload.get("options") or []

                st.markdown(f"**{question}**")

                option_labels = []
                value_by_label = {}

                for opt in options:
                    if not isinstance(opt, dict):
                        continue
                    label = str(opt.get("label") or "").strip()
                    value = str(opt.get("value") or "").strip()
                    desc = str(opt.get("description") or "").strip()

                    if not label or not value:
                        continue

                    full_label = f"{label} — {desc}" if desc else label
                    option_labels.append(full_label)
                    value_by_label[full_label] = value

                if option_labels:
                    radio_key = f"news_mode_radio_{msg_idx}"
                    selected_label = st.radio(
                        "Choose one option",
                        option_labels,
                        key=radio_key,
                    )

                    if st.button("Confirm selection", key=f"confirm_news_mode_{msg_idx}"):
                        selected_value = value_by_label[selected_label]

                        st.session_state["chat_pending_clarification"] = None
                        st.session_state["chat_selected_news_mode"] = selected_value

                        if st.session_state.get("base_state") is None:
                            _append_chat_message(
                                "assistant",
                                "Please run the base portfolio first, then I can use news in the selected mode."
                            )
                            st.rerun()

                        if selected_value == "probabilistic":
                            out = _run_prob_news_refine_flow(
                                selected_tickers=selected_tickers,
                                rf=float(rf),
                                w_max=float(w_max),
                                current_weights_dict=current_weights_dict,
                                pain_points=[],
                                excluded_assets=[],
                                extra_notes="User requested mathematical news integration into the optimization process.",
                                use_llm_refine=True,
                            )

                            base_state = st.session_state.get("base_state") or {}

                            _append_chat_message(
                                "assistant",
                                "Portfolio refined using mathematical news integration.",
                                kind="prob_news_refine_result",
                                payload={
                                    "chosen_candidate": _get_chosen_candidate(out or {}),
                                    "objective_key": (out or {}).get("objective_key"),
                                    "base_objective": base_state.get("objective_key"),
                                    "optimized_metrics": (out or {}).get("optimized_metrics") or {},
                                    "base_metrics": base_state.get("optimized_metrics") or {},
                                    "explanation": (out or {}).get("explanation"),
                                    "insight_raw_text": (out or {}).get("insight_raw_text"),
                                    "prob_news_trace": (out or {}).get("prob_news_trace"),
                                    "news_adjustment_evaluation": (out or {}).get("news_adjustment_evaluation"),
                                    "prob_prediction_evaluation": (out or {}).get("prob_prediction_evaluation"),

                                    "historical_prediction_evaluation": (out or {}).get("historical_prediction_evaluation")
                                    or ((out or {}).get("prob_news_trace") or {}).get("historical_prediction_evaluation"),


                                },
                            )
                            st.rerun()

                        elif selected_value == "llm_actions":
                            out = _run_news_actions_flow(
                                selected_tickers=selected_tickers,
                                rf=float(rf),
                                w_max=float(w_max),
                                current_weights_dict=current_weights_dict,
                            )

                            actions = (out or {}).get("news_actions") or []
                            n_actions = len(actions)

                            _append_chat_message(
                                "assistant",
                                f"I generated {n_actions} news-based action(s).",
                                kind="news_actions",
                                payload={
                                    "state": out,
                                    "actions": actions,
                                    "evidence_snapshot": (out or {}).get("news_evidence_snapshot_text"),
                                },
                            )
                            st.rerun()
                        elif selected_value == "prediction_model":
                            out = load_saved_news_prediction_outputs()
                            st.session_state["news_prediction_state"] = out

                            if out.get("ok"):
                                _append_chat_message(
                                    "assistant",
                                    "Loaded the saved news return prediction model results.",
                                    kind="news_prediction_model",
                                    payload={
                                        "state": out,
                                    },
                                )
                            else:
                                _append_chat_message(
                                    "assistant",
                                    out.get("reason", "Could not load saved news prediction results."),
                                )

                            st.rerun()


            elif kind == "news_overview":
                snapshot = payload.get("snapshot_text")
                risk_json = payload.get("risk_json")

                if snapshot:
                    st.markdown("**News overview**")
                    st.write(snapshot)

                if isinstance(risk_json, dict) and risk_json:
                    glob = risk_json.get("global", {})
                    by_ticker = risk_json.get("by_ticker", {})

                    if isinstance(glob, dict) and glob:
                        vol_regime = str(glob.get("vol_regime") or "normal")
                        st.caption(f"Global regime: **{vol_regime}**")



            elif kind == "news_actions":
                nas = payload.get("state") or {}
                actions = payload.get("actions") or []
                evidence_snapshot = payload.get("evidence_snapshot")

                if actions:
                    news_items_by_id = _get_news_items_by_id(nas)

                    st.markdown("**Proposed actions**")

                    selected_action_indices = []

                    for i, a in enumerate(actions, start=1):
                        if isinstance(a, dict):
                            t = str(a.get("type") or a.get("action") or "").strip()
                            ticker = str(a.get("ticker") or "").strip()
                            intensity = str(a.get("intensity") or "").strip()
                            reason = str(a.get("reason") or "").strip()

                            label_parts = [x for x in [t, ticker, intensity] if x]
                            base_label = " | ".join(label_parts) if label_parts else f"Action #{i}"

                            checkbox_label = base_label
                            if a.get("weak_evidence"):
                                checkbox_label += " ⚠️ (indirect evidence)"

                            checked = st.checkbox(
                                f"Select: {checkbox_label}",
                                key=f"news_action_checkbox_{msg_idx}_{i}",
                            )

                            if checked:
                                selected_action_indices.append(i - 1)

                            with st.expander(base_label, expanded=False):
                                if reason:
                                    st.write(reason)

                                _render_action_evidence(
                                    action=a,
                                    news_items_by_id=news_items_by_id,
                                    universe_tickers=selected_tickers,
                                )
                        else:
                            st.write(f"- {a}")

                    st.session_state["temp_selected_actions"] = selected_action_indices


                    if st.button("Apply selected actions", key=f"apply_news_actions_{msg_idx}"):
                            selected_actions = [actions[idx] for idx in selected_action_indices]

                            print("\n===== FRONTEND DEBUG: selected_actions from checkboxes =====")
                            print(json.dumps(selected_actions, indent=2, default=str))
                            print("===========================================================\n")

                            # refine backend'e gönderilecek gerçek selection
                            st.session_state["selected_news_actions"] = selected_actions

                            print("\n===== FRONTEND DEBUG: selected_news_actions saved to session =====")
                            print(json.dumps(st.session_state["selected_news_actions"], indent=2, default=str))
                            print("=================================================================\n")

                            _append_chat_message(
                                "assistant",
                                f"Applying {len(selected_actions)} selected action(s) to refine the portfolio..."
                            )

                            out = _run_refine_flow(
                                selected_tickers=selected_tickers,
                                rf=float(rf),
                                w_max=float(w_max),
                                current_weights_dict=current_weights_dict,
                                pain_points=[],
                                excluded_assets=[],
                                extra_notes="Applying selected news actions",
                                use_llm_refine=False,
                            )

                            _append_chat_message(
                                "assistant",
                                "Portfolio refined using selected actions.",
                                kind="refine_result",
                                payload={
                                    "chosen_candidate": _get_chosen_candidate(out or {}),
                                    "explanation": (out or {}).get("explanation"),
                                    "insight_raw_text": (out or {}).get("insight_raw_text"),
                                },
                            )

                            # artık eski seçim state'te kalmasın
                            st.session_state["selected_news_actions"] = []
                            st.session_state["temp_selected_actions"] = []

                            for i in range(1, len(actions) + 1):
                                checkbox_key = f"news_action_checkbox_{msg_idx}_{i}"
                                if checkbox_key in st.session_state:
                                    del st.session_state[checkbox_key]

                            st.rerun()

                if evidence_snapshot:
                    st.markdown("**Evidence snapshot**")
                    st.markdown(evidence_snapshot)

            

            elif kind == "base_refined_comparison_bundle":
                base_sum = _portfolio_summary_from_state(st.session_state.get("base_state"))
                ref_sum = _portfolio_summary_from_state(st.session_state.get("refined_state"))

                st.markdown("**Before / After Metric Comparison**")
                _render_base_vs_refined_metric_chart(
                    base_sum,
                    ref_sum,
                    chart_key=f"chat_metric_comparison_chart_{msg_idx}",
                )

                st.markdown("**Weight Changes by Asset**")
                _render_weight_change_chart(
                    st.session_state.get("base_state"),
                    st.session_state.get("refined_state"),
                    chart_key=f"chat_weight_change_chart_{msg_idx}",
                )

            elif kind == "refine_result":
                chosen = payload.get("chosen_candidate")
                explanation = payload.get("explanation")
                insight_text = payload.get("insight_raw_text")

                st.markdown("**Refine result**")
                if chosen:
                    st.write(f"Chosen candidate: `{chosen}`")
                if explanation:
                    st.write(explanation)
                if insight_text:
                    st.markdown("**Insight**")
                    st.markdown(insight_text)

            elif kind == "prob_news_refine_result":
                chosen = payload.get("chosen_candidate")
                explanation = payload.get("explanation")
                insight_text = payload.get("insight_raw_text")
                optimized_metrics = payload.get("optimized_metrics") or {}
                base_metrics = payload.get("base_metrics") or {}
                objective_key = payload.get("objective_key")
                base_objective = payload.get("base_objective")
                prob_news_trace = payload.get("prob_news_trace")
                news_adjustment_evaluation = payload.get("news_adjustment_evaluation")
                prob_prediction_evaluation = payload.get("prob_prediction_evaluation")

                historical_prediction_evaluation = payload.get("historical_prediction_evaluation")
                

                st.markdown("**News-integrated refine result**")
                st.write("Recent news was incorporated into the optimization inputs before re-optimizing the portfolio.")

                if base_objective or objective_key:
                    st.write(f"Base objective: `{base_objective}` → News-adjusted result: `{objective_key}`")

                if chosen:
                    st.write(f"Chosen candidate: `{chosen}`")

                if optimized_metrics:
                    ret_str = _fmt_pct_from_decimal(_safe_float(optimized_metrics.get("return")))
                    vol_str = _fmt_pct_from_decimal(_safe_float(optimized_metrics.get("vol")))
                    sharpe_str = _fmt_num(_safe_float(optimized_metrics.get("sharpe")))
                    st.write(f"Return: {ret_str} | Volatility: {vol_str} | Sharpe: {sharpe_str}")

                if base_metrics and optimized_metrics:
                    d_ret = _safe_diff(optimized_metrics.get("return"), base_metrics.get("return"))
                    d_vol = _safe_diff(optimized_metrics.get("vol"), base_metrics.get("vol"))
                    d_sharpe = _safe_diff(optimized_metrics.get("sharpe"), base_metrics.get("sharpe"))

                    st.markdown("**Change vs base portfolio**")
                    st.write(f"Δ Return: {_fmt_pct_from_decimal(d_ret)}")
                    st.write(f"Δ Volatility: {_fmt_pct_from_decimal(d_vol)}")
                    st.write(f"Δ Sharpe: {_fmt_num(d_sharpe)}")

                if news_adjustment_evaluation:
                    with st.expander("Show portfolio-level news adjustment evaluation", expanded=True):
                        _render_news_adjustment_evaluation(news_adjustment_evaluation)

                if historical_prediction_evaluation:
                    with st.expander("Show historical news predictive evaluation", expanded=False):
                        _render_prediction_evaluation(
                            historical_prediction_evaluation,
                            title="Historical news predictive evaluation",
                            caption=(
                                "This does not test FinBERT as a general sentiment classifier. "
                                "It tests whether FinBERT-based historical news signals were directionally aligned "
                                "with subsequent stock returns for the selected portfolio tickers."
                            ),
                        )

                with st.expander("Show expected return calculation debug"):
                    debug_path = "data/processed_yahoo/debug_daily_vs_annual_returns.csv"

                    st.caption(
                        "Expected returns are annualized from historical daily mean returns: "
                        "μ_annual = μ_daily × 252. These are model inputs, not guaranteed future returns."
                    )

                    if os.path.exists(debug_path):
                        df_debug = pd.read_csv(debug_path, index_col=0)

                        # Optional: sadece seçili tickerları göster
                        selected_debug_tickers = [t for t in selected_tickers if t in df_debug.index]
                        if selected_debug_tickers:
                            df_debug = df_debug.loc[selected_debug_tickers]

                        st.dataframe(
                            df_debug[
                                ["mu_daily_pct", "mu_annual_pct", "sigma_daily_pct", "sigma_annual_pct", "sharpe"]
                            ].round(4),
                            use_container_width=True,
                        )
                    else:
                        st.info("Debug file not found. Run the Yahoo preprocessing/build script first.")



                if prob_news_trace:
                        with st.expander("Show how news changed the mathematical model"):
                            _render_prob_news_trace(prob_news_trace)

                if explanation:
                    with st.expander("Show decision explanation"):
                        st.write(explanation)

                if insight_text:
                    st.markdown("**Insight**")
                    st.markdown(insight_text)
            elif kind == "news_prediction_model":
                st.markdown("**Trained News Prediction Model**")

                # ✅ FIX: sadece prediction_model_used=True değilse çalıştır
                # Böylece prob_news flow'u (veya başka bir flow) refined_state'i set etmişse EZMEYİZ
                active_state = st.session_state.get("refined_state")
                if active_state is None or not active_state.get("prediction_model_used"):
                    run_prediction_constrained_optimization(
                        selected_tickers=selected_tickers,
                        rf=rf,
                        w_max=w_max,
                        current_weights_dict=current_weights_dict,
                    )
                    st.rerun()

                # rerun sonrası buraya gelir
                active_state = st.session_state.get("refined_state")

                _render_saved_news_prediction_model(
                    selected_tickers=selected_tickers,
                    active_state=active_state,
                )
                _render_prediction_weight_shift_chart(
                    active_state,
                    chart_key=f"prediction_shift_{msg_idx}",
                )
                _render_prediction_constraint_table(active_state)

                if active_state:
                    st.markdown("## Prediction-Constrained Portfolio")
                    st.write(active_state.get("optimized_weights"))
                    st.write(active_state.get("optimized_metrics"))
                            
            elif kind == "final_portfolio_insight":
                chosen = payload.get("chosen_candidate")
                insight_text = payload.get("insight_text")
                explanation_text = payload.get("explanation_text")

                st.markdown("**Final portfolio insight**")
                if chosen:
                    st.write(f"Active candidate: `{chosen}`")

                if insight_text:
                    st.markdown(insight_text)
                else:
                    st.write("No insight text available.")

                if explanation_text:
                    with st.expander("Show technical explanation"):
                        st.write(explanation_text)

    chat_msg = st.chat_input(
        "Ask something like: build base portfolio, use news, show news prediction model, generate actions from news, make it safer, exclude NVDA"
    )

    if chat_msg:
        _append_chat_message("user", chat_msg)
        _handle_chat_command(
            user_msg=chat_msg,
            selected_tickers=selected_tickers,
            rf=float(rf),
            w_max=float(w_max),
            current_weights_dict=current_weights_dict,
        )
        st.rerun()


    st.markdown("---")



st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Evaluation: Base vs Refine ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Evaluation — Base vs Refine</div>', unsafe_allow_html=True)

base_sum = _portfolio_summary_from_state(st.session_state.get("base_state"))
ref_sum = _portfolio_summary_from_state(st.session_state.get("refined_state"))

if base_sum is None:
    st.info("Run **Base Portfolio** first to enable evaluation.")
else:
    st.markdown("**Before / After Metric Comparison**")
    _render_base_vs_refined_metric_chart(
    base_sum,
    ref_sum,
    chart_key="evaluation_base_vs_refined_metric_chart",
)

    st.markdown("---")
    col1, col2, col3 = st.columns([1.0, 1.0, 1.0])

    with col1:
        st.markdown("**Base**")
        st.write(f"- Candidate: `{base_sum['candidate']}`")
        st.write(f"- Return: {_fmt_pct_from_decimal(_safe_float(base_sum['return']))}")
        st.write(f"- Vol: {_fmt_pct_from_decimal(_safe_float(base_sum['vol']))}")
        st.write(f"- Sharpe: {_fmt_num(base_sum['sharpe'])}")
        st.write(f"- Active assets: {base_sum['active_assets']}")
        st.write(f"- Max weight: {_fmt_pct_from_decimal(_safe_float(base_sum['max_weight']))}")
        st.write(f"- Effective N: {base_sum['effective_n']:.1f}")

    with col2:
        st.markdown("**Refined**")
        if ref_sum is None:
            st.write("Not computed yet.")
        else:
            st.write(f"- Candidate: `{ref_sum['candidate']}`")
            st.write(f"- Return: {_fmt_pct_from_decimal(_safe_float(ref_sum['return']))}")
            st.write(f"- Vol: {_fmt_pct_from_decimal(_safe_float(ref_sum['vol']))}")
            st.write(f"- Sharpe: {_fmt_num(ref_sum['sharpe'])}")
            st.write(f"- Active assets: {ref_sum['active_assets']}")
            st.write(f"- Max weight: {_fmt_pct_from_decimal(_safe_float(ref_sum['max_weight']))}")
            st.write(f"- Effective N: {ref_sum['effective_n']:.1f}")

    with col3:
        st.markdown("**Delta (Refined − Base)**")
        if ref_sum is None:
            st.write("Run refinement to see deltas.")
        else:
            d_ret = _safe_diff(ref_sum.get("return"), base_sum.get("return"))
            d_vol = _safe_diff(ref_sum.get("vol"), base_sum.get("vol"))

            d_sh = (
                (float(ref_sum["sharpe"]) - float(base_sum["sharpe"]))
                if (_safe_float(ref_sum.get("sharpe")) is not None and _safe_float(base_sum.get("sharpe")) is not None)
                else None
            )
            d_eff = float(ref_sum["effective_n"] - base_sum["effective_n"])
            d_mx = _safe_diff(ref_sum.get("max_weight"), base_sum.get("max_weight"))
            d_act = int(ref_sum["active_assets"] - base_sum["active_assets"])

            st.write(f"- Δ Return: {_fmt_pct_from_decimal(d_ret)}")
            st.write(f"- Δ Vol: {_fmt_pct_from_decimal(d_vol)}")
            st.write(f"- Δ Sharpe: {_fmt_num(d_sh)}")
            st.write(f"- Δ Max weight: {_fmt_pct_from_decimal(d_mx)}")
            st.write(f"- Δ Effective N: {d_eff:+.1f}")
            st.write(f"- Δ Active assets: {d_act:+d}")

with st.expander("📦 Export run logs (JSON)"):
    if st.session_state.get("base_state") is not None:
        st.download_button(
            "Download BASE state (JSON)",
            data=json.dumps(st.session_state["base_state"], indent=2, default=str),
            file_name="base_state.json",
            mime="application/json",
            use_container_width=True,
        )
    if st.session_state.get("refined_state") is not None:
        st.download_button(
            "Download REFINED state (JSON)",
            data=json.dumps(st.session_state["refined_state"], indent=2, default=str),
            file_name="refined_state.json",
            mime="application/json",
            use_container_width=True,
        )
if st.session_state.get("refined_state") is not None:
    rs = st.session_state["refined_state"]
    with st.expander("🔧 Selection summary (what was chosen?)"):
        chosen = _get_chosen_candidate(rs)
        st.write(f"Chosen candidate: **`{chosen}`**")

        llm_decision = rs.get("llm_decision") or {}
        if llm_decision:
            st.write("LLM decision payload:")
            st.json(llm_decision)

        cand_keys = list((rs.get("optimization_result") or {}).keys())
        st.write(f"Available candidates: {cand_keys}")

        st.write("Insight status:")
        st.write("Base portfolio objective passed to refine:", rs.get("base_portfolio_objective"))
        st.write("Base portfolio metrics present:", rs.get("base_portfolio_metrics") is not None)
        st.write(
            {
                "insight_ok": rs.get("insight_ok"),
                "insight_parse_mode": rs.get("insight_parse_mode"),
                "insight_issues_n": len(rs.get("insight_issues") or []),
            }
        )


st.markdown("</div>", unsafe_allow_html=True)

# Mode C diversification chart (shown when news actions were used)
refined_state_check = st.session_state.get("refined_state") or {}
news_actions_were_applied = bool(
        (refined_state_check.get("clarification_answers") or {}).get("selected_news_actions")
    )
if news_actions_were_applied and base_sum and ref_sum:
        st.markdown("**Mode C — Diversification Focus**")
        _render_mode_c_diversification_chart(
            base_sum,
            ref_sum,
            chart_key="mode_c_diversification_chart_eval",
        )

# ---------------- Efficient Frontier ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-title">📊 Expected Return vs Risk (Efficient Frontier) — {active_label}</div>',
    unsafe_allow_html=True,
)

frontier = None
if isinstance(optimization_result, dict):
    frontier = optimization_result.get("frontier")

if (frontier is None) or (portfolio_metrics is None):
    st.info("Run base/refine to visualize the efficient frontier.")
else:
    frontier_df = pd.DataFrame(frontier)
    with st.expander("🔧 Frontier debug"):
        st.write("Active label:", active_label)
        st.write("Frontier rows:", len(frontier_df))
        st.write("Frontier columns:", list(frontier_df.columns))
        st.dataframe(frontier_df.head(20), use_container_width=True)
    y_col = (
        "realized_return"
        if "realized_return" in frontier_df.columns
        else ("return" if "return" in frontier_df.columns else frontier_df.columns[-1])
    )
    frontier_df["vol"] = pd.to_numeric(frontier_df["vol"], errors="coerce")
    frontier_df[y_col] = pd.to_numeric(frontier_df[y_col], errors="coerce")

    frontier_unique = (
        frontier_df
        .dropna(subset=["vol", y_col])
        .drop_duplicates(subset=["vol", y_col])
    )

    if len(frontier_unique) < 2:
        st.warning(
            "The efficient frontier collapsed to a single feasible point. "
            "This can happen when selected actions and constraints leave almost no allocation flexibility. "
            "For example, if only three assets remain and the max-weight cap is binding, the optimizer may be forced into one equal-weight solution."
        )
        fig_frontier = px.scatter(frontier_unique, x="vol", y=y_col)
    else:
        fig_frontier = px.line(frontier_unique, x="vol", y=y_col, markers=True)

    chosen = portfolio_metrics["candidate"]
    port = optimization_result.get(chosen, {}) if isinstance(optimization_result, dict) else {}

    x_vol = _safe_float(port.get("vol"))
    y_ret = _safe_float(port.get("return"))

    if optimized_metrics is not None and optimized_metrics:
        x_final = _safe_float(optimized_metrics.get("vol"))
        y_final = _safe_float(optimized_metrics.get("return"))
        if x_final is not None:
            x_vol = x_final
        if y_final is not None:
            y_ret = y_final

    if x_vol is not None and y_ret is not None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[x_vol],
                y=[y_ret],
                mode="markers+text",
                name=active_label,
                text=[active_label],
                textposition="top right",
                marker=dict(size=10),
                cliponaxis=False,
            )
        )

    compare_metrics, compare_label = _get_compare_state_for_charts()
    if compare_metrics is not None and compare_label:
        x_cmp = _safe_float(compare_metrics.get("vol"))
        y_cmp = _safe_float(compare_metrics.get("return"))
        if x_cmp is not None and y_cmp is not None:
            fig_frontier.add_trace(
                go.Scatter(
                    x=[x_cmp],
                    y=[y_cmp],
                    mode="markers+text",
                    name=compare_label,
                    text=[compare_label],
                    textposition="bottom right",
                    marker=dict(size=10),
                    cliponaxis=False,
                )
            )
                        

    fig_frontier.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=40, r=40, t=20, b=50),
        xaxis_title="Risk (Volatility, σ)",
        yaxis_title="Expected Return (µ)",
        height=520,
    )
    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_frontier, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Double Frontier: Base vs News-adjusted ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📈 Base vs News-adjusted Efficient Frontier</div>',
    unsafe_allow_html=True,
)

refined_state_for_news_chart = st.session_state.get("refined_state") or {}

news_was_used = bool(
    refined_state_for_news_chart.get("prob_news_trace")
    or refined_state_for_news_chart.get("news_adjustment_evaluation")
)

if news_was_used:
    _render_double_frontier_chart(
        st.session_state.get("base_state"),
        st.session_state.get("refined_state"),
        chart_key="base_vs_news_adjusted_frontier_chart",
    )
else:
    st.info("News-adjusted efficient frontier is shown only when mathematical news integration is used.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Why This Portfolio Cards ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-title"> Why this portfolio? — {active_label}</div>',
    unsafe_allow_html=True,
)

_render_why_this_portfolio_cards(graph_state)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Portfolio Story Timeline ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title"> Portfolio Story Timeline</div>',
    unsafe_allow_html=True,
)

_render_portfolio_story_timeline(
    st.session_state.get("base_state"),
    st.session_state.get("refined_state"),
)

st.markdown("</div>", unsafe_allow_html=True)

def _render_asset_detail_drilldown(
    base_state: Optional[Dict[str, Any]],
    refined_state: Optional[Dict[str, Any]],
    selected_tickers: List[str],
):
    active_state = refined_state or base_state

    if not active_state:
        st.info("Run a portfolio first to inspect asset details.")
        return

    base_w = _weights_from_state(base_state)
    ref_w = _weights_from_state(refined_state)

    selected_asset = st.selectbox(
        "Inspect asset",
        options=selected_tickers,
        key="asset_detail_selected_ticker",
    )

    base_weight = float(base_w.get(selected_asset, 0.0)) if base_w is not None else 0.0
    refined_weight = float(ref_w.get(selected_asset, 0.0)) if ref_w is not None else None
    delta_weight = None if refined_weight is None else refined_weight - base_weight

    trace = (active_state.get("prob_news_trace") or {})
    ticker_signals = trace.get("ticker_signals") or {}
    prediction_signals = trace.get("prediction_signals") or {}

    sig = ticker_signals.get(selected_asset) or {}
    pred = prediction_signals.get(selected_asset) or {}

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Base weight</div>
                <div class="metric-value">{base_weight:.1%}</div>
                <div class="metric-sub">{selected_asset}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        refined_text = "–" if refined_weight is None else f"{refined_weight:.1%}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Refined weight</div>
                <div class="metric-value">{refined_text}</div>
                <div class="metric-sub">{selected_asset}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        delta_text = "–" if delta_weight is None else f"{delta_weight:+.1%}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Weight change</div>
                <div class="metric-value">{delta_text}</div>
                <div class="metric-sub">Refined − Base</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    news_available = bool(sig or pred)

    st.markdown("**News / FinBERT signal**")

    if not news_available:
        st.info(
            "News signals are available after mathematical news integration. "
            "The base portfolio does not use FinBERT/news adjustments."
        )
        return

    df_asset_news = pd.DataFrame(
        [
            {
                "Ticker": selected_asset,
                "FinBERT sentiment": _fmt_small_num(sig.get("sentiment_score"), 3),
                "Confidence": _fmt_pct_from_decimal(_safe_float(sig.get("confidence_score"))),
                "Articles used": sig.get("raw_article_count"),
                "Predicted direction": pred.get("predicted_direction") or "–",
                "Expected return adjustment": _fmt_signed_pct_from_decimal(
                    pred.get("expected_return_adjustment"), 2
                ),
                "Risk adjustment": _fmt_signed_pct_from_decimal(
                    pred.get("risk_adjustment"), 2
                ),
            }
        ]
    )

    st.dataframe(df_asset_news, use_container_width=True, hide_index=True)

    explanation = []

    if delta_weight is not None:
        if delta_weight > 1e-6:
            explanation.append(
                f"{selected_asset}'s allocation increased by {delta_weight:+.1%} after refinement."
            )
        elif delta_weight < -1e-6:
            explanation.append(
                f"{selected_asset}'s allocation decreased by {delta_weight:+.1%} after refinement."
            )
        else:
            explanation.append(
                f"{selected_asset}'s allocation stayed almost unchanged."
            )

    exp_adj = _safe_float(pred.get("expected_return_adjustment"))
    risk_adj = _safe_float(pred.get("risk_adjustment"))

    if exp_adj is not None:
        explanation.append(
            f"The news module adjusted the expected return input by {_fmt_signed_pct_from_decimal(exp_adj, 2)}."
        )

    if risk_adj is not None:
        explanation.append(
            f"The news module adjusted the risk input by {_fmt_signed_pct_from_decimal(risk_adj, 2)}."
        )

    direction = pred.get("predicted_direction")
    confidence = _safe_float(sig.get("confidence_score"))

    if direction:
        conf_text = _fmt_pct_from_decimal(confidence)
        explanation.append(
            f"The aggregated FinBERT signal for {selected_asset} is `{direction}` with confidence {conf_text}."
        )

    if explanation:
        st.markdown("**Interpretation**")
        for line in explanation:
            st.write(f"- {line}")
# ---------------- Asset Detail Drilldown ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">🔎 Asset Detail Drilldown</div>',
    unsafe_allow_html=True,
)

_render_asset_detail_drilldown(
    st.session_state.get("base_state"),
    st.session_state.get("refined_state"),
    selected_tickers,
)


st.markdown("</div>", unsafe_allow_html=True)
# ---------------- Bottom: Weights + News + Insight + Explanation ----------------
st.markdown("")
bottom_left, bottom_right = st.columns([1.3, 1.0])

with bottom_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📉 Weights — {active_label}</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("No portfolio yet.")
    else:
        df_weights = portfolio_weights.to_frame("Weight")
        st.dataframe(df_weights.style.format("{:.3f}"), use_container_width=True, height=360)

    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if graph_state is None:
        st.info("News, insight & explanation will appear after base/refine.")
    else:
        with st.expander(" LLM decision (candidate selection)"):
            st.json(graph_state.get("llm_decision", {}))
        with st.expander(" Chat last command"):
            st.json(st.session_state.get("chat_last_command") or {})

        with st.expander(" News risk (raw JSON)"):
            st.json(graph_state.get("news_risk_json") or graph_state.get("news_signals") or {})


        with st.expander(" Debug notes (graph trace)"):
            notes = graph_state.get("debug_notes", [])
            if not notes:
                st.write("No debug notes.")
            else:
                for n in notes:
                    st.write(f"- {n}")

        

    st.markdown("</div>", unsafe_allow_html=True)