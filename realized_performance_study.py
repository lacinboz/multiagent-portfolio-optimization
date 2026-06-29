# mode_b_7d_realized_performance_study.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from agents_langgraph import (
    optimization_agent_from_mu_cov,
    prediction_constrained_optimization_agent,
)
from news_constraint_integration import build_news_probability_constraints


PRICE_DIR = Path("data/raw/daily_yahoo")
PREDICTIONS_PATH = Path("data/news_prediction/best_news_prediction_predictions.csv")
OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AS_OF_DATE = "2026-01-15"
HORIZON_TRADING_DAYS = 7

DAYS_PER_YEAR = 252

RF = 0.02
W_MAX = 0.30
LAMBDA_L2 = 1e-3

BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40
DELTA = 0.02

OBJECTIVE_KEY = "maxsharpe"


def _load_signals_as_of(predictions_path: Path, as_of_date: str) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["news_date_dt"] = pd.to_datetime(df["news_date_dt"], errors="coerce")
    df["predicted_positive_probability"] = pd.to_numeric(
        df["predicted_positive_probability"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["ticker", "news_date_dt", "predicted_positive_probability"]
    ).copy()

    as_of = pd.to_datetime(as_of_date)
    df = df[df["news_date_dt"] <= as_of].copy()

    if df.empty:
        raise ValueError(f"No signals available on or before {as_of_date}")

    latest = (
        df.sort_values(["ticker", "news_date_dt"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )

    return latest.reset_index(drop=True)


def _load_mu_cov_as_of(
    tickers: List[str],
    as_of_date: str,
    min_observations: int = 60,
) -> tuple[pd.Series, pd.DataFrame]:
    frames = {}
    as_of = pd.to_datetime(as_of_date)

    for t in tickers:
        path = PRICE_DIR / f"{t}_daily.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df[["timestamp", "close"]].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df[df["timestamp"] < as_of]

        if len(df) < min_observations:
            continue

        frames[t] = df.set_index("timestamp")["close"].astype(float)

    if not frames:
        raise RuntimeError("No usable historical price data before as-of date.")

    prices = pd.DataFrame(frames).sort_index()
    prices = prices.ffill().dropna(axis=1, how="any")

    returns = prices.pct_change().dropna(how="any")

    mu = returns.mean() * DAYS_PER_YEAR
    cov = returns.cov() * DAYS_PER_YEAR

    common = [t for t in tickers if t in mu.index and t in cov.index]

    return mu.loc[common].astype(float), cov.loc[common, common].astype(float)


def _load_7d_price_window(
    tickers: List[str],
    as_of_date: str,
    horizon_trading_days: int,
) -> tuple[pd.DataFrame, str, str]:
    frames = {}
    as_of = pd.to_datetime(as_of_date)

    for t in tickers:
        path = PRICE_DIR / f"{t}_daily.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df[["timestamp", "close"]].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df[df["timestamp"] >= as_of].copy()

        if len(df) <= horizon_trading_days:
            continue

        df = df.iloc[: horizon_trading_days + 1]

        frames[t] = df.set_index("timestamp")["close"].astype(float)

    if not frames:
        raise RuntimeError("No usable 7-day realized price window.")

    prices = pd.DataFrame(frames).sort_index()
    prices = prices.ffill().dropna(axis=1, how="any")

    start_date = str(prices.index.min().date())
    end_date = str(prices.index.max().date())

    return prices, start_date, end_date


def _realized_metrics(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    rf: float,
) -> Dict[str, float]:
    common = [
        t for t in weights
        if t in prices.columns and float(weights[t]) > 1e-8
    ]

    if not common:
        return {
            "sharpe": float("nan"),
            "return": float("nan"),
            "vol": float("nan"),
            "max_dd": float("nan"),
            "period_return": float("nan"),
        }

    w = np.array([float(weights[t]) for t in common])
    w = w / w.sum()

    px = prices[common].copy()
    daily_ret = px.pct_change().dropna(how="any")

    port_ret = daily_ret.values @ w

    period_return = float(np.prod(1 + port_ret) - 1.0)
    ann_return = float(np.mean(port_ret) * DAYS_PER_YEAR)
    ann_vol = float(np.std(port_ret, ddof=1) * np.sqrt(DAYS_PER_YEAR))

    sharpe = (
        float((ann_return - rf) / ann_vol)
        if ann_vol > 0
        else float("nan")
    )

    cum = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum / running_max - 1.0))

    return {
        "sharpe": sharpe,
        "return": ann_return,
        "vol": ann_vol,
        "max_dd": max_dd,
        "period_return": period_return,
    }


def _turnover(w1: Dict[str, float], w2: Dict[str, float]) -> float:
    tickers = sorted(set(w1.keys()) | set(w2.keys()))

    return (
        sum(
            abs(float(w2.get(t, 0.0)) - float(w1.get(t, 0.0)))
            for t in tickers
        )
        / 2.0
    )


def run_mode_b_7d_study():
    print("\n" + "=" * 90)
    print("MODE B — 7 TRADING DAY NO-LOOKAHEAD REALIZED PERFORMANCE STUDY")
    print(f"Signal as-of date: {AS_OF_DATE}")
    print(f"Holding period: {HORIZON_TRADING_DAYS} trading days")
    print("=" * 90)

    latest_signals = _load_signals_as_of(
        predictions_path=PREDICTIONS_PATH,
        as_of_date=AS_OF_DATE,
    )

    selected_tickers = sorted(latest_signals["ticker"].unique().tolist())

    mu, cov = _load_mu_cov_as_of(
        tickers=selected_tickers,
        as_of_date=AS_OF_DATE,
    )

    tickers = list(mu.index)
    latest_signals = latest_signals[latest_signals["ticker"].isin(tickers)].copy()

    print(f"Universe after mu/cov alignment: {len(tickers)}")
    print(f"Signals after alignment: {len(latest_signals)}")

    baseline_res = optimization_agent_from_mu_cov(
        mu=mu,
        cov=cov,
        rf=RF,
        w_max=W_MAX,
        lambda_l2=LAMBDA_L2,
    )

    objective_key = OBJECTIVE_KEY
    if objective_key not in baseline_res:
        objective_key = "maxsharpe"

    baseline_weights = baseline_res[objective_key]["weights"]

    news_constraints = build_news_probability_constraints(
        latest_signals=latest_signals,
        baseline_weights=baseline_weights,
        bullish_threshold=BULLISH_THRESHOLD,
        bearish_threshold=BEARISH_THRESHOLD,
        delta=DELTA,
        w_max=W_MAX,
    )

    n_bull = sum(1 for c in news_constraints.values() if c.get("type") == "bullish")
    n_bear = sum(1 for c in news_constraints.values() if c.get("type") == "bearish")

    print(f"Constraints: {len(news_constraints)} total")
    print(f"  Bullish: {n_bull}")
    print(f"  Bearish: {n_bear}")
    print(f"  Tickers: {sorted(news_constraints.keys())}")

    constrained_res = prediction_constrained_optimization_agent(
        mu=mu,
        cov=cov,
        rf=RF,
        w_max=W_MAX,
        lambda_l2=LAMBDA_L2,
        news_constraints=news_constraints,
    )

    constrained_weights = constrained_res[objective_key]["weights"]

    prices, realized_start, realized_end = _load_7d_price_window(
        tickers=tickers,
        as_of_date=AS_OF_DATE,
        horizon_trading_days=HORIZON_TRADING_DAYS,
    )

    print(f"Realized window: {realized_start} → {realized_end}")
    print(f"Price matrix: {prices.shape[0]} trading days × {prices.shape[1]} tickers")

    base_m = _realized_metrics(baseline_weights, prices, RF)
    news_m = _realized_metrics(constrained_weights, prices, RF)

    turnover = _turnover(baseline_weights, constrained_weights)

    rows = [
        {
            "portfolio": "Baseline no-lookahead",
            "objective": objective_key,
            "constraints": 0,
            "n_bullish": 0,
            "n_bearish": 0,
            "turnover_pct": 0.0,
            "period_return_pct": base_m["period_return"] * 100,
            "ann_return_pct": base_m["return"] * 100,
            "ann_vol_pct": base_m["vol"] * 100,
            "sharpe": base_m["sharpe"],
            "max_dd_pct": base_m["max_dd"] * 100,
            "delta_period_return_pct": 0.0,
            "delta_sharpe": 0.0,
            "delta_vol_pct": 0.0,
            "delta_max_dd_pct": 0.0,
        },
        {
            "portfolio": "Prediction-constrained 7d",
            "objective": objective_key,
            "constraints": len(news_constraints),
            "n_bullish": n_bull,
            "n_bearish": n_bear,
            "turnover_pct": turnover * 100,
            "period_return_pct": news_m["period_return"] * 100,
            "ann_return_pct": news_m["return"] * 100,
            "ann_vol_pct": news_m["vol"] * 100,
            "sharpe": news_m["sharpe"],
            "max_dd_pct": news_m["max_dd"] * 100,
            "delta_period_return_pct": (
                news_m["period_return"] - base_m["period_return"]
            ) * 100,
            "delta_sharpe": news_m["sharpe"] - base_m["sharpe"],
            "delta_vol_pct": (news_m["vol"] - base_m["vol"]) * 100,
            "delta_max_dd_pct": (news_m["max_dd"] - base_m["max_dd"]) * 100,
        },
    ]

    print("\n" + "=" * 90)
    print("MODE B — 7D REALIZED PERFORMANCE TABLE")
    print("=" * 90)

    for r in rows:
        print(
            f"{r['portfolio']:<32} "
            f"7d Return={r['period_return_pct']:.2f}% "
            f"Ann.Return={r['ann_return_pct']:.2f}% "
            f"Vol={r['ann_vol_pct']:.2f}% "
            f"Sharpe={r['sharpe']:.4f} "
            f"MaxDD={r['max_dd_pct']:.2f}% "
            f"Turnover={r['turnover_pct']:.2f}%"
        )

    print("\nDelta constrained vs baseline:")
    print(f"  Δ7d Return: {rows[1]['delta_period_return_pct']:+.2f}%")
    print(f"  ΔSharpe:    {rows[1]['delta_sharpe']:+.4f}")
    print(f"  ΔVol:       {rows[1]['delta_vol_pct']:+.2f}%")
    print(f"  ΔMaxDD:     {rows[1]['delta_max_dd_pct']:+.2f}%")

    csv_path = OUT_DIR / "mode_b_7d_realized_performance_study.csv"
    json_path = OUT_DIR / "mode_b_7d_realized_performance_study.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "Mode B - prediction-constrained optimization",
                "signal_as_of_date": AS_OF_DATE,
                "holding_period_trading_days": HORIZON_TRADING_DAYS,
                "realized_window": {
                    "start": realized_start,
                    "end": realized_end,
                },
                "objective_key": objective_key,
                "parameters": {
                    "rf": RF,
                    "w_max": W_MAX,
                    "lambda_l2": LAMBDA_L2,
                    "bullish_threshold": BULLISH_THRESHOLD,
                    "bearish_threshold": BEARISH_THRESHOLD,
                    "delta": DELTA,
                },
                "constraints": news_constraints,
                "results": rows,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")

    return {
        "baseline": base_m,
        "constrained": news_m,
        "constraints": news_constraints,
        "rows": rows,
        "prices": prices,
    }


if __name__ == "__main__":
    run_mode_b_7d_study()