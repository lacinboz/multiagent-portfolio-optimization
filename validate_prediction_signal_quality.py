# validate_prediction_signal_quality_7d.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PRICE_DIR = Path("data/raw/daily_yahoo")
PREDICTIONS_PATH = Path("data/news_prediction/best_news_prediction_predictions.csv")
OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AS_OF_DATE = "2026-01-15"

HORIZON_TRADING_DAYS = 7

BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40


def _load_signals_as_of(
    predictions_path: Path,
    as_of_date: str,
) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    df = pd.read_csv(predictions_path)

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["news_date_dt"] = pd.to_datetime(df["news_date_dt"], errors="coerce")
    df["predicted_positive_probability"] = pd.to_numeric(
        df["predicted_positive_probability"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "ticker",
            "news_date_dt",
            "predicted_positive_probability",
        ]
    ).copy()

    as_of = pd.to_datetime(as_of_date)

    df = df[df["news_date_dt"] <= as_of].copy()

    if df.empty:
        raise ValueError(f"No prediction signals available on or before {as_of_date}")

    latest = (
        df.sort_values(["ticker", "news_date_dt"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )

    return latest.reset_index(drop=True)


def _future_return_after_news(
    ticker: str,
    news_date,
    horizon_trading_days: int,
) -> dict | None:
    path = PRICE_DIR / f"{ticker}_daily.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    news_date = pd.to_datetime(news_date)

    # first available trading day on or after news date
    future_prices = df[df["timestamp"] >= news_date].copy()

    if len(future_prices) <= horizon_trading_days:
        return None

    start_row = future_prices.iloc[0]
    end_row = future_prices.iloc[horizon_trading_days]

    start_close = float(start_row["close"])
    end_close = float(end_row["close"])

    if start_close <= 0:
        return None

    future_return = (end_close / start_close) - 1.0

    return {
        "start_price_date": start_row["timestamp"],
        "future_price_date": end_row["timestamp"],
        "start_close": start_close,
        "future_close": end_close,
        "future_return_7d": future_return,
    }


def run_signal_quality_check_7d():
    print("\n" + "=" * 80)
    print("PREDICTION SIGNAL QUALITY CHECK — 7 TRADING DAY HORIZON")
    print(f"Signal as-of date: {AS_OF_DATE}")
    print(f"Horizon: {HORIZON_TRADING_DAYS} trading days after each news signal")
    print("=" * 80)

    signals = _load_signals_as_of(
        predictions_path=PREDICTIONS_PATH,
        as_of_date=AS_OF_DATE,
    )

    rows = []

    for _, row in signals.iterrows():
        ticker = row["ticker"]
        prob = float(row["predicted_positive_probability"])
        news_date = row["news_date_dt"]

        ret_info = _future_return_after_news(
            ticker=ticker,
            news_date=news_date,
            horizon_trading_days=HORIZON_TRADING_DAYS,
        )

        if ret_info is None:
            continue

        realized_return = float(ret_info["future_return_7d"])

        if prob >= BULLISH_THRESHOLD:
            signal = "Bullish"
        elif prob <= BEARISH_THRESHOLD:
            signal = "Bearish"
        else:
            signal = "Neutral"

        predicted_direction = "Positive" if prob >= 0.5 else "Negative"
        realized_direction = "Positive" if realized_return > 0 else "Negative"

        direction_correct_05 = predicted_direction == realized_direction

        if signal == "Bullish":
            threshold_direction_correct = realized_return > 0
        elif signal == "Bearish":
            threshold_direction_correct = realized_return < 0
        else:
            threshold_direction_correct = np.nan

        rows.append(
            {
                "ticker": ticker,
                "news_date_dt": news_date,
                "predicted_positive_probability": prob,
                "prediction_confidence": abs(prob - 0.5) * 2.0,
                "signal": signal,
                "predicted_direction_05": predicted_direction,
                "start_price_date": ret_info["start_price_date"],
                "future_price_date": ret_info["future_price_date"],
                "start_close": ret_info["start_close"],
                "future_close": ret_info["future_close"],
                "future_return_7d": realized_return,
                "realized_direction": realized_direction,
                "direction_correct_05": direction_correct_05,
                "threshold_signal_correct": threshold_direction_correct,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No 7-day future returns could be computed.")

    non_neutral = df[df["signal"] != "Neutral"].copy()

    print("\nAll signals:")
    print(
        df[
            [
                "ticker",
                "news_date_dt",
                "predicted_positive_probability",
                "signal",
                "future_return_7d",
                "realized_direction",
                "direction_correct_05",
                "threshold_signal_correct",
            ]
        ]
        .sort_values("predicted_positive_probability", ascending=False)
        .to_string(index=False)
    )

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print("Total tickers:", len(df))
    print("Non-neutral signals:", len(non_neutral))
    print("Directional accuracy using 0.5 cutoff:", f"{df['direction_correct_05'].mean():.2%}")

    if not non_neutral.empty:
        valid_threshold = non_neutral.dropna(subset=["threshold_signal_correct"])
        print(
            "Directional accuracy on non-neutral threshold signals:",
            f"{valid_threshold['threshold_signal_correct'].mean():.2%}",
        )
        print(
            "Average 7d return - Bullish:",
            f"{non_neutral[non_neutral['signal'] == 'Bullish']['future_return_7d'].mean():.2%}",
        )
        print(
            "Average 7d return - Bearish:",
            f"{non_neutral[non_neutral['signal'] == 'Bearish']['future_return_7d'].mean():.2%}",
        )

    corr = df["predicted_positive_probability"].corr(df["future_return_7d"])
    print("Correlation(probability, 7d future_return):", f"{corr:.4f}")

    bucket_df = (
        df.assign(
            probability_bucket=pd.cut(
                df["predicted_positive_probability"],
                bins=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                include_lowest=True,
            )
        )
        .groupby("probability_bucket", observed=False)
        .agg(
            n=("ticker", "count"),
            avg_probability=("predicted_positive_probability", "mean"),
            avg_7d_return=("future_return_7d", "mean"),
            positive_rate_7d=("future_return_7d", lambda x: float((x > 0).mean())),
            direction_accuracy_05=("direction_correct_05", "mean"),
        )
        .reset_index()
    )

    print("\nProbability bucket check:")
    print(bucket_df.to_string(index=False))

    out_path = OUT_DIR / "prediction_signal_quality_check_7d.csv"
    bucket_path = OUT_DIR / "prediction_signal_quality_buckets_7d.csv"

    df.to_csv(out_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)

    print(f"\n[Saved] {out_path}")
    print(f"[Saved] {bucket_path}")

    return df, bucket_df


if __name__ == "__main__":
    run_signal_quality_check_7d()