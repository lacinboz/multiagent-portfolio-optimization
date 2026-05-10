from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents_langgraph import fetch_company_news_for_ticker_window
from probabilistic_news_integration import build_article_signals_with_finbert

from dotenv import load_dotenv
load_dotenv()


OUT_DIR = Path("data/news_prediction")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7.csv"
PRICE_DIR = Path("data/raw/daily_yahoo")
BUILD_V2_FROM_V1_IF_MISSING = True
TICKERS_FOR_NEWS_DATASET = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

ALL_TICKERS_RAW_PATH_V1 = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v1.csv"

ALL_TICKERS_RAW_PATH = "data/news_prediction/news_timeseries_dataset_raw_h7_alltickers_v2_enrichedd.csv"
ALL_TICKERS_OUTPUT_TAG = "alltickers_v2_enriched"

NEWS_LOOKBACK_DAYS = 365
EXCLUDE_RECENT_DAYS = 14
WINDOW_SIZE_DAYS = 30
MAX_ITEMS_PER_WINDOW = 80

HORIZON_DAYS = 7

MIN_ROWS_FOR_REAL_USE = 2000
MIN_TEST_ROWS_FOR_REAL_USE = 300

REFRESH_RAW_DATASET = False

CACHE_TTL_S = 7 * 86400
API_SLEEP_S = 1.0

# Important for API-safe all-ticker building
ALL_TICKER_BATCH_SIZE = 5
ALL_TICKER_BATCH_INDEX = 21
MAX_API_CALLS_PER_RUN = 70
USE_TICKER_FEATURES = True

def build_v2_enriched_raw_from_v1(
    v1_path: str = ALL_TICKERS_RAW_PATH_V1,
    v2_path: str = ALL_TICKERS_RAW_PATH,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    if not Path(v1_path).exists():
        raise FileNotFoundError(f"V1 raw dataset not found: {v1_path}")

    df_v1 = pd.read_csv(v1_path)

    print("\n=== BUILD V2 FROM V1 DEBUG ===")
    print("[DEBUG] v1 path:", v1_path)
    print("[DEBUG] v2 path:", v2_path)
    print("[DEBUG] v1 rows:", len(df_v1))
    print("[DEBUG] v1 tickers:", df_v1["ticker"].nunique())

    all_parts = []
    n = len(df_v1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = df_v1.iloc[start:end].copy()

        print(f"\n[DEBUG] Processing chunk {start}:{end} / {n}")

        raw_news = chunk.to_dict(orient="records")
        tickers = sorted(chunk["ticker"].dropna().astype(str).str.upper().unique().tolist())

        article_signals = build_article_signals_with_finbert(
            raw_news=raw_news,
            tickers=tickers,
        )

        print("[DEBUG] article_signals:", len(article_signals))

        if not article_signals:
            continue

        signals_df = pd.DataFrame([asdict(x) for x in article_signals])

        signals_df["prob_positive"] = signals_df["probs"].apply(
            lambda x: x.get("positive") if isinstance(x, dict) else None
        )
        signals_df["prob_negative"] = signals_df["probs"].apply(
            lambda x: x.get("negative") if isinstance(x, dict) else None
        )
        signals_df["prob_neutral"] = signals_df["probs"].apply(
            lambda x: x.get("neutral") if isinstance(x, dict) else None
        )

        keep_signal_cols = [
            "ticker",
            "headline",
            "datetime",
            "article_sentiment",
            "article_confidence",
            "recency_weight",
            "combined_weight",
            "prob_positive",
            "prob_negative",
            "prob_neutral",
        ]

        signals_df = signals_df[keep_signal_cols].copy()

        signals_df = signals_df.drop_duplicates(
            subset=["ticker", "headline", "datetime"],
            keep="last",
        )

        merged_chunk = chunk.drop(
            columns=[
                "article_sentiment",
                "article_confidence",
                "prob_positive",
                "prob_negative",
                "prob_neutral",
                "recency_weight",
                "combined_weight",
            ],
            errors="ignore",
        ).merge(
            signals_df,
            on=["ticker", "headline", "datetime"],
            how="left",
        )

        before_drop = len(merged_chunk)

        merged_chunk = merged_chunk.dropna(subset=[
            "article_sentiment",
            "article_confidence",
            "prob_positive",
            "prob_negative",
            "prob_neutral",
            "combined_weight",
            "recency_weight",
        ])

        print("[DEBUG] merged chunk rows:", before_drop, "->", len(merged_chunk))

        all_parts.append(merged_chunk)

        partial = pd.concat(all_parts, ignore_index=True)
        partial.to_csv(v2_path, index=False)
        print("[DEBUG] partial saved rows:", len(partial))

    if not all_parts:
        print("[STOP] No enriched rows created.")
        return pd.DataFrame()

    final_df = pd.concat(all_parts, ignore_index=True)

    final_df = final_df.drop_duplicates(
        subset=["ticker", "headline", "datetime"],
        keep="last",
    )

    final_df.to_csv(v2_path, index=False)

    print("\n=== V2 ENRICHED RAW DATASET CREATED FROM V1 ===")
    print("v1 path:", v1_path)
    print("v2 path:", v2_path)
    print("rows:", len(final_df))
    print("tickers:", final_df["ticker"].nunique())

    return final_df
def load_all_available_tickers_for_news_model() -> List[str]:
    summary_path = Path("data/processed_yahoo/summary_per_asset_annual.csv")
    if not summary_path.exists():
        print(f"[WARNING] Could not find ticker summary file: {summary_path}")
        return TICKERS_FOR_NEWS_DATASET

    summary = pd.read_csv(summary_path, index_col=0)
    tickers = [str(t).upper().strip() for t in summary.index if str(t).strip()]
    return list(dict.fromkeys(tickers))


def select_ticker_batch(
    tickers: List[str],
    batch_size: int = ALL_TICKER_BATCH_SIZE,
    batch_index: int = ALL_TICKER_BATCH_INDEX,
) -> List[str]:
    start = batch_index * batch_size
    end = start + batch_size
    return tickers[start:end]


# ============================================================
# HELPERS
# ============================================================

def _to_datetime_utc(x: Any) -> Optional[datetime]:
    if x is None:
        return None

    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)

    try:
        if isinstance(x, (int, float)):
            return datetime.fromtimestamp(float(x), tz=timezone.utc)
    except Exception:
        pass

    try:
        return pd.to_datetime(x, utc=True).to_pydatetime()
    except Exception:
        return None


def _safe_float_raw(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _make_date_windows(
    *,
    lookback_days: int,
    exclude_recent_days: int,
    window_size_days: int = WINDOW_SIZE_DAYS,
) -> List[Dict[str, str]]:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=exclude_recent_days)
    start_date = end_date - timedelta(days=lookback_days)

    windows = []
    cur = start_date

    while cur <= end_date:
        nxt = min(cur + timedelta(days=window_size_days - 1), end_date)
        windows.append({"from": cur.isoformat(), "to": nxt.isoformat()})
        cur = nxt + timedelta(days=1)

    return windows


def _dedup_raw_news_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []

    for it in items or []:
        url = str(it.get("url") or "").strip()
        headline = str(it.get("headline") or "").strip()
        source = str(it.get("source") or "").strip()
        dt = str(it.get("datetime") or "").strip()
        ticker = str(it.get("ticker") or "").strip()

        key = url if url else f"{ticker}|{headline}|{source}|{dt}"

        if key in seen:
            continue

        seen.add(key)
        out.append(it)

    return out


def _merge_with_existing_raw_dataset(
    new_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if new_df.empty:
        if output_path.exists():
            return pd.read_csv(output_path)
        return new_df

    if output_path.exists():
        old_df = pd.read_csv(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if "url" in combined.columns:
        has_url = combined["url"].notna() & (combined["url"].astype(str).str.strip() != "")

        with_url = combined[has_url].drop_duplicates(
            subset=["ticker", "url"],
            keep="last",
        )

        without_url = combined[~has_url].drop_duplicates(
            subset=["ticker", "headline", "datetime"],
            keep="last",
        )

        combined = pd.concat([with_url, without_url], ignore_index=True)
    else:
        combined = combined.drop_duplicates(
            subset=["ticker", "headline", "datetime"],
            keep="last",
        )

    if "news_date" in combined.columns:
        combined = combined.sort_values(["ticker", "news_date"]).reset_index(drop=True)
    else:
        combined = combined.sort_values(["ticker", "datetime"]).reset_index(drop=True)

    return combined


# ============================================================
# FETCH NEWS
# ============================================================

def fetch_windowed_historical_news_for_tickers(
    *,
    tickers: List[str],
    lookback_days: int,
    exclude_recent_days: int,
    max_items_per_window: int = MAX_ITEMS_PER_WINDOW,
    cache_ttl_s: int = CACHE_TTL_S,
    sleep_s: float = API_SLEEP_S,
    max_api_calls_per_run: int = MAX_API_CALLS_PER_RUN,
) -> Dict[str, Any]:
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))

    windows = _make_date_windows(
        lookback_days=lookback_days,
        exclude_recent_days=exclude_recent_days,
        window_size_days=WINDOW_SIZE_DAYS,
    )

    flat_items = []
    errors = {}
    api_calls_done = 0
    stopped_due_to_call_budget = False

    print("\n[WINDOWED FETCH]")
    print("tickers:", tickers)
    print("n_tickers:", len(tickers))
    print("windows:", len(windows))
    print("estimated_api_calls_for_batch:", len(tickers) * len(windows))
    print("max_api_calls_per_run:", max_api_calls_per_run)
    print("max_items_per_window:", max_items_per_window)
    print("sleep_s:", sleep_s)

    for ticker in tickers:
        for w in windows:
            if api_calls_done >= max_api_calls_per_run:
                stopped_due_to_call_budget = True
                print("[STOP FETCH] Max API-call budget reached for this run.")
                break

            try:
                api_calls_done += 1

                items = fetch_company_news_for_ticker_window(
                    ticker=ticker,
                    from_date=w["from"],
                    to_date=w["to"],
                    max_items=max_items_per_window,
                    cache_ttl_s=cache_ttl_s,
                    sleep_s=sleep_s,
                )

                for it in items or []:
                    cp = dict(it)
                    cp["ticker"] = ticker
                    flat_items.append(cp)

                print(
                    f"[WINDOW] {ticker} {w['from']} -> {w['to']} "
                    f"items={len(items)} call={api_calls_done}/{max_api_calls_per_run}"
                )

            except Exception as e:
                key = f"{ticker}_{w['from']}_{w['to']}"
                errors[key] = str(e)
                print(f"[WINDOW ERROR] {key}: {e}")

                if "429" in str(e) or "API limit reached" in str(e):
                    print("[STOP FETCH] Finnhub API limit reached.")
                    before = len(flat_items)
                    flat_items = _dedup_raw_news_items(flat_items)
                    after = len(flat_items)

                    return {
                        "flat_items": flat_items,
                        "stats": {
                            "tickers": len(tickers),
                            "windows": len(windows),
                            "estimated_api_calls_for_batch": len(tickers) * len(windows),
                            "api_calls_done": api_calls_done,
                            "total_items_before_dedup": before,
                            "total_items": after,
                            "duplicates_removed": before - after,
                            "errors": errors,
                            "stopped_early_due_to_api_limit": True,
                            "stopped_due_to_call_budget": False,
                        },
                    }

        if stopped_due_to_call_budget:
            break

    before = len(flat_items)
    flat_items = _dedup_raw_news_items(flat_items)
    after = len(flat_items)

    return {
        "flat_items": flat_items,
        "stats": {
            "tickers": len(tickers),
            "windows": len(windows),
            "estimated_api_calls_for_batch": len(tickers) * len(windows),
            "api_calls_done": api_calls_done,
            "total_items_before_dedup": before,
            "total_items": after,
            "duplicates_removed": before - after,
            "errors": errors,
            "stopped_early_due_to_api_limit": False,
            "stopped_due_to_call_budget": stopped_due_to_call_budget,
        },
    }


# ============================================================
# PRICE FEATURES
# ============================================================

def _load_price_file(ticker: str, price_dir: Path = PRICE_DIR) -> pd.DataFrame:
    path = price_dir / f"{ticker}_daily.csv"

    if not path.exists():
        print(f"[PRICE] Missing price file for {ticker}: {path}")
        return pd.DataFrame()

    prices = pd.read_csv(path)

    if "timestamp" not in prices.columns or "close" not in prices.columns:
        print(f"[PRICE] Invalid columns for {ticker}: {path}")
        return pd.DataFrame()

    prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "close"])
    prices = prices.sort_values("timestamp").reset_index(drop=True)

    return prices


def _build_price_features_for_article(
    *,
    ticker: str,
    news_dt: datetime,
    horizon_days: int = HORIZON_DAYS,
    price_dir: Path = PRICE_DIR,
) -> Optional[Dict[str, Any]]:
    prices = _load_price_file(ticker, price_dir=price_dir)

    if prices.empty:
        return None

    news_date = pd.Timestamp(news_dt.date())
    candidates = prices[prices["timestamp"] > news_date]

    if candidates.empty:
        return None

    start_idx = int(candidates.index[0])
    future_idx = start_idx + int(horizon_days)

    if future_idx >= len(prices):
        return None

    if start_idx < 21:
        return None

    close = prices["close"].astype(float)

    start_close = float(close.iloc[start_idx])
    future_close = float(close.iloc[future_idx])

    if start_close <= 0 or future_close <= 0:
        return None

    def past_return(days: int) -> Optional[float]:
        prev_idx = start_idx - days
        if prev_idx < 0:
            return None
        prev_close = float(close.iloc[prev_idx])
        if prev_close <= 0:
            return None
        return float((start_close / prev_close) - 1.0)

    def past_volatility(days: int) -> Optional[float]:
        prev_idx = start_idx - days
        if prev_idx < 0:
            return None
        window = close.iloc[prev_idx:start_idx + 1].pct_change().dropna()
        if window.empty:
            return None
        return float(window.std())

    future_return = float((future_close / start_close) - 1.0)

    return {
        "news_date": str(news_date.date()),
        "start_price_date": str(prices.loc[start_idx, "timestamp"].date()),
        "future_price_date": str(prices.loc[future_idx, "timestamp"].date()),
        "start_close": start_close,
        "future_close": future_close,
        "future_return": future_return,
        "past_5d_return": past_return(5),
        "past_10d_return": past_return(10),
        "past_20d_return": past_return(20),
        "past_5d_volatility": past_volatility(5),
        "past_10d_volatility": past_volatility(10),
        "past_20d_volatility": past_volatility(20),
        "past_5d_momentum": past_return(5),
        "past_10d_momentum": past_return(10),
        "past_20d_momentum": past_return(20),
    }


# ============================================================
# RAW DATASET REFRESH
# ============================================================
def print_missing_tickers_status(raw_path: str = ALL_TICKERS_RAW_PATH) -> None:
    all_tickers = load_all_available_tickers_for_news_model()

    if not Path(raw_path).exists():
        print("[MISSING CHECK] Raw dataset does not exist yet.")
        print("all available tickers:", len(all_tickers))
        print("missing tickers:", all_tickers)
        return

    raw_df = pd.read_csv(raw_path)

    collected_tickers = (
        raw_df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )

    missing_tickers = sorted(set(all_tickers) - set(collected_tickers))
    extra_tickers = sorted(set(collected_tickers) - set(all_tickers))

    print("\n==============================")
    print("TICKER COLLECTION STATUS")
    print("==============================")
    print("all available tickers:", len(all_tickers))
    print("collected tickers in raw dataset:", len(collected_tickers))
    print("missing tickers:", missing_tickers)
    print("n_missing:", len(missing_tickers))
    print("extra tickers:", extra_tickers)
    print("n_extra:", len(extra_tickers))

def refresh_raw_news_timeseries_dataset(
    *,
    tickers: List[str] = TICKERS_FOR_NEWS_DATASET,
    output_path: str = RAW_PATH,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    exclude_recent_days: int = EXCLUDE_RECENT_DAYS,
    max_items_per_window: int = MAX_ITEMS_PER_WINDOW,
    horizon_days: int = HORIZON_DAYS,
    cache_ttl_s: int = CACHE_TTL_S,
    sleep_s: float = API_SLEEP_S,
    append_to_existing: bool = True,
    max_api_calls_per_run: int = MAX_API_CALLS_PER_RUN,
) -> pd.DataFrame:
    print("\n==============================")
    print("REFRESH RAW NEWS-TIMESERIES DATASET")
    print("==============================")
    print("tickers:", tickers)
    print("lookback_days:", lookback_days)
    print("exclude_recent_days:", exclude_recent_days)
    print("window_size_days:", WINDOW_SIZE_DAYS)
    print("max_items_per_window:", max_items_per_window)
    print("max_api_calls_per_run:", max_api_calls_per_run)
    print("append_to_existing:", append_to_existing)

    fetched = fetch_windowed_historical_news_for_tickers(
        tickers=tickers,
        lookback_days=lookback_days,
        exclude_recent_days=exclude_recent_days,
        max_items_per_window=max_items_per_window,
        cache_ttl_s=cache_ttl_s,
        sleep_s=sleep_s,
        max_api_calls_per_run=max_api_calls_per_run,
    )

    raw_news = (fetched or {}).get("flat_items") or []

    print("\n[FETCH STATS]")
    print((fetched or {}).get("stats"))
    print("raw_news_count:", len(raw_news))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_news:
        print("[INFO] No new raw news fetched.")
        if output_path.exists():
            print("[INFO] Returning existing raw dataset.")
            return pd.read_csv(output_path)
        return pd.DataFrame()

    article_signals = build_article_signals_with_finbert(
        raw_news=raw_news,
        tickers=tickers,
    )

    article_signal_dicts = [asdict(x) for x in article_signals]
    print("article_signal_count:", len(article_signal_dicts))

    rows = []

    for art in article_signal_dicts:
        ticker = str(art.get("ticker") or "").upper().strip()
        news_dt = _to_datetime_utc(art.get("datetime"))

        if not ticker or news_dt is None:
            continue

        price_features = _build_price_features_for_article(
            ticker=ticker,
            news_dt=news_dt,
            horizon_days=horizon_days,
            price_dir=PRICE_DIR,
        )

        if price_features is None:
            continue

        headline = str(art.get("headline") or "")
        summary = str(art.get("summary") or "")
        probs = art.get("probs") or {}

        article_sentiment = _safe_float_raw(art.get("article_sentiment"))
        article_confidence = _safe_float_raw(art.get("article_confidence"))
        prob_positive = _safe_float_raw(probs.get("positive"))
        prob_negative = _safe_float_raw(probs.get("negative"))
        prob_neutral = _safe_float_raw(probs.get("neutral"))
        combined_weight = _safe_float_raw(art.get("combined_weight"))
        recency_weight = _safe_float_raw(art.get("recency_weight"))

        row = {
            "ticker": ticker,
            "headline": headline,
            "summary": summary,
            "source": art.get("source"),
            "datetime": art.get("datetime"),
            "url": art.get("url"),

            "headline_length": len(headline),
            "summary_length": len(summary),

            "article_sentiment": article_sentiment,
            "article_confidence": article_confidence,
            "prob_positive": prob_positive,
            "prob_negative": prob_negative,
            "prob_neutral": prob_neutral,

            "combined_weight": combined_weight,
            "recency_weight": recency_weight,
        }

        row.update(price_features)
        rows.append(row)

    new_df = pd.DataFrame(rows)

    if new_df.empty:
        print("[INFO] No usable new rows after matching news with price data.")
        if output_path.exists():
            return pd.read_csv(output_path)
        return pd.DataFrame()

    required_cols = [
        "ticker",
        "news_date",
        "future_return",
        "article_sentiment",
        "article_confidence",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
        "combined_weight",
        "recency_weight",
        "past_5d_return",
        "past_10d_return",
        "past_20d_return",
        "past_5d_volatility",
        "past_10d_volatility",
        "past_20d_volatility",
    ]

    new_df = new_df.dropna(subset=required_cols)
    new_df = new_df.sort_values(["ticker", "news_date"]).reset_index(drop=True)

    if append_to_existing:
        final_df = _merge_with_existing_raw_dataset(new_df, output_path)
    else:
        final_df = new_df

    final_df.to_csv(output_path, index=False)

    print("\n=== RAW DATASET SAVED ===")
    print("path:", output_path)
    print("new usable rows this run:", len(new_df))
    print("total rows after merge:", len(final_df))
    print("unique tickers:", final_df["ticker"].nunique())
    print("unique news dates:", final_df["news_date"].nunique())
    print("date range:", final_df["news_date"].min(), "->", final_df["news_date"].max())
    print("ticker counts:", final_df["ticker"].value_counts().to_dict())

    return final_df


# ============================================================
# DATASET BUILDING
# ============================================================
def build_ticker_date_prediction_dataset_v2(
    raw_path: str = ALL_TICKERS_RAW_PATH,
    min_abs_return_for_signal: Optional[float] = 0.01,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    if df.empty:
        print("[STOP] Raw dataset is empty.")
        return pd.DataFrame()

    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=[
        "news_date_dt",
        "ticker",
        "future_return",
        "article_sentiment",
        "article_confidence",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
        "combined_weight",
        "past_5d_return",
        "past_20d_return",
        "past_20d_volatility",
    ]).copy()

    if min_abs_return_for_signal is not None:
        df = df[df["future_return"].abs() >= float(min_abs_return_for_signal)].copy()

    if df.empty:
        print("[STOP] No rows left after filtering.")
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    df["is_positive_article"] = (df["prob_positive"] > df["prob_negative"]).astype(int)
    df["is_negative_article"] = (df["prob_negative"] > df["prob_positive"]).astype(int)
    df["sentiment_confidence"] = df["article_sentiment"] * df["article_confidence"]

    def weighted_mean(x, value_col, weight_col="combined_weight"):
        values = x[value_col].astype(float)
        weights = x[weight_col].astype(float)
        if weights.sum() <= 0:
            return float(values.mean())
        return float(np.average(values, weights=weights))

    grouped_rows = []

    for (ticker, news_date, news_date_dt), g in df.groupby(["ticker", "news_date", "news_date_dt"]):
        grouped_rows.append({
            "ticker": ticker,
            "news_date": news_date,
            "news_date_dt": news_date_dt,

            "article_count": int(len(g)),
            "weighted_sentiment": weighted_mean(g, "article_sentiment", weight_col="article_confidence"),
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
            shifted_sentiment = g["weighted_sentiment"].shift(1)
            shifted_confidence = g["mean_confidence"].shift(1)

            g[f"sentiment_flow_{w}d"] = shifted_sentiment.rolling(w, min_periods=1).mean()
            g[f"confidence_flow_{w}d"] = shifted_confidence.rolling(w, min_periods=1).mean()

        flow_parts.append(g)

    out = pd.concat(flow_parts, ignore_index=True)
    out = out.dropna().reset_index(drop=True)

    out["target_direction"] = (out["future_return"] > 0).astype(int)

    print("\n=== Ticker-date prediction dataset v2 ===")
    print("rows:", len(out))
    print("class counts:", out["target_direction"].value_counts().to_dict())
    print("date range:", out["news_date_dt"].min(), "->", out["news_date_dt"].max())
    print("tickers:", out["ticker"].nunique())

    return out
def build_article_level_dataset_from_saved_raw(
    raw_path: str = RAW_PATH,
    min_abs_return_for_signal: Optional[float] = None,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    if df.empty:
        print("[STOP] Raw dataset is empty.")
        return pd.DataFrame()

    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=["news_date_dt", "ticker", "future_return"]).copy()

    if min_abs_return_for_signal is not None:
        df = df[df["future_return"].abs() >= float(min_abs_return_for_signal)].copy()

    if df.empty:
        print("[STOP] No rows left after return threshold filtering.")
        return pd.DataFrame()

    df["target_direction"] = (df["future_return"] > 0).astype(int)
    df = df.sort_values(["ticker", "news_date_dt", "datetime"]).reset_index(drop=True)

    flow_parts = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("news_date_dt").copy()

        for w in [5, 20, 50]:
            g[f"article_sentiment_flow_last_{w}_articles_mean"] = (
                g["article_sentiment"].rolling(w, min_periods=1).mean()
            )
            g[f"article_sentiment_flow_last_{w}_articles_std"] = (
                g["article_sentiment"].rolling(w, min_periods=1).std().fillna(0.0)
            )
            g[f"confidence_flow_last_{w}_articles_mean"] = (
                g["article_confidence"].rolling(w, min_periods=1).mean()
            )
            g[f"positive_prob_flow_last_{w}_articles_mean"] = (
                g["prob_positive"].rolling(w, min_periods=1).mean()
            )
            g[f"negative_prob_flow_last_{w}_articles_mean"] = (
                g["prob_negative"].rolling(w, min_periods=1).mean()
            )

        flow_parts.append(g)

    out = pd.concat(flow_parts, ignore_index=True)
    out = out.dropna().reset_index(drop=True)

    print("\n=== Article-level dataset summary ===")
    print("rows:", len(out))
    print("class counts:", out["target_direction"].value_counts().to_dict())
    print("date range:", out["news_date_dt"].min(), "->", out["news_date_dt"].max())
    print("unique dates:", out["news_date_dt"].nunique())
    print("tickers:", out["ticker"].value_counts().to_dict())

    if len(out) < MIN_ROWS_FOR_REAL_USE:
        print(
            f"\n[WARNING] Dataset has only {len(out)} rows. "
            f"For stronger thesis-level predictive use, target at least {MIN_ROWS_FOR_REAL_USE}+ rows."
        )

    return out

def get_feature_cols() -> List[str]:
    return [
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
def prepare_model_frame(
    dataset: pd.DataFrame,
    *,
    use_ticker_features: bool = True,
) -> tuple[pd.DataFrame, List[str]]:
    base_feature_cols = get_feature_cols()

    df = dataset.dropna(
        subset=base_feature_cols + ["target_direction", "news_date_dt", "ticker"]
    ).copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["news_date_dt", "ticker"]).reset_index(drop=True)

    feature_cols = base_feature_cols.copy()

    if use_ticker_features:
        ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=float)
        df = pd.concat([df, ticker_dummies], axis=1)
        feature_cols += list(ticker_dummies.columns)

    return df, feature_cols
def build_daily_ticker_level_dataset_from_saved_raw(
    raw_path: str = ALL_TICKERS_RAW_PATH,
    min_abs_return_for_signal: Optional[float] = 0.01,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    if df.empty:
        print("[STOP] Raw dataset is empty.")
        return pd.DataFrame()

    df["news_date_dt"] = pd.to_datetime(df["news_date"], errors="coerce")
    df = df.dropna(subset=["news_date_dt", "ticker", "future_return"]).copy()

    if min_abs_return_for_signal is not None:
        df = df[df["future_return"].abs() >= float(min_abs_return_for_signal)].copy()

    if df.empty:
        print("[STOP] No rows left after return threshold filtering.")
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    agg = df.groupby(["ticker", "news_date", "news_date_dt"]).agg(
        article_count=("headline", "count"),

        article_sentiment=("article_sentiment", "mean"),
        article_sentiment_std=("article_sentiment", "std"),
        article_confidence=("article_confidence", "mean"),

        prob_positive=("prob_positive", "mean"),
        prob_negative=("prob_negative", "mean"),
        prob_neutral=("prob_neutral", "mean"),

        headline_length=("headline_length", "mean"),
        summary_length=("summary_length", "mean"),

        past_5d_return=("past_5d_return", "mean"),
        past_10d_return=("past_10d_return", "mean"),
        past_20d_return=("past_20d_return", "mean"),

        past_5d_volatility=("past_5d_volatility", "mean"),
        past_10d_volatility=("past_10d_volatility", "mean"),
        past_20d_volatility=("past_20d_volatility", "mean"),

        past_5d_momentum=("past_5d_momentum", "mean"),
        past_10d_momentum=("past_10d_momentum", "mean"),
        past_20d_momentum=("past_20d_momentum", "mean"),

        future_return=("future_return", "mean"),
    ).reset_index()

    agg["article_sentiment_std"] = agg["article_sentiment_std"].fillna(0.0)
    agg["target_direction"] = (agg["future_return"] > 0).astype(int)

    agg = agg.sort_values(["ticker", "news_date_dt"]).reset_index(drop=True)

    flow_parts = []

    for ticker, g in agg.groupby("ticker"):
        g = g.sort_values("news_date_dt").copy()

        for w in [5, 20, 50]:
            g[f"article_sentiment_flow_last_{w}_articles_mean"] = (
                g["article_sentiment"].rolling(w, min_periods=1).mean()
            )
            g[f"article_sentiment_flow_last_{w}_articles_std"] = (
                g["article_sentiment"].rolling(w, min_periods=1).std().fillna(0.0)
            )
            g[f"confidence_flow_last_{w}_articles_mean"] = (
                g["article_confidence"].rolling(w, min_periods=1).mean()
            )
            g[f"positive_prob_flow_last_{w}_articles_mean"] = (
                g["prob_positive"].rolling(w, min_periods=1).mean()
            )
            g[f"negative_prob_flow_last_{w}_articles_mean"] = (
                g["prob_negative"].rolling(w, min_periods=1).mean()
            )

        flow_parts.append(g)

    out = pd.concat(flow_parts, ignore_index=True)
    out = out.dropna().reset_index(drop=True)

    print("\n=== Daily ticker-level dataset summary ===")
    print("rows:", len(out))
    print("class counts:", out["target_direction"].value_counts().to_dict())
    print("date range:", out["news_date_dt"].min(), "->", out["news_date_dt"].max())
    print("tickers:", out["ticker"].nunique())

    return out
# ============================================================
# MODEL
# ============================================================

def extract_model_explanation(
    model: Any,
    feature_cols: List[str],
    model_type: str,
) -> Dict[str, Any]:
    try:
        if model_type == "random_forest":
            items = [
                {"feature": f, "importance": float(v)}
                for f, v in zip(feature_cols, model.feature_importances_)
            ]
            items = sorted(items, key=lambda x: abs(x["importance"]), reverse=True)
            return {"type": "feature_importance", "items": items}

        clf = model.named_steps["clf"]
        items = [
            {"feature": f, "coefficient": float(v)}
            for f, v in zip(feature_cols, clf.coef_[0])
        ]
        items = sorted(items, key=lambda x: abs(x["coefficient"]), reverse=True)
        return {"type": "logistic_coefficients", "items": items}

    except Exception as e:
        return {"type": "unavailable", "error": str(e)}


def train_news_flow_predictor(
    dataset: pd.DataFrame,
    *,
    model_type: str = "random_forest",
    test_size: float = 0.30,
    random_state: int = 42,
    use_ticker_features: bool = USE_TICKER_FEATURES,
) -> Dict[str, Any]:
    if dataset.empty:
        return {"ok": False, "reason": "Dataset is empty."}

    df, feature_cols = prepare_model_frame(
        dataset,
        use_ticker_features=use_ticker_features,
    )

    if df.empty:
        return {"ok": False, "reason": "No usable rows after feature preparation."}

    y = df["target_direction"].astype(int)

    if y.nunique() < 2:
        return {
            "ok": False,
            "reason": "Target has only one class.",
            "class_counts": y.value_counts().to_dict(),
        }

    split_idx = int(len(df) * (1.0 - test_size))

    if split_idx <= 0 or split_idx >= len(df):
        return {
            "ok": False,
            "reason": "Dataset too small for time-based split.",
            "rows": int(len(df)),
        }

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    train_keys = set(zip(train_df["ticker"], train_df["news_date_dt"].astype(str)))
    test_keys = set(zip(test_df["ticker"], test_df["news_date_dt"].astype(str)))
    overlap_keys = train_keys.intersection(test_keys)

    train_dates = set(train_df["news_date_dt"].astype(str))
    test_dates = set(test_df["news_date_dt"].astype(str))
    overlap_dates = train_dates.intersection(test_dates)

    print("\n[TRAIN/TEST LEAKAGE CHECK]")
    print("overlap ticker-date keys:", len(overlap_keys))
    print("overlap dates:", len(overlap_dates))
    print("sample overlap ticker-date:", list(overlap_keys)[:10])
    print("sample overlap dates:", sorted(list(overlap_dates))[:10])

    y_train = train_df["target_direction"].astype(int)
    y_test = test_df["target_direction"].astype(int)

    if y_train.nunique() < 2:
        return {
            "ok": False,
            "reason": "Training set has only one class.",
            "class_counts_train": y_train.value_counts().to_dict(),
            "class_counts_test": y_test.value_counts().to_dict(),
        }

    X_train = train_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=0.3,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    majority_class = int(y_train.mode().iloc[0])
    baseline_pred = np.full(len(y_test), majority_class, dtype=int)

    roc_auc = None
    inverse_roc_auc = None

    if y_test.nunique() == 2:
        roc_auc = float(roc_auc_score(y_test, proba))
        inverse_roc_auc = float(roc_auc_score(y_test, 1.0 - proba))

    accuracy = float(accuracy_score(y_test, pred))
    baseline_accuracy = float(accuracy_score(y_test, baseline_pred))

    balanced_acc = float(balanced_accuracy_score(y_test, pred))
    baseline_balanced_acc = float(balanced_accuracy_score(y_test, baseline_pred))

    model_minus_baseline_accuracy = float(accuracy - baseline_accuracy)
    model_minus_baseline_balanced_accuracy = float(
        balanced_acc - baseline_balanced_acc
    )

    use_as_portfolio_signal = bool(
        roc_auc is not None
        and roc_auc >= 0.55
        and model_minus_baseline_balanced_accuracy > 0.0
    )

    metrics = {
        "split_type": "time_based",
        "model_type": model_type,
        "use_ticker_features": use_ticker_features,

        "train_date_range": [
            str(train_df["news_date_dt"].min().date()),
            str(train_df["news_date_dt"].max().date()),
        ],
        "test_date_range": [
            str(test_df["news_date_dt"].min().date()),
            str(test_df["news_date_dt"].max().date()),
        ],

        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),

        "n_train_tickers": int(train_df["ticker"].nunique()),
        "n_test_tickers": int(test_df["ticker"].nunique()),
        "train_tickers": sorted(train_df["ticker"].unique().tolist()),
        "test_tickers": sorted(test_df["ticker"].unique().tolist()),

        "class_counts_total": y.value_counts().to_dict(),
        "class_counts_train": y_train.value_counts().to_dict(),
        "class_counts_test": y_test.value_counts().to_dict(),

        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "majority_baseline_class": majority_class,
        "majority_baseline_accuracy": baseline_accuracy,
        "majority_baseline_balanced_accuracy": baseline_balanced_acc,
        "model_minus_baseline_accuracy": model_minus_baseline_accuracy,
        "model_minus_baseline_balanced_accuracy": model_minus_baseline_balanced_accuracy,

        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),

        "roc_auc": roc_auc,
        "inverse_roc_auc": inverse_roc_auc,

        "use_as_portfolio_signal": use_as_portfolio_signal,

        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "baseline_confusion_matrix": confusion_matrix(y_test, baseline_pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }

    if len(df) < MIN_ROWS_FOR_REAL_USE or len(test_df) < MIN_TEST_ROWS_FOR_REAL_USE:
        metrics["reliability_warning"] = (
            "Dataset is too small for reliable financial prediction."
        )
    elif not use_as_portfolio_signal:
        metrics["reliability_warning"] = (
            "Model produces predictions, but validation performance is still weak. "
            "Use probabilities with low confidence in the portfolio layer."
        )
    else:
        metrics["reliability_warning"] = None

    model_explanation = extract_model_explanation(
        model=model,
        feature_cols=feature_cols,
        model_type=model_type,
    )

    predictions = test_df.copy()
    predictions["predicted_direction"] = pred
    predictions["predicted_positive_probability"] = proba
    predictions["prediction_confidence"] = np.abs(proba - 0.5) * 2.0
    predictions["baseline_prediction"] = baseline_pred
    predictions["correct"] = (
        predictions["target_direction"].astype(int)
        == predictions["predicted_direction"].astype(int)
    )
    predictions["baseline_correct"] = (
        predictions["target_direction"].astype(int)
        == predictions["baseline_prediction"].astype(int)
    )

    return {
        "ok": True,
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_explanation": model_explanation,
        "predictions": predictions.reset_index(drop=True),
    }
# ============================================================
# RUN EXPERIMENTS
# ============================================================
def get_missing_tickers(raw_path: str = ALL_TICKERS_RAW_PATH) -> List[str]:
    all_tickers = load_all_available_tickers_for_news_model()

    if not Path(raw_path).exists():
        return all_tickers

    raw_df = pd.read_csv(raw_path)

    collected_tickers = (
        raw_df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(set(all_tickers) - set(collected_tickers))
def run_saved_news_flow_experiment(
    *,
    raw_path: str = RAW_PATH,
    min_abs_return_for_signal: Optional[float] = None,
    model_type: str = "random_forest",
    save_outputs: bool = True,
) -> Dict[str, Any]:
    print("\n==============================")
    print("SAVED NEWS-FLOW PREDICTION EXPERIMENT")
    print("==============================")

    dataset = build_article_level_dataset_from_saved_raw(
        raw_path=raw_path,
        min_abs_return_for_signal=min_abs_return_for_signal,
    )

    result = train_news_flow_predictor(
        dataset=dataset,
        model_type=model_type,
    )

    if save_outputs:
        suffix = "all" if min_abs_return_for_signal is None else str(min_abs_return_for_signal).replace(".", "p")

        dataset_path = OUT_DIR / f"article_level_dataset_thr{suffix}_{model_type}.csv"
        pred_path = OUT_DIR / f"article_level_predictions_thr{suffix}_{model_type}.csv"
        metrics_path = OUT_DIR / f"article_level_metrics_thr{suffix}_{model_type}.json"
        explanation_path = OUT_DIR / f"article_level_model_explanation_thr{suffix}_{model_type}.json"
        model_path = OUT_DIR / f"article_level_model_thr{suffix}_{model_type}.joblib"

        dataset.to_csv(dataset_path, index=False)
        print("\nSaved:", dataset_path)

        if result.get("ok"):
            result["predictions"].to_csv(pred_path, index=False)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

            with open(explanation_path, "w", encoding="utf-8") as f:
                json.dump(result["model_explanation"], f, indent=2, ensure_ascii=False)

            joblib.dump(
                {
                    "model": result["model"],
                    "feature_cols": result["feature_cols"],
                    "metrics": result["metrics"],
                    "threshold": min_abs_return_for_signal,
                    "model_type": model_type,
                    "horizon_days": HORIZON_DAYS,
                    "use_ticker_features": USE_TICKER_FEATURES,
                    "use_as_portfolio_signal": result["metrics"].get("use_as_portfolio_signal"),
                },
                model_path,
            )

            print("Saved:", model_path)
            print("Saved:", pred_path)
            print("Saved:", metrics_path)
            print("Saved:", explanation_path)
            

    return {
        "ok": bool(result.get("ok")),
        "rows": int(len(dataset)),
        "dataset": dataset,
        **result,
    }

def audit_news_prediction_dataset_for_leakage(
    raw_path: str = ALL_TICKERS_RAW_PATH,
    dataset_path: Optional[str] = None,
) -> None:
    print("\n==============================")
    print("DATA LEAKAGE AUDIT")
    print("==============================")

    raw = pd.read_csv(raw_path)
    raw["news_date"] = pd.to_datetime(raw["news_date"], errors="coerce")
    raw["start_price_date"] = pd.to_datetime(raw["start_price_date"], errors="coerce")
    raw["future_price_date"] = pd.to_datetime(raw["future_price_date"], errors="coerce")
    raw["datetime_parsed"] = pd.to_datetime(raw["datetime"], errors="coerce", unit="s", utc=True)

    print("\n[RAW DATA DATE CHECK]")
    print("raw rows:", len(raw))
    print("news_date range:", raw["news_date"].min(), "->", raw["news_date"].max())
    print("datetime range:", raw["datetime_parsed"].min(), "->", raw["datetime_parsed"].max())
    print("start_price_date range:", raw["start_price_date"].min(), "->", raw["start_price_date"].max())
    print("future_price_date range:", raw["future_price_date"].min(), "->", raw["future_price_date"].max())

    bad_start = raw[raw["start_price_date"] <= raw["news_date"]]
    bad_future = raw[raw["future_price_date"] <= raw["start_price_date"]]
    bad_horizon = raw[raw["future_price_date"] <= raw["news_date"]]

    print("\n[LEAKAGE RULE CHECKS]")
    print("Rows where start_price_date <= news_date:", len(bad_start))
    print("Rows where future_price_date <= start_price_date:", len(bad_future))
    print("Rows where future_price_date <= news_date:", len(bad_horizon))

    print("\n[2026 NEWS CHECK]")
    news_2026 = raw[raw["news_date"].dt.year == 2026].copy()
    print("2026 raw rows:", len(news_2026))
    if not news_2026.empty:
        print("2026 news_date range:", news_2026["news_date"].min(), "->", news_2026["news_date"].max())
        print("2026 tickers:", news_2026["ticker"].nunique())
        print(
            news_2026[
                ["ticker", "news_date", "start_price_date", "future_price_date", "future_return"]
            ]
            .sort_values(["news_date", "ticker"])
            .head(20)
        )

    suspicious_cols = [
        c for c in raw.columns
        if any(k in c.lower() for k in ["future", "target", "predicted", "correct", "baseline"])
    ]
    print("\n[SUSPICIOUS COLUMNS IN RAW]")
    print(suspicious_cols)

    if dataset_path is not None and Path(dataset_path).exists():
        ds = pd.read_csv(dataset_path)
        print("\n[MODEL DATASET CHECK]")
        print("dataset rows:", len(ds))
        print("columns:", list(ds.columns))
        print("date range:", pd.to_datetime(ds["news_date_dt"], errors="coerce").min(), "->", pd.to_datetime(ds["news_date_dt"], errors="coerce").max())

        feature_cols = get_feature_cols()
        leaked_features = [
            c for c in feature_cols
            if any(k in c.lower() for k in ["future", "target", "predicted", "correct"])
        ]
        print("feature cols:", feature_cols)
        print("leaked feature cols:", leaked_features)

def save_latest_ticker_prediction_signals(
    *,
    dataset: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    output_path: Path = OUT_DIR / "latest_news_prediction_signals.csv",
    use_ticker_features: bool = USE_TICKER_FEATURES,
) -> pd.DataFrame:
    df, _ = prepare_model_frame(
        dataset,
        use_ticker_features=use_ticker_features,
    )

    # Make sure inference frame has exactly the same columns as training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].astype(float)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["predicted_positive_probability"] = proba
    out["predicted_direction"] = pred
    out["prediction_confidence"] = np.abs(proba - 0.5) * 2.0

    latest = (
        out.sort_values(["ticker", "news_date_dt"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )

    latest["signal_label"] = np.select(
        [
            latest["predicted_positive_probability"] >= 0.60,
            latest["predicted_positive_probability"] <= 0.40,
        ],
        [
            "Bullish",
            "Bearish",
        ],
        default="Neutral",
    )

    latest["signal_strength"] = pd.cut(
        latest["prediction_confidence"],
        bins=[-0.001, 0.10, 0.30, 1.00],
        labels=["Weak", "Medium", "Strong"],
    )

    keep_cols = [
        "ticker",
        "news_date",
        "news_date_dt",
        "predicted_positive_probability",
        "prediction_confidence",
        "predicted_direction",
        "signal_label",
        "signal_strength",
        "article_count",
        "weighted_sentiment",
        "mean_confidence",
        "positive_ratio",
        "negative_ratio",
        "past_5d_return",
        "past_20d_return",
        "past_20d_volatility",
    ]

    keep_cols = [c for c in keep_cols if c in latest.columns]
    latest = latest[keep_cols].sort_values("ticker").reset_index(drop=True)

    latest.to_csv(output_path, index=False)

    print("\nSaved latest ticker prediction signals:", output_path)
    print(latest.head(20))

    return latest

def run_all_ticker_news_prediction_experiment(
    *,
    min_abs_return_for_signal: Optional[float] = 0.01,
    model_type: str = "random_forest",
    output_tag: str = ALL_TICKERS_OUTPUT_TAG,
    refresh_raw_dataset: bool = False,
    save_outputs: bool = True,
    batch_size: int = ALL_TICKER_BATCH_SIZE,
    batch_index: int = ALL_TICKER_BATCH_INDEX,
    max_api_calls_per_run: int = MAX_API_CALLS_PER_RUN,
) -> Dict[str, Any]:
    print("\n==============================")
    print("ALL-TICKER NEWS PREDICTION EXPERIMENT")
    print("==============================")

    all_tickers = load_all_available_tickers_for_news_model()
    missing_tickers = get_missing_tickers(raw_path=ALL_TICKERS_RAW_PATH)
    batch_tickers = missing_tickers[:batch_size]

    print("all available tickers:", len(all_tickers))
    print("batch_index:", batch_index)
    print("batch_size:", batch_size)
    print("batch_tickers:", batch_tickers)
    print("n_batch_tickers:", len(batch_tickers))

    raw_path = ALL_TICKERS_RAW_PATH

    if not Path(raw_path).exists() and BUILD_V2_FROM_V1_IF_MISSING:
        print("\n[STEP] Building v2 enriched raw dataset from existing v1 dataset. No API call.")
        raw_df = build_v2_enriched_raw_from_v1(
            v1_path=ALL_TICKERS_RAW_PATH_V1,
            v2_path=ALL_TICKERS_RAW_PATH,
            chunk_size=5000,
        )

    elif refresh_raw_dataset or not Path(raw_path).exists():
        print("\n[STEP] Building/updating all-ticker raw dataset with API batch...")
        raw_df = refresh_raw_news_timeseries_dataset(
            tickers=batch_tickers,
            output_path=raw_path,
            lookback_days=NEWS_LOOKBACK_DAYS,
            exclude_recent_days=EXCLUDE_RECENT_DAYS,
            max_items_per_window=MAX_ITEMS_PER_WINDOW,
            horizon_days=HORIZON_DAYS,
            cache_ttl_s=CACHE_TTL_S,
            sleep_s=API_SLEEP_S,
            append_to_existing=True,
            max_api_calls_per_run=max_api_calls_per_run,
        )

    else:
        print("\n[STEP] Using existing v2 enriched raw dataset:")
        print(raw_path)
        raw_df = pd.read_csv(raw_path)

    if raw_df.empty:
        return {
            "ok": False,
            "reason": "All-ticker raw dataset is empty.",
            "raw_path": raw_path,
        }

    dataset = build_ticker_date_prediction_dataset_v2(
    raw_path=raw_path,
    min_abs_return_for_signal=min_abs_return_for_signal,
    )

    result = train_news_flow_predictor(
        dataset=dataset,
        model_type=model_type,
    )

    if save_outputs:
        suffix = "all" if min_abs_return_for_signal is None else str(min_abs_return_for_signal).replace(".", "p")
        model_name = f"{model_type}_{output_tag}"

        dataset_path = OUT_DIR / f"ticker_date_dataset_thr{suffix}_{model_name}.csv"
        pred_path = OUT_DIR / f"ticker_date_predictions_thr{suffix}_{model_name}.csv"
        metrics_path = OUT_DIR / f"ticker_date_metrics_thr{suffix}_{model_name}.json"
        explanation_path = OUT_DIR / f"ticker_date_model_explanation_thr{suffix}_{model_name}.json"
        model_path = OUT_DIR / f"ticker_date_model_thr{suffix}_{model_name}.joblib"

        dataset.to_csv(dataset_path, index=False)
        print("\nSaved:", dataset_path)

        if result.get("ok"):
            result["predictions"].to_csv(pred_path, index=False)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

            with open(explanation_path, "w", encoding="utf-8") as f:
                json.dump(result["model_explanation"], f, indent=2, ensure_ascii=False)

            trained_tickers = sorted(dataset["ticker"].dropna().astype(str).str.upper().unique().tolist())

            joblib.dump(
                {
                    "model": result["model"],
                    "feature_cols": result["feature_cols"],
                    "metrics": result["metrics"],
                    "threshold": min_abs_return_for_signal,
                    "model_type": model_name,
                    "base_model_type": model_type,
                    "output_tag": output_tag,
                    "horizon_days": HORIZON_DAYS,
                    "all_available_tickers": all_tickers,
                    "trained_tickers": trained_tickers,
                    "raw_path": raw_path,
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                },
                model_path,
            )

            print("Saved:", model_path)
            print("Saved:", pred_path)
            print("Saved:", metrics_path)
            print("Saved:", explanation_path)
            if (
                model_type == "logistic"
                and float(min_abs_return_for_signal) == 0.02
                and result["metrics"].get("use_as_portfolio_signal")
            ):
                best_model_path = OUT_DIR / "best_news_prediction_model.joblib"
                best_metrics_path = OUT_DIR / "best_news_prediction_metrics.json"
                best_predictions_path = OUT_DIR / "best_news_prediction_predictions.csv"

                joblib.dump(
                    {
                        "model": result["model"],
                        "feature_cols": result["feature_cols"],
                        "metrics": result["metrics"],
                        "threshold": min_abs_return_for_signal,
                        "model_type": model_name,
                        "base_model_type": model_type,
                        "output_tag": output_tag,
                        "horizon_days": HORIZON_DAYS,
                        "all_available_tickers": all_tickers,
                        "trained_tickers": trained_tickers,
                        "raw_path": raw_path,
                    },
                    best_model_path,
                )

                with open(best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

                result["predictions"].to_csv(best_predictions_path, index=False)
                latest_signals_path = OUT_DIR / "latest_news_prediction_signals.csv"

                latest_dataset = build_ticker_date_prediction_dataset_v2(
                    raw_path=raw_path,
                    min_abs_return_for_signal=None,
                )

                save_latest_ticker_prediction_signals(
                    dataset=latest_dataset,
                    model=result["model"],
                    feature_cols=result["feature_cols"],
                    output_path=latest_signals_path,
                    use_ticker_features=USE_TICKER_FEATURES,
                )

                print("Saved BEST model:", best_model_path)
                print("Saved BEST metrics:", best_metrics_path)
                print("Saved BEST predictions:", best_predictions_path)

    return {
        "ok": bool(result.get("ok")),
        "rows": int(len(dataset)),
        "raw_rows": int(len(raw_df)),
        "all_available_tickers": all_tickers,
        "batch_tickers": batch_tickers,
        "trained_tickers": sorted(dataset["ticker"].dropna().astype(str).str.upper().unique().tolist()),
        "raw_path": raw_path,
        "dataset": dataset,
        **result,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if REFRESH_RAW_DATASET:
        refresh_raw_news_timeseries_dataset(
            tickers=TICKERS_FOR_NEWS_DATASET,
            output_path=RAW_PATH,
            lookback_days=NEWS_LOOKBACK_DAYS,
            exclude_recent_days=EXCLUDE_RECENT_DAYS,
            max_items_per_window=MAX_ITEMS_PER_WINDOW,
            horizon_days=HORIZON_DAYS,
            cache_ttl_s=CACHE_TTL_S,
            sleep_s=API_SLEEP_S,
            append_to_existing=True,
            max_api_calls_per_run=MAX_API_CALLS_PER_RUN,
        )

    #print_missing_tickers_status()

    audit_news_prediction_dataset_for_leakage(
    raw_path=ALL_TICKERS_RAW_PATH,
    )
    raw = pd.read_csv(ALL_TICKERS_RAW_PATH)
    raw["news_date"] = pd.to_datetime(raw["news_date"], errors="coerce")

    for t in ["GOOG", "GOOGL"]:
        x = raw[raw["ticker"].astype(str).str.upper().eq(t)].copy()
        print("\n====", t, "====")
        print("rows:", len(x))
        print("date range:", x["news_date"].min(), "->", x["news_date"].max())
        print(
            x.sort_values("news_date", ascending=False)[
                ["ticker", "news_date", "headline", "source", "url"]
            ].head(20)
        )
    experiments = [
        #{"min_abs_return_for_signal": 0.01, "model_type": "logistic", "all_tickers": True},
        {"min_abs_return_for_signal": 0.02, "model_type": "logistic", "all_tickers": True},
        #{"min_abs_return_for_signal": 0.01, "model_type": "random_forest", "all_tickers": True},
        #{"min_abs_return_for_signal": 0.02, "model_type": "random_forest", "all_tickers": True},
    ]


    for exp in experiments:
        if exp.get("all_tickers"):
            result = run_all_ticker_news_prediction_experiment(
                min_abs_return_for_signal=exp["min_abs_return_for_signal"],
                model_type=exp["model_type"],
                output_tag=f"{ALL_TICKERS_OUTPUT_TAG}_{exp['model_type']}",
                refresh_raw_dataset=False,
                save_outputs=True,
                batch_size=5,
                batch_index=0,
                max_api_calls_per_run=70,
            )
        else:
            result = run_saved_news_flow_experiment(
                raw_path=RAW_PATH,
                min_abs_return_for_signal=exp["min_abs_return_for_signal"],
                model_type=exp["model_type"],
                save_outputs=True,
            )

        print("\n=== NEWS FLOW RESULT ===")
        print("threshold:", exp["min_abs_return_for_signal"])
        print("model_type:", exp["model_type"])
        print("OK:", result.get("ok"))
        print("rows:", result.get("rows"))
        print("raw_rows:", result.get("raw_rows"))
        print("trained_tickers:", result.get("trained_tickers"))

        if result.get("ok"):
            print("\n=== Metrics ===")
            for k, v in result["metrics"].items():
                if k != "classification_report":
                    print(k, ":", v)

            print("\n=== Classification report ===")
            print(result["metrics"]["classification_report"])

            print("\n=== Top model explanation items ===")
            for item in result["model_explanation"].get("items", [])[:10]:
                print(item)
        else:
            print("Reason:", result.get("reason"))

        print("\n" + "=" * 80 + "\n")

        