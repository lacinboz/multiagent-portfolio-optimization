from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse your existing fetcher
from agents_langgraph import news_agent_fetch_for_tickers, historical_news_agent_fetch_for_tickers

# =========================================================
# CONFIG
# =========================================================

FINBERT_MODEL_NAME = "ProsusAI/finbert"

DEFAULT_SOURCE_CREDIBILITY = {
    "Reuters": 0.95,
    "Bloomberg": 0.95,
    "CNBC": 0.88,
    "Wall Street Journal": 0.92,
    "MarketWatch": 0.82,
    "Yahoo": 0.75,
    "Yahoo Finance": 0.75,
    "SeekingAlpha": 0.68,
    "Benzinga": 0.65,
    "ChartMill": 0.55,
    "unknown": 0.50,
}


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class ArticleSignal:
    ticker: str
    headline: str
    summary: str
    source: str
    datetime: Optional[Any]
    url: Optional[str]
    probs: Dict[str, float]
    article_sentiment: float
    article_confidence: float
    recency_weight: float
    combined_weight: float


@dataclass
class TickerNewsSignal:
    ticker: str
    sentiment_score: float
    confidence_score: float
    weighted_article_count: float
    raw_article_count: int
    sentiment_variance: float


# =========================================================
# SMALL HELPERS
# =========================================================

def _safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _normalize_ticker(x: Any) -> str:
    return _safe_text(x).upper()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, (int, float)):
        try:
            if float(value) <= 0:
                return None
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    try:
        if s.endswith("Z"):
            s = s[:-1]
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

# REUSE EXISTING NEWS FETCH
# =========================================================

def fetch_news_for_tickers(
    tickers: List[str],
    lookback_days: int = 7,
    max_items_per_ticker: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fetched = news_agent_fetch_for_tickers(
        tickers=tickers,
        include_news=True,
        lookback_days=lookback_days,
        min_company_items=1,
        max_items_per_ticker=max_items_per_ticker,
        include_market_fallback=True,
        market_category="general",
        cache_ttl_s=600,
        sleep_s=0.25,
    )

    flat_items = (fetched or {}).get("flat_items") or []
    evidence_map = (fetched or {}).get("evidence_map") or {}
    return flat_items, evidence_map


def normalize_news_items(raw_news: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for item in raw_news or []:
        ticker = item.get("ticker") or item.get("related") or item.get("symbol") or ""

        normalized.append(
            {
                "ticker": _normalize_ticker(ticker),
                "headline": _safe_text(item.get("headline")),
                "summary": _safe_text(item.get("summary")),
                "source": _safe_text(item.get("source")) or "unknown",
                "datetime": item.get("datetime"),
                "url": item.get("url"),
                "id": item.get("id"),
                "evidence_id": item.get("evidence_id"),
            }
        )

    return [x for x in normalized if x["ticker"]]


# =========================================================
# FINBERT MODEL
# =========================================================

class FinBERTScorer:
    """
    Loads FinBERT once and returns class probabilities for each text.
    """

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT labels are usually something like:
        # 0 -> positive, 1 -> negative, 2 -> neutral
        # But we do NOT hardcode blindly; we read from config.
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.id2label = {int(k): str(v).lower() for k, v in id2label.items()}

    def score_texts(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> List[Dict[str, float]]:
        outputs: List[Dict[str, float]] = []

        if not texts:
            return outputs

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                item = {}
                for idx, p in enumerate(row):
                    label = self.id2label.get(idx, str(idx)).lower()
                    item[label] = float(p)

                # normalize expected keys
                outputs.append(
                    {
                        "positive": float(item.get("positive", 0.0)),
                        "negative": float(item.get("negative", 0.0)),
                        "neutral": float(item.get("neutral", 0.0)),
                    }
                )

        return outputs


# =========================================================
# CONFIDENCE / RECENCY
# =========================================================

def _source_credibility(source: str) -> float:
    source = _safe_text(source)
    if source in DEFAULT_SOURCE_CREDIBILITY:
        return DEFAULT_SOURCE_CREDIBILITY[source]

    for known, val in DEFAULT_SOURCE_CREDIBILITY.items():
        if known.lower() in source.lower():
            return val

    return DEFAULT_SOURCE_CREDIBILITY["unknown"]


def _content_richness(article: Dict[str, Any]) -> float:
    text = f"{_safe_text(article.get('headline'))} {_safe_text(article.get('summary'))}".strip()
    if not text:
        return 0.20
    return float(min(1.0, 0.25 + (len(text) / 500.0)))


def _recency_weight(
    dt_value: Any,
    half_life_days: float = 2.0,
    now: Optional[datetime] = None,
) -> float:
    now = now or _now_utc()
    dt = _to_datetime(dt_value)
    if dt is None:
        return 0.35

    age_seconds = max(0.0, (now - dt).total_seconds())
    age_days = age_seconds / 86400.0
    lam = math.log(2.0) / max(half_life_days, 1e-6)
    return float(math.exp(-lam * age_days))


def compute_article_confidence(
    article: Dict[str, Any],
    finbert_probs: Dict[str, float],
    now: Optional[datetime] = None,
) -> Tuple[float, float]:
    """
    Final confidence combines:
    - model certainty
    - source credibility
    - recency
    - text richness
    """
    model_conf = max(
        float(finbert_probs.get("positive", 0.0)),
        float(finbert_probs.get("negative", 0.0)),
        float(finbert_probs.get("neutral", 0.0)),
    )
    source_conf = _source_credibility(_safe_text(article.get("source")))
    recency = _recency_weight(article.get("datetime"), now=now)
    richness = _content_richness(article)

    confidence = (
        0.45 * model_conf
        + 0.20 * source_conf
        + 0.20 * recency
        + 0.15 * richness
    )
    confidence = float(max(0.0, min(1.0, confidence)))
    return confidence, recency


# =========================================================
# ARTICLE SIGNALS
# =========================================================

def build_article_signals_with_finbert(
    raw_news: List[Dict[str, Any]],
    tickers: Optional[List[str]] = None,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 16,
) -> List[ArticleSignal]:
    normalized = normalize_news_items(raw_news)
    allowed = {_normalize_ticker(t) for t in (tickers or []) if _normalize_ticker(t)}

    filtered = [x for x in normalized if (not allowed or x["ticker"] in allowed)]
    if not filtered:
        return []

    scorer = FinBERTScorer(model_name=model_name)

    texts = []
    for article in filtered:
        headline = _safe_text(article.get("headline"))
        summary = _safe_text(article.get("summary"))
        if summary:
            text = f"{headline} [SEP] {summary}"
        else:
            text = headline
        texts.append(text)

    probs_list = scorer.score_texts(texts=texts, batch_size=batch_size)
    now = _now_utc()

    article_signals: List[ArticleSignal] = []
    for article, probs in zip(filtered, probs_list):
        positive = float(probs.get("positive", 0.0))
        negative = float(probs.get("negative", 0.0))
        neutral = float(probs.get("neutral", 0.0))

        # sentiment in [-1, 1]
        sentiment = positive - negative

        confidence, recency_weight = compute_article_confidence(
            article=article,
            finbert_probs=probs,
            now=now,
        )

        combined_weight = max(1e-6, 0.50 * confidence + 0.50 * recency_weight)

        article_signals.append(
            ArticleSignal(
                ticker=article["ticker"],
                headline=_safe_text(article.get("headline")),
                summary=_safe_text(article.get("summary")),
                source=_safe_text(article.get("source")),
                datetime=article.get("datetime"),
                url=article.get("url"),
                probs={
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                },
                article_sentiment=float(max(-1.0, min(1.0, sentiment))),
                article_confidence=confidence,
                recency_weight=recency_weight,
                combined_weight=combined_weight,
            )
        )

    return article_signals


# =========================================================
# AGGREGATE TO TICKER LEVEL
# =========================================================

def aggregate_news_signal_by_ticker(
    article_signals: List[ArticleSignal],
    tickers: Optional[List[str]] = None,
) -> Dict[str, TickerNewsSignal]:
    allowed = {_normalize_ticker(t) for t in (tickers or []) if _normalize_ticker(t)}
    grouped: Dict[str, List[ArticleSignal]] = {}

    for art in article_signals:
        if allowed and art.ticker not in allowed:
            continue
        grouped.setdefault(art.ticker, []).append(art)

    out: Dict[str, TickerNewsSignal] = {}

    for ticker, items in grouped.items():
        weights = np.array([x.combined_weight for x in items], dtype=float)
        sentiments = np.array([x.article_sentiment for x in items], dtype=float)
        confidences = np.array([x.article_confidence for x in items], dtype=float)

        if weights.sum() <= 0:
            weights = np.ones_like(weights)

        weights = weights / weights.sum()

        sentiment_score = float(np.sum(weights * sentiments))
        confidence_score = float(np.sum(weights * confidences))
        sentiment_variance = float(np.sum(weights * (sentiments - sentiment_score) ** 2))

        out[ticker] = TickerNewsSignal(
            ticker=ticker,
            sentiment_score=max(-1.0, min(1.0, sentiment_score)),
            confidence_score=max(0.0, min(1.0, confidence_score)),
            weighted_article_count=float(np.sum([x.combined_weight for x in items])),
            raw_article_count=len(items),
            sentiment_variance=max(0.0, sentiment_variance),
        )

    for ticker in allowed:
        if ticker not in out:
            out[ticker] = TickerNewsSignal(
                ticker=ticker,
                sentiment_score=0.0,
                confidence_score=0.0,
                weighted_article_count=0.0,
                raw_article_count=0,
                sentiment_variance=0.0,
            )

    return out

def build_prediction_signals(
    ticker_signals: Dict[str, TickerNewsSignal],
    alpha: float,
    beta: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Convert news signals into forward-looking prediction signals.
    """

    out = {}

    for ticker, sig in ticker_signals.items():
        s = float(sig.sentiment_score)
        c = float(sig.confidence_score)
        var = float(sig.sentiment_variance)

        # 🔥 direction
        if s > 0.05:
            direction = "positive"
        elif s < -0.05:
            direction = "negative"
        else:
            direction = "neutral"

        # 🔥 expected return adjustment (same logic as mu delta)
        mu_delta = alpha * s * c

        # 🔥 risk adjustment (same logic as covariance uncertainty)
        uncertainty = (
            0.55 * (1.0 - c)
            + 0.45 * min(1.0, 3.0 * var)
        )

        variance_delta = beta * uncertainty

        out[ticker] = {
            "predicted_direction": direction,
            "prediction_confidence": c,
            "expected_return_adjustment": mu_delta,
            "risk_adjustment": variance_delta,
            "sentiment_score": s,
            "sentiment_variance": var,
        }

    return out


# =========================================================
# MVO ADJUSTMENT
# =========================================================

def adjust_expected_returns(
    mu: pd.Series,
    ticker_signals: Dict[str, TickerNewsSignal],
    alpha: float = 0.08,
    power: float = 1.5,
) -> pd.Series:
    """
    Stronger nonlinear return adjustment:
    mu' = mu + alpha * sign(s) * |s|^power * confidence
    """
    adjusted = mu.copy().astype(float)

    for ticker in adjusted.index:
        t = _normalize_ticker(ticker)
        sig = ticker_signals.get(t)
        if sig is None:
            continue

        s = float(sig.sentiment_score)
        c = float(sig.confidence_score)

        nonlinear_sentiment = math.copysign(abs(s) ** power, s)
        delta = alpha * s * c

        adjusted.loc[ticker] = float(adjusted.loc[ticker]) + float(delta)

    return adjusted

def adjust_covariance_matrix(
    cov: pd.DataFrame,
    ticker_signals: Dict[str, TickerNewsSignal],
    beta: float = 0.35,
    variance_floor: float = 1e-10,
    confidence_weight: float = 0.55,
    variance_weight: float = 0.45,
    variance_scale: float = 3.0,
) -> pd.DataFrame:
    """
    Stronger diagonal variance adjustment:
    uncertainty_i = confidence_weight * (1 - confidence)
                  + variance_weight * min(1, variance_scale * sentiment_variance)

    sigma_i'^2 = sigma_i^2 * (1 + beta * uncertainty_i)
    """
    adjusted = cov.copy().astype(float)

    for ticker in adjusted.index:
        t = _normalize_ticker(ticker)
        sig = ticker_signals.get(t)
        if sig is None:
            continue

        original_var = float(adjusted.loc[ticker, ticker])
        original_var = max(original_var, variance_floor)

        disagreement_term = min(1.0, variance_scale * float(sig.sentiment_variance))
        uncertainty = (
            confidence_weight * (1.0 - float(sig.confidence_score))
            + variance_weight * disagreement_term
        )

        multiplier = 1.0 + beta * uncertainty
        adjusted.loc[ticker, ticker] = original_var * multiplier

    adjusted = 0.5 * (adjusted + adjusted.T)
    return adjusted

# =========================================================
# NEWS ADJUSTMENT EVALUATION
# =========================================================

def _safe_float_eval(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _effective_number_of_holdings(weights: Dict[str, Any]) -> float:
    vals = np.array([float(v) for v in (weights or {}).values() if abs(float(v)) > 1e-8], dtype=float)
    if vals.size == 0:
        return 0.0
    denom = float(np.sum(vals ** 2))
    return float(1.0 / denom) if denom > 0 else 0.0


def _max_weight(weights: Dict[str, Any]) -> float:
    vals = [abs(float(v)) for v in (weights or {}).values()]
    return float(max(vals)) if vals else 0.0


def _turnover(base_weights: Dict[str, Any], news_weights: Dict[str, Any]) -> float:
    tickers = sorted(set((base_weights or {}).keys()) | set((news_weights or {}).keys()))
    return float(
        0.5 * sum(
            abs(float(news_weights.get(t, 0.0)) - float(base_weights.get(t, 0.0)))
            for t in tickers
        )
    )


def evaluate_news_adjustment_effect(
    *,
    base_weights: Dict[str, Any],
    base_metrics: Dict[str, Any],
    news_weights: Dict[str, Any],
    news_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates the portfolio-level impact of the news/FinBERT adjustment.

    Important thesis framing:
    This does NOT evaluate news as a standalone price-direction predictor.
    It evaluates whether news-adjusted inputs changed the optimized portfolio
    in terms of return, risk, efficiency, concentration, and allocation.
    """

    base_weights = base_weights or {}
    news_weights = news_weights or {}
    base_metrics = base_metrics or {}
    news_metrics = news_metrics or {}

    base_return = _safe_float_eval(base_metrics.get("return"))
    news_return = _safe_float_eval(news_metrics.get("return"))

    base_vol = _safe_float_eval(base_metrics.get("vol"))
    news_vol = _safe_float_eval(news_metrics.get("vol"))

    base_sharpe = _safe_float_eval(base_metrics.get("sharpe"))
    news_sharpe = _safe_float_eval(news_metrics.get("sharpe"))

    base_max_w = _max_weight(base_weights)
    news_max_w = _max_weight(news_weights)

    base_eff_n = _effective_number_of_holdings(base_weights)
    news_eff_n = _effective_number_of_holdings(news_weights)

    tickers = sorted(set(base_weights.keys()) | set(news_weights.keys()))

    weight_changes = {
        t: {
            "base_weight": float(base_weights.get(t, 0.0)),
            "news_weight": float(news_weights.get(t, 0.0)),
            "delta_weight": float(news_weights.get(t, 0.0) - base_weights.get(t, 0.0)),
        }
        for t in tickers
    }

    delta_return = None if base_return is None or news_return is None else news_return - base_return
    delta_vol = None if base_vol is None or news_vol is None else news_vol - base_vol
    delta_sharpe = None if base_sharpe is None or news_sharpe is None else news_sharpe - base_sharpe

    delta_max_weight = news_max_w - base_max_w
    delta_effective_n = news_eff_n - base_eff_n
    turnover = _turnover(base_weights, news_weights)

    risk_effect = "unchanged"
    if delta_vol is not None:
        if delta_vol > 1.0:
            "increased_risk"
        elif delta_vol > 0:
            "slightly_increased_risk"
        else:
            "reduced_risk"

    efficiency_effect = "unchanged"
    if delta_sharpe is not None:
        if delta_sharpe > 1e-6:
            efficiency_effect = "improved_efficiency"
        elif delta_sharpe < -1e-6:
            efficiency_effect = "reduced_efficiency"

    concentration_effect = "unchanged"
    if delta_effective_n > 1e-6 and delta_max_weight < 1e-6:
        concentration_effect = "more_diversified"
    elif delta_effective_n < -1e-6 or delta_max_weight > 1e-6:
        concentration_effect = "more_concentrated"

    return {
        "thesis_framing": (
            "The news module is evaluated as a portfolio-level adjustment mechanism, "
            "not as a standalone short-term price-direction predictor."
        ),
        "base": {
            "return": base_return,
            "vol": base_vol,
            "sharpe": base_sharpe,
            "max_weight": base_max_w,
            "effective_n": base_eff_n,
        },
        "news_adjusted": {
            "return": news_return,
            "vol": news_vol,
            "sharpe": news_sharpe,
            "max_weight": news_max_w,
            "effective_n": news_eff_n,
        },
        "deltas": {
            "return": delta_return,
            "vol": delta_vol,
            "sharpe": delta_sharpe,
            "max_weight": delta_max_weight,
            "effective_n": delta_effective_n,
            "turnover": turnover,
        },
        "effects": {
            "risk_effect": risk_effect,
            "efficiency_effect": efficiency_effect,
            "concentration_effect": concentration_effect,
        },
        "weight_changes": weight_changes,
    }

def evaluate_news_prediction_against_future_returns(
    *,
    article_signals: List[Dict[str, Any]],
    price_dir: str = "data/raw/daily_yahoo",
    horizons: List[int] = [1, 3, 7],
    sentiment_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates whether FinBERT news sentiment predicts future price direction.

    For each news article:
    - get ticker
    - get article date
    - get close price at/after news date
    - get close price after h trading days
    - compare predicted direction with realized future return direction
    """

    rows = []

    for art in article_signals or []:
        if not isinstance(art, dict):
            continue

        ticker = _normalize_ticker(art.get("ticker"))
        if not ticker:
            continue

        sentiment = _safe_float_eval(art.get("article_sentiment"))
        confidence = _safe_float_eval(art.get("article_confidence"))
        news_dt = _to_datetime(art.get("datetime"))

        if sentiment is None or news_dt is None:
            continue

        if sentiment > sentiment_threshold:
            predicted_direction = "positive"
            predicted_sign = 1
        elif sentiment < -sentiment_threshold:
            predicted_direction = "negative"
            predicted_sign = -1
        else:
            predicted_direction = "neutral"
            predicted_sign = 0

        if predicted_sign == 0:
            continue

        price_path = Path(price_dir) / f"{ticker}_daily.csv"
        if not price_path.exists():
            continue

        try:
            prices = pd.read_csv(price_path)
        except Exception:
            continue

        if prices.empty or "timestamp" not in prices.columns or "close" not in prices.columns:
            continue

        prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="coerce")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = prices.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

        if prices.empty:
            continue

        news_date = pd.Timestamp(news_dt.date())

        # first trading day on or after news date
        # leakage-safe: first trading day strictly AFTER news date
        candidates = prices[prices["timestamp"] > news_date]
        if candidates.empty:
            continue

        start_idx = int(candidates.index[0])
        start_date = prices.loc[start_idx, "timestamp"]
        start_close = float(prices.loc[start_idx, "close"])
        if candidates.empty:
            continue

        start_idx = int(candidates.index[0])
        start_close = float(prices.loc[start_idx, "close"])

        for h in horizons:
            future_idx = start_idx + int(h)

            # future price not available yet
            if future_idx >= len(prices):
                continue

            future_close = float(prices.loc[future_idx, "close"])
            future_date = prices.loc[future_idx, "timestamp"]
            future_return = (future_close / start_close) - 1.0

            actual_sign = 1 if future_return > 0 else (-1 if future_return < 0 else 0)
            correct = bool(predicted_sign == actual_sign) if actual_sign != 0 else None

            rows.append(
                {
                    "ticker": ticker,
                    "news_date": str(news_date.date()),
                    "start_price_date": str(start_date.date()),
                    "future_price_date": str(future_date.date()),
                    "start_close": float(start_close),
                    "future_close": float(future_close),
                    "horizon_days": int(h),
                    "headline": art.get("headline"),
                    "source": art.get("source"),
                    "sentiment": float(sentiment),
                    "confidence": float(confidence) if confidence is not None else None,
                    "predicted_direction": predicted_direction,
                    "future_return": float(future_return),
                    "actual_direction": "positive" if actual_sign > 0 else "negative" if actual_sign < 0 else "flat",
                    "correct": correct,
                }
            )

    if not rows:
        return {
            "ok": False,
            "reason": "No evaluable news articles. Future prices may not be available yet.",
            "rows": [],
            "summary": {},
        }

    df = pd.DataFrame(rows)

    summary = {}
    for h, g in df.groupby("horizon_days"):
        valid = g[g["correct"].notna()]
        accuracy = float(valid["correct"].mean()) if len(valid) else None

        corr = None
        try:
            corr = float(g[["sentiment", "future_return"]].corr().iloc[0, 1])
            if not np.isfinite(corr):
                corr = None
        except Exception:
            corr = None

        summary[int(h)] = {
            "n": int(len(g)),
            "valid_n": int(len(valid)),
            "direction_accuracy": accuracy,
            "avg_future_return": float(g["future_return"].mean()),
            "sentiment_future_return_corr": corr,
        }

    return {
        "ok": True,
        "rows": rows,
        "summary": summary,
    }

def build_historical_news_prediction_evaluation(
    *,
    tickers: List[str],
    lookback_days: int = 365,
    exclude_recent_days: int = 14,
    max_items_per_ticker: int = 100,
    price_dir: str = "data/raw/daily_yahoo",
    horizons: List[int] = [1, 3, 7],
    sentiment_threshold: float = 0.05,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 16,
) -> Dict[str, Any]:

    tickers = [_normalize_ticker(t) for t in tickers if _normalize_ticker(t)]

    fetched = historical_news_agent_fetch_for_tickers(
        tickers=tickers,
        include_news=True,
        lookback_days=lookback_days,
        exclude_recent_days=exclude_recent_days,
        max_items_per_ticker=max_items_per_ticker,
        cache_ttl_s=86400,
        sleep_s=0.25,
    )

    raw_news = (fetched or {}).get("flat_items") or []
    evidence_map = (fetched or {}).get("evidence_map") or {}

    base_payload = {
        "from_date": (fetched or {}).get("from_date"),
        "to_date": (fetched or {}).get("to_date"),
        "lookback_days": lookback_days,
        "exclude_recent_days": exclude_recent_days,
        "tickers": tickers,
        "fetch_stats": (fetched or {}).get("stats") or {},
        "raw_news_count": len(raw_news),
        "evidence_map_count": len(evidence_map),
        "sample_news": raw_news[:10],
        "thesis_framing": (
            "This evaluation does not test FinBERT as a general sentiment classifier. "
            "It tests whether FinBERT-based historical news signals are directionally "
            "aligned with subsequent stock returns for the selected portfolio tickers."
        ),
    }

    if not raw_news:
        return {
            "ok": False,
            "reason": "No historical news could be fetched for the selected tickers.",
            "rows": [],
            "summary": {},
            "article_signal_count": 0,
            **base_payload,
        }

    article_signals = build_article_signals_with_finbert(
        raw_news=raw_news,
        tickers=tickers,
        model_name=model_name,
        batch_size=batch_size,
    )

    if not article_signals:
        return {
            "ok": False,
            "reason": "Historical news was fetched, but FinBERT produced no usable article signals.",
            "rows": [],
            "summary": {},
            "article_signal_count": 0,
            "sample_article_signals": [],
            **base_payload,
        }

    article_signal_dicts = [asdict(x) for x in article_signals]

    evaluation = evaluate_news_prediction_against_future_returns(
        article_signals=article_signal_dicts,
        price_dir=price_dir,
        horizons=horizons,
        sentiment_threshold=sentiment_threshold,
    )

    rows = evaluation.get("rows") or []

    evaluation.update(
        {
            **base_payload,
            "historical_news_count": len(raw_news),
            "article_signal_count": len(article_signals),
            "sample_article_signals": article_signal_dicts[:10],
            "sample_rows": rows[:20],
            "horizons": horizons,
            "sentiment_threshold": sentiment_threshold,
            "price_dir": price_dir,
        }
    )

    return evaluation
# =========================================================
# MAIN ENTRY
# =========================================================

def build_probabilistic_news_adjusted_inputs(
    mu: pd.Series,
    cov: pd.DataFrame,
    tickers: List[str],
    raw_news: Optional[List[Dict[str, Any]]] = None,
    *,
    fetch_if_missing: bool = False,
    lookback_days: int = 7,
    max_items_per_ticker: int = 12,
    alpha: float = 0.08,
    beta: float = 0.35,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Main entry point.

    If raw_news is given -> uses it.
    If raw_news is None and fetch_if_missing=True -> fetches via existing fetcher.
    """
    if raw_news is None:
        if not fetch_if_missing:
            raise ValueError("raw_news is None and fetch_if_missing=False")

        raw_news, evidence_map = fetch_news_for_tickers(
            tickers=tickers,
            lookback_days=lookback_days,
            max_items_per_ticker=max_items_per_ticker,
        )
    else:
        evidence_map = {}

    article_signals = build_article_signals_with_finbert(
        raw_news=raw_news,
        tickers=tickers,
        model_name=model_name,
        batch_size=batch_size,
    )

    ticker_signals = aggregate_news_signal_by_ticker(
        article_signals=article_signals,
        tickers=tickers,
    )
    prediction_signals = build_prediction_signals(
    ticker_signals=ticker_signals,
    alpha=alpha,
    beta=beta,
    )

    adjusted_mu = adjust_expected_returns(
    mu=mu,
    ticker_signals=ticker_signals,
    alpha=alpha,
    power=1.5,
)

    adjusted_cov = adjust_covariance_matrix(
        cov=cov,
        ticker_signals=ticker_signals,
        beta=beta,
        confidence_weight=0.55,
        variance_weight=0.45,
        variance_scale=3.0,
    )
    prediction_evaluation = evaluate_news_prediction_against_future_returns(
        article_signals=[asdict(x) for x in article_signals],
        price_dir="data/raw/daily_yahoo",
        horizons=[1, 3, 7],
    )
    historical_prediction_evaluation = build_historical_news_prediction_evaluation(
            tickers=tickers,
            lookback_days=365,
            exclude_recent_days=14,
            max_items_per_ticker=20,
            price_dir="data/raw/daily_yahoo",
            horizons=[1, 3, 7],
            sentiment_threshold=0.05,
            model_name=model_name,
            batch_size=batch_size,
        )

    return {
        "raw_news": raw_news,
        "evidence_map": evidence_map,
        "article_signals": [asdict(x) for x in article_signals],
        "ticker_signals": {k: asdict(v) for k, v in ticker_signals.items()},
        "adjusted_mu": adjusted_mu,
        "adjusted_cov": adjusted_cov,
        "parameters": {
            "alpha": alpha,
            "beta": beta,
            "lookback_days": lookback_days,
            "max_items_per_ticker": max_items_per_ticker,
            "model_name": model_name,
            "batch_size": batch_size,
        },
        "prediction_signals": prediction_signals,
        "prediction_evaluation": prediction_evaluation,
        "historical_prediction_evaluation": historical_prediction_evaluation,
    }


def build_adjusted_inputs_from_existing_news_state(
    mu: pd.Series,
    cov: pd.DataFrame,
    tickers: List[str],
    news_raw: List[Dict[str, Any]],
    *,
    alpha: float = 0.08,
    beta: float = 0.35,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Use this if your graph already has state['news_raw'].
    """
    return build_probabilistic_news_adjusted_inputs(
        mu=mu,
        cov=cov,
        tickers=tickers,
        raw_news=news_raw,
        fetch_if_missing=False,
        alpha=alpha,
        beta=beta,
        model_name=model_name,
        batch_size=batch_size,
    )