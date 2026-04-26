from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse your existing fetcher
from agents_langgraph import news_agent_fetch_for_tickers


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


# =========================================================
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