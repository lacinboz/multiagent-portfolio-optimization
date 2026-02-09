# finnhub_news_probe.py
# Usage examples:
#   python finnhub_news_probe.py --tickers AAPL MSFT NVDA --lookback-days 7
#   python finnhub_news_probe.py --tickers-file tickers.txt --lookback-days 3 --market-categories general merger
# Output:
#   out_company_news.json, out_market_news.json, out_market_filtered.json


from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
import json
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE_URL = "https://finnhub.io/api/v1"


@dataclass
class NewsItem:
    source: str
    headline: str
    summary: str
    url: str
    datetime_unix: int
    related: str = ""
    category: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NewsItem":
        return NewsItem(
            source=str(d.get("source", "")),
            headline=str(d.get("headline", "")),
            summary=str(d.get("summary", "")),
            url=str(d.get("url", "")),
            datetime_unix=int(d.get("datetime", 0) or 0),
            related=str(d.get("related", "")),
            category=str(d.get("category", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "headline": self.headline,
            "summary": self.summary,
            "url": self.url,
            "datetime": self.datetime_unix,
            "related": self.related,
            "category": self.category,
        }


def _utc_today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _days_ago_str(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()


def read_tickers_from_file(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            # allow comma-separated
            parts = [p.strip() for p in t.split(",") if p.strip()]
            tickers.extend(parts)
    # unique keep order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def finnhub_get(endpoint: str, params: Dict[str, Any], api_key: str, timeout: int = 30) -> Any:
    url = f"{BASE_URL}{endpoint}"
    params = dict(params)
    params["token"] = api_key
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        # try to show useful error body
        raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:300]}")
    return r.json()


def fetch_company_news(symbol: str, frm: str, to: str, api_key: str) -> List[NewsItem]:
    data = finnhub_get("/company-news", {"symbol": symbol, "from": frm, "to": to}, api_key)
    if not isinstance(data, list):
        return []
    return [NewsItem.from_dict(x) for x in data]


def fetch_market_news(category: str, api_key: str, min_id: int = 0) -> List[NewsItem]:
    data = finnhub_get("/news", {"category": category, "minId": min_id}, api_key)
    if not isinstance(data, list):
        return []
    return [NewsItem.from_dict(x) for x in data]


def simple_ticker_filter(news: List[NewsItem], tickers: List[str]) -> List[NewsItem]:
    """
    Very basic filter:
    keep item if any ticker appears in headline or summary or related field.
    (This is just a quick sanity check before LLM-based mapping.)
    """
    tset = [t.upper() for t in tickers]
    out: List[NewsItem] = []
    for item in news:
        text = f"{item.headline} {item.summary} {item.related}".upper()
        if any(t in text for t in tset):
            out.append(item)
    return out


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=[], help="Tickers like AAPL MSFT NVDA")
    ap.add_argument("--tickers-file", default="", help="File with one ticker per line (or comma-separated).")
    ap.add_argument("--lookback-days", type=int, default=7, help="Lookback window in days.")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep between API calls to avoid rate limits.")
    ap.add_argument("--market-categories", nargs="*", default=["general"], help="general / forex / crypto / merger")
    ap.add_argument("--out-prefix", default="out", help="Output prefix for JSON files.")
    args = ap.parse_args()

    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing FINNHUB_API_KEY env var. Example: export FINNHUB_API_KEY='YOUR_KEY'")

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_from_file(args.tickers_file))
    if args.tickers:
        tickers.extend([t.strip() for t in args.tickers if t.strip()])

    # unique keep order
    seen = set()
    uniq = []
    for t in tickers:
        t = t.upper()
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    tickers = uniq

    frm = _days_ago_str(args.lookback_days)
    to = _utc_today_str()

    print(f"Tickers: {len(tickers)} | lookback: {frm} -> {to}")
    print(f"Market categories: {args.market_categories}")

    # 1) Company news per ticker
    company_results: Dict[str, Any] = {
        "from": frm,
        "to": to,
        "tickers": tickers,
        "items_by_ticker": {},
        "empty_tickers": [],
        "errors": {},
    }

    for i, t in enumerate(tickers):
        try:
            items = fetch_company_news(t, frm, to, api_key)
            company_results["items_by_ticker"][t] = [x.to_dict() for x in items]
            if len(items) == 0:
                company_results["empty_tickers"].append(t)
            print(f"[company_news] {t}: {len(items)} items")
        except Exception as e:
            company_results["errors"][t] = str(e)
            print(f"[company_news] {t}: ERROR -> {e}")
        time.sleep(args.sleep)

    save_json(f"{args.out_prefix}_company_news.json", company_results)

    # 2) Market news (general etc.)
    market_results: Dict[str, Any] = {
        "categories": args.market_categories,
        "items_by_category": {},
        "errors": {},
    }

    all_market_items: List[NewsItem] = []
    for cat in args.market_categories:
        try:
            items = fetch_market_news(cat, api_key, min_id=0)
            market_results["items_by_category"][cat] = [x.to_dict() for x in items]
            all_market_items.extend(items)
            print(f"[market_news] {cat}: {len(items)} items")
        except Exception as e:
            market_results["errors"][cat] = str(e)
            print(f"[market_news] {cat}: ERROR -> {e}")
        time.sleep(args.sleep)

    save_json(f"{args.out_prefix}_market_news.json", market_results)

    # 3) Simple filter: which market news mentions our tickers?
    filtered = simple_ticker_filter(all_market_items, tickers) if tickers else []
    filtered_obj = {
        "tickers": tickers,
        "note": "Simple keyword filter (headline/summary/related). LLM mapping will be better.",
        "count_all_market": len(all_market_items),
        "count_filtered": len(filtered),
        "items": [x.to_dict() for x in filtered],
    }
    save_json(f"{args.out_prefix}_market_filtered.json", filtered_obj)

    print("\nSaved:")
    print(f"  - {args.out_prefix}_company_news.json")
    print(f"  - {args.out_prefix}_market_news.json")
    print(f"  - {args.out_prefix}_market_filtered.json")

    if company_results["empty_tickers"]:
        print("\n company_news returned EMPTY for these tickers (possible non-NA coverage / no news in window):")
        print(company_results["empty_tickers"])


if __name__ == "__main__":
    main()
