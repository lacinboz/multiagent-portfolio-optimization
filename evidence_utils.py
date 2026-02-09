# evidence_utils.py
from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

def parse_dt(x: Any) -> Optional[datetime]:
    if not x:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, (int, float)):
        try:
            if x > 0:
                return datetime.fromtimestamp(float(x), tz=timezone.utc)
        except Exception:
            return None

    s = str(x).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1]
        return datetime.fromisoformat(s)
    except Exception:
        return None


def assign_evidence_ids_and_map(
    items: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Stable evidence_id strategy (deterministic):

    - Stable field: url
    - Stable key: sha1(url)[:8]
    - Evidence id: f"{TICKER}_{stable_key}"
    - Deterministic ordering WITHIN a ticker: datetime DESC (newest first)
    """

    def stable_url_key(it: Dict[str, Any]) -> str:
        url = str(it.get("url") or "").strip()
        if url:
            base = url
        else:
            headline = str(it.get("headline") or "").strip()
            date = str(it.get("date") or it.get("datetime") or "").strip()
            fid = str(it.get("id") or "").strip()
            base = f"NO_URL|{headline}|{date}|{fid}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]

    tickers_seen: List[str] = []
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for it in (items or []):
        t = str(it.get("ticker") or "").upper().strip() or "UNK"
        if t not in buckets:
            tickers_seen.append(t)
        buckets[t].append(it)

    out: List[Dict[str, Any]] = []
    evidence_map: Dict[str, Dict[str, Any]] = {}

    for t in tickers_seen:
        its = buckets.get(t) or []

        decorated = []
        for it in its:
            dt = parse_dt(it.get("datetime"))
            dt_sort = dt if dt is not None else datetime.min.replace(tzinfo=timezone.utc)
            url = str(it.get("url") or "").strip()
            headline = str(it.get("headline") or "").strip()
            fid = str(it.get("id") or "").strip()
            decorated.append((dt_sort, url, headline, fid, it))

        decorated.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)

        seen_eids: set[str] = set()

        for dt_sort, url, headline, fid, it in decorated:
            sk = stable_url_key(it)
            eid = f"{t}_{sk}"
            if eid in seen_eids:
                continue
            seen_eids.add(eid)

            cp = dict(it)
            cp["evidence_id"] = eid
            out.append(cp)

            evidence_map[eid] = {
                "ticker": t,
                "headline": cp.get("headline"),
                "source": cp.get("source") or cp.get("provider"),
                "date": cp.get("date"),
                "url": cp.get("url"),
                "finnhub_id": cp.get("id"),
            }

    return out, evidence_map

def filter_news_by_evidence_ids(news_items: list[dict], evidence_ids: set[str]) -> list[dict]:
    if not evidence_ids:
        return []
    out = []
    for it in news_items:
        eid = it.get("evidence_id") or it.get("evidenceId")  # hangi alan varsa
        if eid in evidence_ids:
            out.append(it)
    return out
