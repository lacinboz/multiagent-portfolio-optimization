"""
Mode A Ablation Study — Parameter Sensitivity for α, β, t½
============================================================
FIXED VERSION: FinBERT runs ONCE on the full news cache. The 120-point
grid search then only repeats the cheap, parameter-dependent aggregation
(recency weighting, confidence, mu/cov adjustment) and re-optimization.

Usage:
    python mode_a_ablation.py
"""
from __future__ import annotations
from scipy.optimize import minimize
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from agents_langgraph import data_agent_get_mu_cov
    from portfolio_core import run_portfolio_optimization
    import probabilistic_news_integration as pni
except ImportError as exc:
    print(f"[ABLATION] Import error: {exc}")
    sys.exit(1)

OUT_DIR = Path("data/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed_yahoo")

# ═════════════════════════════════════════════════════════════════════════
# 1. GRID
# ═════════════════════════════════════════════════════════════════════════
ALPHA_GRID     = [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]
BETA_GRID      = [0.10, 0.20, 0.35, 0.50, 0.65]
HALF_LIFE_GRID = [1.0, 2.0, 5.0, 10.0]

RF     = 0.02
W_MAX  = 0.30
LAMBDA = 1e-3

# ═════════════════════════════════════════════════════════════════════════
# 2. BASE DATA / BASE PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════
def load_base_data():
    mu, cov = data_agent_get_mu_cov(
        list(pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0).index)
    )
    return mu, cov, list(mu.index)


def run_base_portfolio(mu, cov):
    res = run_portfolio_optimization(
        mu=mu, cov=cov, rf=RF, w_max=W_MAX, lambda_l2=LAMBDA,
        data_dir=DATA_DIR, save_csv=False,
    )
    return res.get("maxsharpe", {})

# ═════════════════════════════════════════════════════════════════════════
# 3. NEWS LOADING (from disk cache)
# ═════════════════════════════════════════════════════════════════════════
def load_production_scale_news(tickers, lookback_days=7, max_items_per_ticker=20):
    from agents_langgraph import news_agent_fetch_for_tickers
    fetched = news_agent_fetch_for_tickers(
        tickers=tickers,
        include_news=True,
        lookback_days=lookback_days,
        min_company_items=1,
        max_items_per_ticker=max_items_per_ticker,
        include_market_fallback=True,
        market_category="general",
        cache_ttl_s=86400,   # cache for the whole ablation run
        sleep_s=0.25,
    )
    flat_items = (fetched or {}).get("flat_items") or []
    print(f"[ABLATION] Loaded {len(flat_items)} news items "
          f"(production-scale: lookback={lookback_days}d, max={max_items_per_ticker}/ticker)")
    return flat_items
def load_cached_news(tickers):
    cache_dir = Path("data/news_cache")
    if not cache_dir.exists():
        return _synthetic_news(tickers)

    flat = []
    ticker_set = {t.upper() for t in tickers}
    for f in sorted(cache_dir.glob("finnhub_company-news_*.json")):
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            items = obj.get("data", [])
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                related = str(it.get("related") or "").upper()
                headline = str(it.get("headline") or "").upper()
                matched = next((t for t in ticker_set if t in related or t in headline), None)
                if matched:
                    cp = dict(it)
                    cp["ticker"] = matched
                    flat.append(cp)
        except Exception:
            continue

    if not flat:
        return _synthetic_news(tickers)

    print(f"[ABLATION] Loaded {len(flat)} news items from disk cache.")
    return flat

def run_maxsharpe_only(mu, cov, rf=RF, w_max=W_MAX, lambda_l2=LAMBDA):
    tickers = list(mu.index)
    cov = cov.loc[tickers, tickers].astype(float)
    mu = mu.astype(float)
    n = len(tickers)

    eigvals = np.linalg.eigvalsh(cov.values)
    if eigvals.min() < 0:
        from portfolio_core import near_psd
        cov = pd.DataFrame(near_psd(cov.values), index=tickers, columns=tickers)

    eff_w_max = max(w_max, 1.0 / n + 1e-6)
    bounds = [(0.0, eff_w_max)] * n
    w0 = np.full(n, 1 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def obj_neg_sharpe(w):
        from portfolio_core import sharpe_ratio
        return -sharpe_ratio(w, mu, cov, rf=rf)

    res = minimize(obj_neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    from portfolio_core import portfolio_stats
    r, v = portfolio_stats(res.x, mu, cov)
    return {"return": r, "vol": v, "sharpe": (r - rf) / v if v > 0 else float("nan"),
            "weights": {t: float(res.x[i]) for i, t in enumerate(tickers)}}

def _synthetic_news(tickers):
    import hashlib, time
    headlines = {
        "positive": ["{t} beats earnings, raises guidance",
                     "{t} announces strategic deal; shares rise",
                     "{t} reports record quarterly revenue"],
        "negative": ["{t} misses revenue forecast; outlook cut",
                     "{t} faces regulatory probe",
                     "{t} warns of supply chain headwinds"],
        "neutral":  ["{t} holds annual shareholder meeting",
                     "{t} announces leadership transition",
                     "{t} provides mid-year operational update"],
    }
    sentiments = ["positive", "negative", "neutral"]
    now = int(time.time())
    items = []
    for t in tickers:
        h = int(hashlib.sha1(t.encode()).hexdigest(), 16)
        sent = sentiments[h % 3]
        for j in range(1 + h % 3):
            hl = headlines[sent][(h + j) % 3].format(t=t)
            items.append({
                "ticker": t, "headline": hl, "summary": hl + ".",
                "source": "synthetic", "datetime": now - j * 3600 * 6,
                "url": f"https://example.com/{t.lower()}/{j}",
            })
    print(f"[ABLATION] Generated {len(items)} synthetic news items.")
    return items

# ═════════════════════════════════════════════════════════════════════════
# 4. ONE-TIME FINBERT PRECOMPUTE
#    Computes everything that does NOT depend on alpha/beta/half_life:
#    FinBERT sentiment, model confidence, source credibility, richness.
# ═════════════════════════════════════════════════════════════════════════
def precompute_article_base_signals(news_raw, tickers, model_name=pni.FINBERT_MODEL_NAME):
    normalized = pni.normalize_news_items(news_raw)
    allowed = {pni._normalize_ticker(t) for t in tickers}
    filtered = [x for x in normalized if x["ticker"] in allowed]

    print(f"[PRECOMPUTE] {len(filtered)} / {len(normalized)} news items match selected tickers.")
    if not filtered:
        return []

    print("[PRECOMPUTE] Loading FinBERT and scoring articles (one-time cost)…")
    scorer = pni.FinBERTScorer(model_name=model_name)

    texts = []
    for a in filtered:
        headline = a["headline"]
        summary = a["summary"]
        texts.append(f"{headline} [SEP] {summary}" if summary else headline)

    probs_list = scorer.score_texts(texts=texts, batch_size=16)

    base_signals = []
    for art, probs in zip(filtered, probs_list):
        positive = float(probs.get("positive", 0.0))
        negative = float(probs.get("negative", 0.0))
        neutral = float(probs.get("neutral", 0.0))
        sentiment = max(-1.0, min(1.0, positive - negative))
        model_conf = max(positive, negative, neutral)

        base_signals.append({
            "ticker": art["ticker"],
            "datetime": art["datetime"],
            "sentiment": sentiment,
            "model_conf": model_conf,
            "source_conf": pni._source_credibility(art["source"]),
            "richness": pni._content_richness(art),
        })

    print(f"[PRECOMPUTE] Done. {len(base_signals)} article-level signals cached.")
    return base_signals

# ═════════════════════════════════════════════════════════════════════════
# 5. FAST PER-GRID-POINT AGGREGATION (no FinBERT, no network)
#    Re-applies recency weighting (half_life), aggregates to ticker level,
#    and adjusts mu/cov with alpha/beta.
# ═════════════════════════════════════════════════════════════════════════
def fast_adjust_inputs(mu, cov, tickers, base_signals, *, alpha, beta, half_life):
    now = pni._now_utc()
    grouped: Dict[str, List[tuple]] = {}

    for a in base_signals:
        recency = pni._recency_weight(a["datetime"], half_life_days=half_life, now=now)
        confidence = (
            0.45 * a["model_conf"]
            + 0.20 * a["source_conf"]
            + 0.20 * recency
            + 0.15 * a["richness"]
        )
        confidence = max(0.0, min(1.0, confidence))
        combined_weight = max(1e-6, 0.5 * confidence + 0.5 * recency)

        grouped.setdefault(a["ticker"], []).append((a["sentiment"], confidence, combined_weight))

    ticker_signals: Dict[str, pni.TickerNewsSignal] = {}

    for t in tickers:
        tt = pni._normalize_ticker(t)
        items = grouped.get(tt)

        if not items:
            ticker_signals[t] = pni.TickerNewsSignal(
                ticker=t, sentiment_score=0.0, confidence_score=0.0,
                weighted_article_count=0.0, raw_article_count=0, sentiment_variance=0.0,
            )
            continue

        sentiments = np.array([x[0] for x in items], dtype=float)
        confidences = np.array([x[1] for x in items], dtype=float)
        weights = np.array([x[2] for x in items], dtype=float)

        weighted_count = float(weights.sum())
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights_norm = weights / weights.sum()

        sentiment_score = float(np.sum(weights_norm * sentiments))
        confidence_score = float(np.sum(weights_norm * confidences))
        sentiment_variance = float(np.sum(weights_norm * (sentiments - sentiment_score) ** 2))

        ticker_signals[t] = pni.TickerNewsSignal(
            ticker=t,
            sentiment_score=max(-1.0, min(1.0, sentiment_score)),
            confidence_score=max(0.0, min(1.0, confidence_score)),
            weighted_article_count=weighted_count,
            raw_article_count=len(items),
            sentiment_variance=max(0.0, sentiment_variance),
        )

    adjusted_mu = pni.adjust_expected_returns(mu, ticker_signals, alpha=alpha, power=1.5)
    adjusted_cov = pni.adjust_covariance_matrix(cov, ticker_signals, beta=beta)

    return adjusted_mu, adjusted_cov, ticker_signals

# ═════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═════════════════════════════════════════════════════════════════════════
def _safe(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def portfolio_metrics(d, *, rf=RF):
    r = _safe(d.get("return")) or 0.0
    v = _safe(d.get("vol")) or 1e-9
    s = _safe(d.get("sharpe"))
    if s is None:
        s = (r - rf) / v if v > 0 else float("nan")
    return {"return": r, "vol": v, "sharpe": s}


def turnover(w_base, w_news):
    tickers = set(w_base) | set(w_news)
    return 0.5 * sum(abs(w_news.get(t, 0.0) - w_base.get(t, 0.0)) for t in tickers)

# ═════════════════════════════════════════════════════════════════════════
# 7. SINGLE GRID POINT (fast)
# ═════════════════════════════════════════════════════════════════════════
def evaluate_point(mu, cov, tickers, base_signals, base_port, *, alpha, beta, half_life):
    try:
        mu_adj, cov_adj, ticker_signals = fast_adjust_inputs(
            mu, cov, tickers, base_signals, alpha=alpha, beta=beta, half_life=half_life
        )

        news_port = run_maxsharpe_only(mu_adj, cov_adj, rf=RF, w_max=W_MAX, lambda_l2=LAMBDA)

        bm = portfolio_metrics(base_port)
        nm = portfolio_metrics(news_port)
        to = turnover(base_port.get("weights") or {}, news_port.get("weights") or {})

        n_signalled = sum(
            1 for s in ticker_signals.values()
            if abs(s.sentiment_score) > 0.05
        )

        return {
            "alpha": alpha, "beta": beta, "half_life": half_life,
            "sharpe_base":  round(bm["sharpe"], 5),
            "sharpe_news":  round(nm["sharpe"], 5),
            "delta_sharpe": round(nm["sharpe"] - bm["sharpe"], 5),
            "return_base":  round(bm["return"], 5),
            "return_news":  round(nm["return"], 5),
            "delta_return": round(nm["return"] - bm["return"], 5),
            "vol_base":     round(bm["vol"], 5),
            "vol_news":     round(nm["vol"], 5),
            "delta_vol":    round(nm["vol"] - bm["vol"], 5),
            "turnover":     round(to, 5),
            "n_tickers_signalled": n_signalled,
            "status": "ok",
        }

    except Exception as exc:
        return {"alpha": alpha, "beta": beta, "half_life": half_life, "status": f"error: {exc}"}

# ═════════════════════════════════════════════════════════════════════════
# 8. MAIN GRID LOOP
# ═════════════════════════════════════════════════════════════════════════
def run_ablation():
    n_total = len(ALPHA_GRID) * len(BETA_GRID) * len(HALF_LIFE_GRID)
    print("=" * 60)
    print("Mode A Ablation Study")
    print(f"  α grid:  {ALPHA_GRID}")
    print(f"  β grid:  {BETA_GRID}")
    print(f"  t½ grid: {HALF_LIFE_GRID}")
    print(f"  Total evaluations: {n_total}")
    print("=" * 60)

    print("\n[1/5] Loading base data …")
    mu, cov, tickers = load_base_data()
    print(f"      n_tickers = {len(tickers)}")

    print("[2/5] Loading news (production-scale fetch) …")
    news_raw = load_production_scale_news(tickers, lookback_days=7, max_items_per_ticker=20)

    print("[3/5] Running base portfolio …")
    base_port = run_base_portfolio(mu, cov)
    bm = portfolio_metrics(base_port)
    print(f"      Base  Sharpe={bm['sharpe']:.4f}  "
          f"Return={bm['return']*100:.2f}%  Vol={bm['vol']*100:.2f}%")

    print("\n[4/5] Precomputing FinBERT article signals (ONCE) …")
    base_signals = precompute_article_base_signals(news_raw, tickers)

    print(f"\n[5/5] Evaluating {n_total} grid points (fast aggregation only) …\n")
    rows = []
    for i, (alpha, beta, hl) in enumerate(
            itertools.product(ALPHA_GRID, BETA_GRID, HALF_LIFE_GRID), 1):
        row = evaluate_point(mu, cov, tickers, base_signals, base_port,
                              alpha=alpha, beta=beta, half_life=hl)
        rows.append(row)
        st = row.get("status", "?")
        if st == "ok":
            print(f"  [{i:3d}/{n_total}] α={alpha:.2f}  β={beta:.2f}  t½={hl:4.1f}  "
                  f"ΔSharpe={row['delta_sharpe']:+.4f}  "
                  f"ΔRet={row['delta_return']*100:+.2f}%  "
                  f"Turn={row['turnover']*100:.2f}%")
        else:
            print(f"  [{i:3d}/{n_total}] α={alpha:.2f}  β={beta:.2f}  t½={hl:4.1f}  STATUS={st}")

    return pd.DataFrame(rows)

# ═════════════════════════════════════════════════════════════════════════
# 9. SUMMARY + SAVE  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════
def summarize(df):
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\n[ABLATION] All evaluations failed.")
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best = ok.loc[ok["delta_sharpe"].idxmax()]
    low_t = ok.loc[ok["turnover"].idxmin()]
    print(f"\nBest ΔSharpe:    α={best['alpha']:.2f}  β={best['beta']:.2f}  "
          f"t½={best['half_life']:.1f}  ΔSharpe={best['delta_sharpe']:+.4f}  "
          f"Turnover={best['turnover']*100:.2f}%")
    print(f"Lowest turnover: α={low_t['alpha']:.2f}  β={low_t['beta']:.2f}  "
          f"t½={low_t['half_life']:.1f}  "
          f"Turnover={low_t['turnover']*100:.2f}%  ΔSharpe={low_t['delta_sharpe']:+.4f}")

    def_ = ok[(ok["alpha"] == 0.08) & (ok["beta"] == 0.35) & (ok["half_life"] == 2.0)]
    if not def_.empty:
        d = def_.iloc[0]
        print("\nThesis defaults (α=0.08, β=0.35, t½=2.0):")
        print(f"  ΔSharpe={d['delta_sharpe']:+.4f}  ΔReturn={d['delta_return']*100:+.2f}%  "
              f"ΔVol={d['delta_vol']*100:+.2f}%  Turnover={d['turnover']*100:.2f}%")
    else:
        print("\n[NOTE] Thesis default (α=0.08, β=0.35, t½=2.0) not found in grid.")

    print("\nCorrelation with ΔSharpe:")
    for col in ["alpha", "beta", "half_life"]:
        print(f"  {col:12s}: r = {ok[col].corr(ok['delta_sharpe']):+.3f}")

    for param in ["alpha", "beta", "half_life"]:
        print(f"\nMarginal ΔSharpe by {param}:")
        print(ok.groupby(param)["delta_sharpe"].mean().round(4).to_string())

    print("\nMarginal Turnover by α:")
    print(ok.groupby("alpha")["turnover"].mean().map(lambda x: f"{x*100:.2f}%").to_string())


def save_results(df):
    csv_path = OUT_DIR / "mode_a_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] CSV  → {csv_path}")

    try:
        xlsx_path = OUT_DIR / "mode_a_ablation_results.xlsx"
        ok = df[df["status"] == "ok"].copy()
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Results", index=False)
            for hl in sorted(ok["half_life"].unique()):
                sub = ok[ok["half_life"] == hl]
                sub.pivot_table(index="alpha", columns="beta",
                                 values="delta_sharpe", aggfunc="first").round(4) \
                   .to_excel(writer, sheet_name=f"ΔSharpe_t½={hl:.1f}")
                (sub.pivot_table(index="alpha", columns="beta",
                                 values="turnover", aggfunc="first") * 100).round(2) \
                   .to_excel(writer, sheet_name=f"Turn%_t½={hl:.1f}")
        print(f"[SAVED] XLSX → {xlsx_path}")
    except Exception as exc:
        print(f"[SKIP] Excel: {exc}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ok = df[df["status"] == "ok"].copy()
        if ok.empty:
            return

        for hl in sorted(ok["half_life"].unique()):
            sub = ok[ok["half_life"] == hl]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Mode A Ablation — t½ = {hl:.1f} days", fontsize=13)

            for ax, metric, label, fmt in [
                (axes[0], "delta_sharpe", "ΔSharpe (news − base)", ".4f"),
                (axes[1], "turnover", "Turnover (one-way)", ".3f"),
            ]:
                pivot = sub.pivot_table(index="alpha", columns="beta",
                                         values=metric, aggfunc="first")
                im = ax.imshow(pivot.values,
                               cmap="RdYlGn" if metric == "delta_sharpe" else "YlOrRd_r",
                               aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_yticks(range(len(pivot.index)))
                ax.set_xticklabels([f"β={v}" for v in pivot.columns], rotation=45)
                ax.set_yticklabels([f"α={v}" for v in pivot.index])
                ax.set_xlabel("β")
                ax.set_ylabel("α")
                ax.set_title(label)
                plt.colorbar(im, ax=ax)
                for ii in range(len(pivot.index)):
                    for jj in range(len(pivot.columns)):
                        val = pivot.values[ii, jj]
                        if np.isfinite(val):
                            ax.text(jj, ii, f"{val:{fmt}}", ha="center", va="center",
                                    fontsize=7, color="black")
                if hl == 2.0:
                    try:
                        ri = list(pivot.index).index(0.08)
                        ci = list(pivot.columns).index(0.35)
                        ax.add_patch(plt.Rectangle((ci - .5, ri - .5), 1, 1,
                                                    fill=False, edgecolor="blue", linewidth=2.5))
                    except ValueError:
                        pass

            plt.tight_layout()
            p = OUT_DIR / f"mode_a_ablation_heatmap_hl{hl:.0f}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] PNG  → {p}")
    except ImportError:
        print("[SKIP] matplotlib not available.")
    except Exception as exc:
        print(f"[SKIP] Plot: {exc}")


if __name__ == "__main__":
    df = run_ablation()
    summarize(df)
    save_results(df)
    print("\n[DONE] Ablation complete.")