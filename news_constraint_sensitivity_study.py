# news_constraint_sensitivity_study.py
# ============================================================
# Hocamın notu: "Prob model can run multiple theses → variance"
# Farklı threshold ve delta değerleriyle constraint sensitivity analizi.
# Mevcut kodlara dokunmaz. Standalone script.
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_PATH = Path("data/news_prediction/latest_news_prediction_signals.csv")
MU_PATH = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH = Path("data/processed_yahoo/cov_annual.csv")

# ============================================================
# Portfolio optimization (standalone, no dashboard dependency)
# ============================================================

def _near_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    return (vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T)


def _load_mu_cov(tickers: List[str]):
    summary = pd.read_csv(MU_PATH, index_col=0)
    cov_df = pd.read_csv(COV_PATH, index_col=0)
    common = [t for t in tickers if t in summary.index and t in cov_df.index]
    mu = summary.loc[common, "mu_annual"].astype(float)
    cov = cov_df.loc[common, common].astype(float)
    return mu, cov


def _optimize(mu: pd.Series, cov: pd.DataFrame, rf: float, w_max: float,
               lambda_l2: float, constraints: List) -> Optional[Dict]:
    tickers = list(mu.index)
    n = len(tickers)
    effective_w_max = max(w_max, 1.0 / n + 1e-6)

    cov_np = cov.values.copy()
    if np.linalg.eigvalsh(cov_np).min() < 0:
        cov_np = _near_psd(cov_np)
    cov_fixed = pd.DataFrame(cov_np, index=tickers, columns=tickers)

    bounds = [(0.0, effective_w_max)] * n
    eq_con = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    all_cons = eq_con + constraints

    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov_fixed.values @ w))
        return -(r - rf) / v if v > 0 else np.inf

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=all_cons)
    if not res.success:
        res = minimize(neg_sharpe, w0, method="trust-constr",
                       bounds=bounds, constraints=all_cons)

    w = pd.Series(res.x, index=tickers)
    r = float(w.values @ mu.values)
    v = float(np.sqrt(w.values @ cov_fixed.values @ w.values))
    sharpe = (r - rf) / v if v > 0 else 0.0

    return {
        "weights": {t: float(w[t]) for t in tickers},
        "return": r,
        "vol": v,
        "sharpe": sharpe,
        "success": bool(res.success),
    }


# ============================================================
# Constraint builder (standalone copy, no import needed)
# ============================================================

def _build_constraints(
    latest_signals: pd.DataFrame,
    baseline_weights: Dict[str, float],
    bullish_threshold: float,
    bearish_threshold: float,
    delta: float,
    w_max: float,
) -> Dict[str, Any]:
    constraints = {}
    for _, row in latest_signals.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        if ticker not in baseline_weights:
            continue
        prob = float(row["predicted_positive_probability"])
        base_w = float(baseline_weights[ticker])

        if prob >= bullish_threshold:
            min_w = min(base_w + delta, w_max - 1e-4)
            constraints[ticker] = {"type": "bullish", "prob": prob,
                                   "min_weight": min_w}
        elif prob <= bearish_threshold:
            max_w = max(0.0, base_w - delta)
            constraints[ticker] = {"type": "bearish", "prob": prob,
                                   "max_weight": max_w}
    return constraints


def _constraints_to_scipy(
    news_constraints: Dict[str, Any],
    ticker_to_idx: Dict[str, int],
) -> List:
    sci = []
    for ticker, cdict in news_constraints.items():
        if ticker not in ticker_to_idx:
            continue
        idx = ticker_to_idx[ticker]
        if "min_weight" in cdict:
            mw = float(cdict["min_weight"])
            sci.append({"type": "ineq", "fun": lambda w, i=idx, m=mw: w[i] - m})
        if "max_weight" in cdict:
            mw = float(cdict["max_weight"])
            sci.append({"type": "ineq", "fun": lambda w, i=idx, m=mw: m - w[i]})
    return sci


# ============================================================
# Sensitivity grid
# ============================================================

# Hangi parametreleri değiştiriyoruz?
BULLISH_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]   # default: 0.60
BEARISH_THRESHOLDS = [0.30, 0.35, 0.40, 0.45]   # default: 0.40
DELTAS = [0.01, 0.02, 0.03, 0.05]               # default: 0.02


def run_constraint_sensitivity_study(
    signals_path: Path = SIGNALS_PATH,
    tickers: Optional[List[str]] = None,
    rf: float = 0.02,
    w_max: float = 0.30,
    lambda_l2: float = 1e-3,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("NEWS CONSTRAINT SENSITIVITY STUDY")
    print("=" * 70)

    if not signals_path.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_path}")

    latest_signals = pd.read_csv(signals_path)
    latest_signals["ticker"] = latest_signals["ticker"].astype(str).str.upper().str.strip()

    if tickers is None:
        tickers = latest_signals["ticker"].unique().tolist()

    print(f"Tickers: {tickers}")
    print(f"Signals loaded: {len(latest_signals)} rows")

    # Load mu / cov
    mu, cov = _load_mu_cov(tickers)
    tickers = list(mu.index)  # filtered to common
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    print(f"After mu/cov filter: {tickers}")

    # ── Step 1: Baseline (no news constraints) ─────────────────
    baseline_res = _optimize(mu, cov, rf, w_max, lambda_l2, constraints=[])
    baseline_weights = baseline_res["weights"]

    print(f"\nBaseline: return={baseline_res['return']*100:.2f}% "
          f"vol={baseline_res['vol']*100:.2f}% sharpe={baseline_res['sharpe']:.4f}")
    print(f"Baseline weights: { {t: f'{v:.1%}' for t, v in baseline_weights.items()} }")

    # ── Step 2: Show current signal distribution ───────────────
    sig_filtered = latest_signals[latest_signals["ticker"].isin(tickers)].copy()
    print(f"\nSignal distribution for these tickers:")
    for _, row in sig_filtered.iterrows():
        p = float(row["predicted_positive_probability"])
        label = "BULLISH" if p >= 0.60 else ("BEARISH" if p <= 0.40 else "neutral")
        print(f"  {row['ticker']}: p={p:.3f} [{label}]")

    # ── Step 3: Grid search ────────────────────────────────────
    rows = []

    for bull_thr in BULLISH_THRESHOLDS:
        for bear_thr in BEARISH_THRESHOLDS:
            if bear_thr >= bull_thr:
                continue  # infeasible threshold pair
            for delta in DELTAS:

                news_constraints = _build_constraints(
                    latest_signals=sig_filtered,
                    baseline_weights=baseline_weights,
                    bullish_threshold=bull_thr,
                    bearish_threshold=bear_thr,
                    delta=delta,
                    w_max=w_max,
                )

                sci_constraints = _constraints_to_scipy(news_constraints, ticker_to_idx)

                try:
                    res = _optimize(mu, cov, rf, w_max, lambda_l2, sci_constraints)
                except Exception as e:
                    print(f"  FAILED bull={bull_thr} bear={bear_thr} delta={delta}: {e}")
                    continue

                n_bullish = sum(1 for c in news_constraints.values() if c["type"] == "bullish")
                n_bearish = sum(1 for c in news_constraints.values() if c["type"] == "bearish")
                n_constrained = n_bullish + n_bearish

                # weight changes vs baseline
                w_delta = {t: res["weights"].get(t, 0) - baseline_weights.get(t, 0)
                           for t in tickers}
                max_w_change = max(abs(v) for v in w_delta.values())
                turnover = sum(abs(v) for v in w_delta.values()) / 2.0

                sharpe_delta = res["sharpe"] - baseline_res["sharpe"]
                return_delta = res["return"] - baseline_res["return"]
                vol_delta = res["vol"] - baseline_res["vol"]

                rows.append({
                    "bullish_threshold": bull_thr,
                    "bearish_threshold": bear_thr,
                    "delta": delta,
                    "n_constrained": n_constrained,
                    "n_bullish": n_bullish,
                    "n_bearish": n_bearish,
                    "return": res["return"],
                    "vol": res["vol"],
                    "sharpe": res["sharpe"],
                    "return_pct": res["return"] * 100,
                    "vol_pct": res["vol"] * 100,
                    "sharpe_delta": sharpe_delta,
                    "return_delta_pct": return_delta * 100,
                    "vol_delta_pct": vol_delta * 100,
                    "max_weight_change": max_w_change,
                    "turnover": turnover,
                    "optimizer_success": res["success"],
                    **{f"w_{t}": res["weights"].get(t, 0) for t in tickers},
                })

    df = pd.DataFrame(rows)

    # ── Step 4: Print summary ──────────────────────────────────
    print("\n" + "=" * 70)
    print("SENSITIVITY RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total configs tested: {len(df)}")
    print(f"All succeeded: {df['optimizer_success'].all()}")

    print(f"\nSharpe delta range: "
          f"{df['sharpe_delta'].min():.4f} to {df['sharpe_delta'].max():.4f}")
    print(f"Turnover range: "
          f"{df['turnover'].min():.4f} to {df['turnover'].max():.4f}")
    print(f"Return delta range: "
          f"{df['return_delta_pct'].min():.2f}% to {df['return_delta_pct'].max():.2f}%")

    # Most sensitive parameter
    print("\n--- Effect of delta (averaged over thresholds) ---")
    delta_summary = df.groupby("delta")[["sharpe_delta", "turnover", "return_delta_pct"]].mean()
    print(delta_summary.round(4))

    print("\n--- Effect of bullish_threshold (averaged over delta) ---")
    bull_summary = df.groupby("bullish_threshold")[["sharpe_delta", "n_bullish", "turnover"]].mean()
    print(bull_summary.round(4))

    print("\n--- Effect of bearish_threshold (averaged over delta) ---")
    bear_summary = df.groupby("bearish_threshold")[["sharpe_delta", "n_bearish", "turnover"]].mean()
    print(bear_summary.round(4))

    # Best and worst configs
    best = df.loc[df["sharpe_delta"].idxmax()]
    worst = df.loc[df["sharpe_delta"].idxmin()]
    print(f"\nBest config (max Sharpe delta={best['sharpe_delta']:.4f}): "
          f"bull={best['bullish_threshold']} bear={best['bearish_threshold']} delta={best['delta']}")
    print(f"Worst config (min Sharpe delta={worst['sharpe_delta']:.4f}): "
          f"bull={worst['bullish_threshold']} bear={worst['bearish_threshold']} delta={worst['delta']}")

    # Variance across configs — key thesis metric
    print(f"\n--- Variance across all configs ---")
    print(f"Sharpe std: {df['sharpe_delta'].std():.4f}")
    print(f"Turnover std: {df['turnover'].std():.4f}")
    print(f"Return delta std: {df['return_delta_pct'].std():.4f}%")

    if save_outputs:
        csv_path = OUT_DIR / "constraint_sensitivity_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        summary = {
            "baseline": {
                "return": baseline_res["return"],
                "vol": baseline_res["vol"],
                "sharpe": baseline_res["sharpe"],
                "weights": baseline_weights,
            },
            "n_configs": len(df),
            "sharpe_delta": {
                "mean": float(df["sharpe_delta"].mean()),
                "std": float(df["sharpe_delta"].std()),
                "min": float(df["sharpe_delta"].min()),
                "max": float(df["sharpe_delta"].max()),
            },
            "turnover": {
                "mean": float(df["turnover"].mean()),
                "std": float(df["turnover"].std()),
                "min": float(df["turnover"].min()),
                "max": float(df["turnover"].max()),
            },
            "return_delta_pct": {
                "mean": float(df["return_delta_pct"].mean()),
                "std": float(df["return_delta_pct"].std()),
            },
            "best_config": {
                "bullish_threshold": float(best["bullish_threshold"]),
                "bearish_threshold": float(best["bearish_threshold"]),
                "delta": float(best["delta"]),
                "sharpe_delta": float(best["sharpe_delta"]),
            },
        }

        json_path = OUT_DIR / "constraint_sensitivity_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Saved] {json_path}")

    return {"df": df, "baseline": baseline_res, "summary": summary}


if __name__ == "__main__":
    # Dashboard'daki 4 ticker ile çalıştır
    run_constraint_sensitivity_study(
        tickers=["AVGO", "GOOGL", "MU", "NVDA"],
        rf=0.02,
        w_max=0.30,
        save_outputs=True,
    )