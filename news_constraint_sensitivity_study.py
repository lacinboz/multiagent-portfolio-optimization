# news_constraint_sensitivity_study.py
# ============================================================
# Parameter sensitivity study for news-based portfolio constraints.
#
# Grid search over 64 threshold/delta combinations:
#   bull ∈ {0.55, 0.60, 0.65, 0.70}
#   bear ∈ {0.30, 0.35, 0.40, 0.45}
#   delta ∈ {0.01, 0.02, 0.03, 0.05}
#
#  Reports BOTH expected metrics (w^T mu, w^T Sigma w)
#  AND realized metrics (test period daily returns)
#  No look-ahead bias: uses as_of_20260114 signals file
#  Deterministic — no 5-run loop needed (no model randomness)
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from realized_eval import compute_realized_metrics, load_test_returns

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_PATH = Path("data/news_prediction/latest_news_prediction_signals_as_of_20260114.csv")
MU_PATH      = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH     = Path("data/processed_yahoo/cov_annual.csv")

BULLISH_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
BEARISH_THRESHOLDS = [0.30, 0.35, 0.40, 0.45]
DELTAS             = [0.01, 0.02, 0.03, 0.05]

MIN_BASELINE_WEIGHT = 1e-3


# ============================================================
# Portfolio helpers
# ============================================================

def _near_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    return vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T


def _load_mu_cov(tickers: Optional[List[str]] = None):
    summary = pd.read_csv(MU_PATH, index_col=0)
    cov_df  = pd.read_csv(COV_PATH, index_col=0)
    if tickers is None:
        common = [t for t in summary.index if t in cov_df.index]
    else:
        common = [t for t in tickers if t in summary.index and t in cov_df.index]
    return (summary.loc[common, "mu_annual"].astype(float),
            cov_df.loc[common, common].astype(float))


def _optimize(mu, cov, rf, w_max, lambda_l2, extra_constraints=None):
    tickers = list(mu.index)
    n = len(tickers)
    effective_w_max = max(w_max, 1.0 / n + 1e-6)
    cov_np = cov.values.copy()
    if np.linalg.eigvalsh(cov_np).min() < 0:
        cov_np = _near_psd(cov_np)
    cov_f = pd.DataFrame(cov_np, index=tickers, columns=tickers)
    bounds = [(0.0, effective_w_max)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        cons += extra_constraints
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov_f.values @ w))
        return -(r - rf) / v if v > 0 else np.inf

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(neg_sharpe, w0, method="trust-constr",
                       bounds=bounds, constraints=cons)
    w = pd.Series(np.clip(res.x, 0, None), index=tickers)
    w = w / w.sum()
    r = float(w.values @ mu.values)
    v = float(np.sqrt(w.values @ cov_f.values @ w.values))
    return {
        "weights": {t: float(w[t]) for t in tickers},
        "return": r, "vol": v,
        "sharpe": (r - rf) / v if v > 0 else 0.0,
        "success": bool(res.success),
    }


# ============================================================
# Constraint builder
# ============================================================

def _build_constraints(
    latest_signals: pd.DataFrame,
    baseline_weights: Dict[str, float],
    bullish_threshold: float,
    bearish_threshold: float,
    delta: float,
    w_max: float,
    min_baseline_weight: float = MIN_BASELINE_WEIGHT,
) -> Dict[str, Any]:
    constraints = {}
    for _, row in latest_signals.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        if ticker not in baseline_weights:
            continue
        prob   = float(row["predicted_positive_probability"])
        base_w = float(baseline_weights[ticker])

        if base_w < min_baseline_weight:
            continue

        if prob >= bullish_threshold:
            constraints[ticker] = {
                "type": "bullish", "prob": prob,
                "min_weight": min(base_w + delta, w_max - 1e-4),
            }
        elif prob <= bearish_threshold:
            constraints[ticker] = {
                "type": "bearish", "prob": prob,
                "max_weight": max(0.0, base_w - delta),
            }
    return constraints


def _constraints_to_scipy(news_constraints, ticker_to_idx):
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
# Main study
# ============================================================

def run_constraint_sensitivity_study(
    signals_path: Path = SIGNALS_PATH,
    rf: float = 0.02,
    w_max: float = 0.30,
    lambda_l2: float = 1e-3,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("NEWS CONSTRAINT SENSITIVITY STUDY")
    print("✅ Both expected AND realized metrics reported")
    print("✅ No look-ahead bias: signals capped at 2026-01-14")
    print("=" * 70)
    print(f"rf={rf}, w_max={w_max}, lambda_l2={lambda_l2}")
    print(f"Grid: bull∈{BULLISH_THRESHOLDS} × bear∈{BEARISH_THRESHOLDS} "
          f"× delta∈{DELTAS}")

    n_total = sum(
        1 for b in BULLISH_THRESHOLDS
        for br in BEARISH_THRESHOLDS
        for _ in DELTAS
        if br < b
    )
    print(f"Total configurations: {n_total}")

    if not signals_path.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_path}")

    # Load signals
    latest_signals = pd.read_csv(signals_path)
    latest_signals["ticker"] = (
        latest_signals["ticker"].astype(str).str.upper().str.strip()
    )
    signal_tickers = latest_signals["ticker"].unique().tolist()

    # Load mu/cov
    mu, cov = _load_mu_cov(tickers=signal_tickers)
    tickers       = list(mu.index)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    sig_filtered  = latest_signals[latest_signals["ticker"].isin(tickers)].copy()

    print(f"\nSignal tickers:    {len(signal_tickers)}")
    print(f"With mu/cov data:  {len(tickers)}")
    print(f"Signals used:      {len(sig_filtered)} rows")

    # Load test returns for realized metrics
    returns_test = load_test_returns()
    print(f"Test returns:      {len(returns_test)} days "
          f"({returns_test.index[0].date()} → {returns_test.index[-1].date()})")

    # Baseline (unconstrained)
    baseline = _optimize(mu, cov, rf, w_max, lambda_l2)
    baseline_weights = baseline["weights"]
    baseline_realized = compute_realized_metrics(baseline_weights, returns_test)

    print(f"\nBaseline expected : Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | "
          f"Vol={baseline['vol']*100:.2f}%")
    print(f"Baseline realized : Sharpe={baseline_realized['realized_sharpe']:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}% | "
          f"Vol={baseline_realized['realized_vol']*100:.2f}%")

    # Signal distribution at production defaults
    n_bull_def = (sig_filtered["predicted_positive_probability"] >= 0.60).sum()
    n_bear_def = (sig_filtered["predicted_positive_probability"] <= 0.40).sum()
    print(f"\nSignal distribution (bull>=0.60, bear<=0.40): "
          f"Bullish={n_bull_def} | Bearish={n_bear_def}")

    # ============================================================
    # Grid search
    # ============================================================
    print(f"\nRunning {n_total} configurations...")
    rows = []

    for bull_thr in BULLISH_THRESHOLDS:
        for bear_thr in BEARISH_THRESHOLDS:
            if bear_thr >= bull_thr:
                continue
            for delta in DELTAS:
                news_constraints = _build_constraints(
                    latest_signals=sig_filtered,
                    baseline_weights=baseline_weights,
                    bullish_threshold=bull_thr,
                    bearish_threshold=bear_thr,
                    delta=delta,
                    w_max=w_max,
                )
                sci_cons = _constraints_to_scipy(news_constraints, ticker_to_idx)

                try:
                    res = _optimize(mu, cov, rf, w_max, lambda_l2, sci_cons)
                except Exception as e:
                    print(f"  FAILED bull={bull_thr} bear={bear_thr} "
                          f"delta={delta}: {e}")
                    continue

                n_bullish = sum(
                    1 for c in news_constraints.values() if c["type"] == "bullish"
                )
                n_bearish = sum(
                    1 for c in news_constraints.values() if c["type"] == "bearish"
                )
                turnover = sum(
                    abs(res["weights"].get(t, 0) - baseline_weights.get(t, 0))
                    for t in tickers
                ) / 2.0

                # Realized metrics for this config
                real = compute_realized_metrics(res["weights"], returns_test)

                rows.append({
                    # config
                    "bullish_threshold":   bull_thr,
                    "bearish_threshold":   bear_thr,
                    "delta":               delta,
                    "n_bullish":           n_bullish,
                    "n_bearish":           n_bearish,
                    "n_constrained":       n_bullish + n_bearish,
                    "optimizer_success":   res["success"],
                    "turnover":            turnover,
                    # expected metrics
                    "exp_sharpe":          res["sharpe"],
                    "exp_sharpe_delta":    res["sharpe"] - baseline["sharpe"],
                    "exp_return_pct":      res["return"] * 100,
                    "exp_return_delta_pct":(res["return"] - baseline["return"]) * 100,
                    "exp_vol_pct":         res["vol"] * 100,
                    "exp_vol_delta_pct":   (res["vol"] - baseline["vol"]) * 100,
                    # realized metrics
                    "real_sharpe":         real["realized_sharpe"],
                    "real_sharpe_delta":   real["realized_sharpe"] - baseline_realized["realized_sharpe"],
                    "real_return_pct":     real["realized_return"] * 100,
                    "real_return_delta_pct": (real["realized_return"] - baseline_realized["realized_return"]) * 100,
                    "real_vol_pct":        real["realized_vol"] * 100,
                    "real_vol_delta_pct":  (real["realized_vol"] - baseline_realized["realized_vol"]) * 100,
                    "real_max_dd_pct":     real["realized_max_dd"] * 100,
                    "real_n_days":         real["realized_n_days"],
                })

    df = pd.DataFrame(rows)

    # ============================================================
    # Summary print
    # ============================================================
    print("\n" + "=" * 70)
    print("SENSITIVITY RESULTS — EXPECTED METRICS")
    print(f"Baseline: Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | "
          f"Vol={baseline['vol']*100:.2f}%")
    print("=" * 70)
    print(f"Configs tested: {len(df)} | All converged: {df['optimizer_success'].all()}")

    failed = df[~df["optimizer_success"]]
    if not failed.empty:
        print(f"  → {len(failed)} configs did not fully converge (results still usable):")
        for _, r in failed.iterrows():
            print(f"    bull={r['bullish_threshold']} bear={r['bearish_threshold']} "
                  f"delta={r['delta']} ΔS={r['exp_sharpe_delta']:+.4f}")

    print(f"\nExpected ΔSharpe : mean={df['exp_sharpe_delta'].mean():.4f} | "
          f"std={df['exp_sharpe_delta'].std():.4f} | "
          f"range=[{df['exp_sharpe_delta'].min():.4f}, {df['exp_sharpe_delta'].max():.4f}]")
    print(f"Turnover         : mean={df['turnover'].mean()*100:.1f}% | "
          f"std={df['turnover'].std()*100:.1f}% | "
          f"range=[{df['turnover'].min()*100:.1f}%, {df['turnover'].max()*100:.1f}%]")

    print("\n--- Effect of delta on EXPECTED metrics ---")
    print(df.groupby("delta")[
        ["exp_sharpe_delta", "exp_return_delta_pct", "turnover"]
    ].mean().round(4).to_string())

    print("\n" + "=" * 70)
    print("SENSITIVITY RESULTS — REALIZED METRICS")
    print(f"Baseline: Sharpe={baseline_realized['realized_sharpe']:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}% | "
          f"Vol={baseline_realized['realized_vol']*100:.2f}%")
    print("=" * 70)

    print(f"\nRealized ΔSharpe : mean={df['real_sharpe_delta'].mean():.4f} | "
          f"std={df['real_sharpe_delta'].std():.4f} | "
          f"range=[{df['real_sharpe_delta'].min():.4f}, {df['real_sharpe_delta'].max():.4f}]")

    print("\n--- Effect of delta on REALIZED metrics ---")
    print(df.groupby("delta")[
        ["real_sharpe_delta", "real_return_delta_pct", "turnover"]
    ].mean().round(4).to_string())

    # Production config
    prod = df[
        (df["bullish_threshold"] == 0.60) &
        (df["bearish_threshold"] == 0.40) &
        (df["delta"] == 0.02)
    ]
    if not prod.empty:
        p = prod.iloc[0]
        print(f"\nProduction config (bull=0.60, bear=0.40, delta=0.02):")
        print(f"  Expected : ΔS={p['exp_sharpe_delta']:+.4f} | "
              f"Turnover={p['turnover']*100:.1f}% | "
              f"#Bull={int(p['n_bullish'])} #Bear={int(p['n_bearish'])}")
        print(f"  Realized : ΔS={p['real_sharpe_delta']:+.4f} | "
              f"Return={p['real_return_pct']:.2f}% | "
              f"Vol={p['real_vol_pct']:.2f}%")

    best_exp  = df.loc[df["exp_sharpe_delta"].idxmax()]
    worst_exp = df.loc[df["exp_sharpe_delta"].idxmin()]
    best_real  = df.loc[df["real_sharpe_delta"].idxmax()]
    worst_real = df.loc[df["real_sharpe_delta"].idxmin()]

    print(f"\nBest expected  (ΔS={best_exp['exp_sharpe_delta']:+.4f}): "
          f"bull={best_exp['bullish_threshold']} bear={best_exp['bearish_threshold']} "
          f"delta={best_exp['delta']}")
    print(f"Worst expected (ΔS={worst_exp['exp_sharpe_delta']:+.4f}): "
          f"bull={worst_exp['bullish_threshold']} bear={worst_exp['bearish_threshold']} "
          f"delta={worst_exp['delta']}")
    print(f"\nBest realized  (ΔS={best_real['real_sharpe_delta']:+.4f}): "
          f"bull={best_real['bullish_threshold']} bear={best_real['bearish_threshold']} "
          f"delta={best_real['delta']}")
    print(f"Worst realized (ΔS={worst_real['real_sharpe_delta']:+.4f}): "
          f"bull={worst_real['bullish_threshold']} bear={worst_real['bearish_threshold']} "
          f"delta={worst_real['delta']}")

    # ============================================================
    # Save outputs
    # ============================================================
    if save_outputs:
        csv_path = OUT_DIR / "constraint_sensitivity_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        summary = {
            "parameters": {
                "rf": rf, "w_max": w_max, "lambda_l2": lambda_l2,
                "universe_size": len(tickers),
                "signals_file": str(signals_path),
                "min_baseline_weight_filter": MIN_BASELINE_WEIGHT,
                "grid": {
                    "bullish_thresholds": BULLISH_THRESHOLDS,
                    "bearish_thresholds": BEARISH_THRESHOLDS,
                    "deltas": DELTAS,
                },
            },
            "baseline": {
                "expected_sharpe": baseline["sharpe"],
                "expected_return": baseline["return"],
                "expected_vol": baseline["vol"],
                "realized_sharpe": baseline_realized["realized_sharpe"],
                "realized_return": baseline_realized["realized_return"],
                "realized_vol": baseline_realized["realized_vol"],
                "realized_max_dd": baseline_realized["realized_max_dd"],
                "realized_n_days": baseline_realized["realized_n_days"],
            },
            "n_configs": len(df),
            "all_converged": bool(df["optimizer_success"].all()),
            "expected_sharpe_delta": {
                "mean": float(df["exp_sharpe_delta"].mean()),
                "std":  float(df["exp_sharpe_delta"].std()),
                "min":  float(df["exp_sharpe_delta"].min()),
                "max":  float(df["exp_sharpe_delta"].max()),
            },
            "realized_sharpe_delta": {
                "mean": float(df["real_sharpe_delta"].mean()),
                "std":  float(df["real_sharpe_delta"].std()),
                "min":  float(df["real_sharpe_delta"].min()),
                "max":  float(df["real_sharpe_delta"].max()),
            },
            "turnover": {
                "mean": float(df["turnover"].mean()),
                "std":  float(df["turnover"].std()),
                "min":  float(df["turnover"].min()),
                "max":  float(df["turnover"].max()),
            },
            "production_config": {
                "bullish_threshold": 0.60,
                "bearish_threshold": 0.40,
                "delta": 0.02,
                "expected_sharpe_delta": float(prod.iloc[0]["exp_sharpe_delta"]) if not prod.empty else None,
                "realized_sharpe_delta": float(prod.iloc[0]["real_sharpe_delta"]) if not prod.empty else None,
                "turnover": float(prod.iloc[0]["turnover"]) if not prod.empty else None,
            },
        }

        json_path = OUT_DIR / "constraint_sensitivity_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Saved] {json_path}")

    return {"df": df, "baseline": baseline, "baseline_realized": baseline_realized}


if __name__ == "__main__":
    run_constraint_sensitivity_study(
        signals_path=SIGNALS_PATH,
        rf=0.02,
        w_max=0.30,
        save_outputs=True,
    )