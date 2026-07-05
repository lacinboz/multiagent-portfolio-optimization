# ablation_study.py
# ============================================================
# Constraint Design Ablation Study
#
# PURPOSE: Justify why production uses A1 config
# ✅ Both expected AND realized metrics reported
# ✅ No look-ahead bias: uses as_of_20260114 signals
# ✅ Deterministic — no 5-run loop needed
# ============================================================
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from news_constraint_integration_ablation import (
    AblationConfig,
    ABLATION_CONFIG_DESCRIPTIONS,
    build_news_probability_constraints,
)
from realized_eval import compute_realized_metrics, load_test_returns

OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_PATH = Path("data/news_prediction/latest_news_prediction_signals_as_of_20260114.csv")
MU_PATH      = Path("data/processed_yahoo/summary_per_asset_annual.csv")
COV_PATH     = Path("data/processed_yahoo/cov_annual.csv")

RF       = 0.02
W_MAX    = 0.30
LAMBDA   = 1e-3
BULL_THR = 0.60
BEAR_THR = 0.40
DELTA    = 0.02
MIN_BASE_W = 1e-3


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


def _optimize(mu, cov, rf, w_max, lambda_l2,
              news_constraints=None) -> Dict[str, Any]:
    tickers = list(mu.index)
    n = len(tickers)
    effective_w_max = max(w_max, 1.0 / n + 1e-6)

    cov_np = cov.values.copy()
    if np.linalg.eigvalsh(cov_np).min() < 0:
        cov_np = _near_psd(cov_np)
    cov_f = pd.DataFrame(cov_np, index=tickers, columns=tickers)

    bounds = []
    for t in tickers:
        upper = effective_w_max
        if news_constraints and t in news_constraints:
            relaxed = news_constraints[t].get("relaxed_w_max")
            if relaxed is not None:
                upper = max(float(relaxed), effective_w_max)
        bounds.append((0.0, upper))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if news_constraints:
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        for ticker, cdict in news_constraints.items():
            if ticker not in ticker_to_idx:
                continue
            idx = ticker_to_idx[ticker]
            if "min_weight" in cdict:
                mw = float(cdict["min_weight"])
                cons.append({"type": "ineq",
                             "fun": lambda w, i=idx, m=mw: w[i] - m})
            if "max_weight" in cdict:
                mw = float(cdict["max_weight"])
                cons.append({"type": "ineq",
                             "fun": lambda w, i=idx, m=mw: m - w[i]})

    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov_f.values @ w))
        return -(r - rf) / v if v > 0 else np.inf

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=cons)
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
# Run one config
# ============================================================

def _run_config(
    cfg: AblationConfig,
    mu, cov,
    baseline_weights: Dict[str, float],
    baseline_sharpe: float,
    baseline_return: float,
    baseline_vol: float,
    baseline_realized_sharpe: float,
    latest_signals: pd.DataFrame,
    returns_test: pd.DataFrame,
    rf: float, w_max: float, lambda_l2: float,
) -> Dict[str, Any]:

    t0 = time.time()

    sig_filtered = latest_signals[
        latest_signals["ticker"].apply(
            lambda t: baseline_weights.get(t, 0.0) >= MIN_BASE_W
        )
    ].copy()

    constraints = build_news_probability_constraints(
        latest_signals=sig_filtered,
        baseline_weights=baseline_weights,
        bullish_threshold=BULL_THR,
        bearish_threshold=BEAR_THR,
        delta=DELTA,
        w_max=w_max,
        config=cfg,
    )

    result = _optimize(mu, cov, rf, w_max, lambda_l2, constraints)
    elapsed = time.time() - t0

    tickers = list(mu.index)
    turnover = sum(
        abs(result["weights"].get(t, 0) - baseline_weights.get(t, 0))
        for t in tickers
    ) / 2.0

    n_bull = sum(1 for c in constraints.values() if c.get("type") == "bullish")
    n_bear = sum(1 for c in constraints.values() if c.get("type") == "bearish")

    # ✅ Realized metrics
    real = compute_realized_metrics(result["weights"], returns_test)

    return {
        "config": cfg,
        "description": ABLATION_CONFIG_DESCRIPTIONS[cfg],
        "n_constraints": len(constraints),
        "n_bullish": n_bull,
        "n_bearish": n_bear,
        # expected
        "exp_sharpe": result["sharpe"],
        "exp_sharpe_delta": result["sharpe"] - baseline_sharpe,
        "exp_return": result["return"],
        "exp_return_delta": result["return"] - baseline_return,
        "exp_vol": result["vol"],
        "exp_vol_delta": result["vol"] - baseline_vol,
        # realized
        "real_sharpe": real["realized_sharpe"],
        "real_sharpe_delta": real["realized_sharpe"] - baseline_realized_sharpe,
        "real_return_pct": real["realized_return"] * 100,
        "real_vol_pct": real["realized_vol"] * 100,
        "real_max_dd_pct": real["realized_max_dd"] * 100,
        "real_n_days": real["realized_n_days"],
        # common
        "turnover": turnover,
        "optimizer_success": result["success"],
        "elapsed_s": round(elapsed, 3),
        "weights": result["weights"],
    }


# ============================================================
# Main study
# ============================================================

def run_ablation_study(
    signals_path: Path = SIGNALS_PATH,
    rf: float = RF,
    w_max: float = W_MAX,
    lambda_l2: float = LAMBDA,
    save_outputs: bool = True,
) -> Dict[str, Any]:

    print("\n" + "=" * 70)
    print("CONSTRAINT DESIGN ABLATION STUDY")
    print("✅ Both expected AND realized metrics reported")
    print("✅ No look-ahead bias: signals capped at 2026-01-14")
    print("=" * 70)
    print(f"rf={rf}, w_max={w_max}, lambda_l2={lambda_l2}")
    print(f"Fixed: bull>={BULL_THR}, bear<={BEAR_THR}, base_delta={DELTA}")

    print("\nConfigs:")
    for cfg, desc in ABLATION_CONFIG_DESCRIPTIONS.items():
        print(f"  {cfg}: {desc}")

    # Load signals
    if not signals_path.exists():
        raise FileNotFoundError(f"Signals not found: {signals_path}")
    latest_signals = pd.read_csv(signals_path)
    latest_signals["ticker"] = (
        latest_signals["ticker"].astype(str).str.upper().str.strip()
    )
    signal_tickers = latest_signals["ticker"].unique().tolist()

    # Load mu/cov
    mu, cov = _load_mu_cov(tickers=signal_tickers)
    tickers = list(mu.index)
    print(f"\nUniverse: {len(tickers)} tickers")

    latest_signals = latest_signals[
        latest_signals["ticker"].isin(tickers)
    ].copy()

    # Load test returns
    returns_test = load_test_returns()
    print(f"Test returns: {len(returns_test)} days "
          f"({returns_test.index[0].date()} → {returns_test.index[-1].date()})")

    # Signal distribution
    n_bull = (latest_signals["predicted_positive_probability"] >= BULL_THR).sum()
    n_bear = (latest_signals["predicted_positive_probability"] <= BEAR_THR).sum()
    print(f"Signal distribution (bull>={BULL_THR}, bear<={BEAR_THR}): "
          f"Bullish={n_bull} | Bearish={n_bear} | "
          f"Neutral={len(latest_signals)-n_bull-n_bear}")

    # Baseline
    baseline = _optimize(mu, cov, rf, w_max, lambda_l2)
    baseline_weights = baseline["weights"]
    baseline_realized = compute_realized_metrics(baseline_weights, returns_test)
    baseline_realized_sharpe = baseline_realized["realized_sharpe"]

    print(f"\nBaseline expected : Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | "
          f"Vol={baseline['vol']*100:.2f}%")
    print(f"Baseline realized : Sharpe={baseline_realized_sharpe:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}% | "
          f"Vol={baseline_realized['realized_vol']*100:.2f}%")

    # Run all 6 configs
    print(f"\nRunning 6 constraint configurations...")
    rows = []
    for cfg in ("A1", "A2", "B1", "B2", "C1", "C2"):
        res = _run_config(
            cfg, mu, cov,
            baseline_weights=baseline_weights,
            baseline_sharpe=baseline["sharpe"],
            baseline_return=baseline["return"],
            baseline_vol=baseline["vol"],
            baseline_realized_sharpe=baseline_realized_sharpe,
            latest_signals=latest_signals,
            returns_test=returns_test,
            rf=rf, w_max=w_max, lambda_l2=lambda_l2,
        )
        rows.append(res)
        print(f"  {cfg}: Exp ΔS={res['exp_sharpe_delta']:+.4f} | "
              f"Real ΔS={res['real_sharpe_delta']:+.4f} | "
              f"Turnover={res['turnover']*100:.1f}% | "
              f"#Bull={res['n_bullish']} #Bear={res['n_bearish']} | "
              f"{'✓' if res['optimizer_success'] else '✗'}")

    # ============================================================
    # Print EXPECTED table
    # ============================================================
    print(f"\n\n{'='*70}")
    print("CONSTRAINT DESIGN — EXPECTED METRICS")
    print(f"Baseline: Sharpe={baseline['sharpe']:.4f} | "
          f"Return={baseline['return']*100:.2f}% | "
          f"Vol={baseline['vol']*100:.2f}%")
    print(f"{'='*70}")

    col = 11
    header = (
        f"{'Config':<8}"
        f"{'Exp ΔSharpe':>{col}}"
        f"{'Exp ΔReturn':>{col}}"
        f"{'Exp ΔVol':>{col}}"
        f"{'Turnover':>{col}}"
        f"{'#Bull':>6}"
        f"{'#Bear':>6}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Baseline':<8}"
        f"{'+0.0000':>{col}}"
        f"{'+0.00%':>{col}}"
        f"{'+0.00%':>{col}}"
        f"{'0.0%':>{col}}"
        f"{'—':>6}"
        f"{'—':>6}"
    )
    for r in rows:
        prod = " ← PROD" if r["config"] == "A1" else ""
        print(
            f"{r['config']:<8}"
            f"{r['exp_sharpe_delta']:>+{col}.4f}"
            f"{r['exp_return_delta']*100:>+{col}.2f}%"
            f"{r['exp_vol_delta']*100:>+{col}.2f}%"
            f"{r['turnover']*100:>{col}.1f}%"
            f"{r['n_bullish']:>6}"
            f"{r['n_bearish']:>6}"
            f"{prod}"
        )

    # ============================================================
    # Print REALIZED table
    # ============================================================
    print(f"\n\n{'='*70}")
    print("CONSTRAINT DESIGN — REALIZED METRICS")
    print(f"Baseline: Sharpe={baseline_realized_sharpe:.4f} | "
          f"Return={baseline_realized['realized_return']*100:.2f}% | "
          f"Vol={baseline_realized['realized_vol']*100:.2f}%")
    print(f"{'='*70}")

    header_r = (
        f"{'Config':<8}"
        f"{'Real Sharpe':>{col}}"
        f"{'Real ΔSharpe':>{col}}"
        f"{'Real Return':>{col}}"
        f"{'Real Vol':>{col}}"
        f"{'Real MaxDD':>{col}}"
        f"{'Turnover':>{col}}"
    )
    print(header_r)
    print("-" * len(header_r))
    print(
        f"{'Baseline':<8}"
        f"{baseline_realized_sharpe:>{col}.4f}"
        f"{'+0.0000':>{col}}"
        f"{baseline_realized['realized_return']*100:>{col}.2f}%"
        f"{baseline_realized['realized_vol']*100:>{col}.2f}%"
        f"{baseline_realized['realized_max_dd']*100:>{col}.2f}%"
        f"{'0.0%':>{col}}"
    )
    for r in rows:
        prod = " ← PROD" if r["config"] == "A1" else ""
        print(
            f"{r['config']:<8}"
            f"{r['real_sharpe']:>{col}.4f}"
            f"{r['real_sharpe_delta']:>+{col}.4f}"
            f"{r['real_return_pct']:>{col}.2f}%"
            f"{r['real_vol_pct']:>{col}.2f}%"
            f"{r['real_max_dd_pct']:>{col}.2f}%"
            f"{r['turnover']*100:>{col}.1f}%"
            f"{prod}"
        )

    # ============================================================
    # Save outputs
    # ============================================================
    if save_outputs:
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "weights"}
            for r in rows
        ])
        csv_path = OUT_DIR / "ablation_comparison_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        summary = {
            "parameters": {
                "rf": rf, "w_max": w_max, "lambda_l2": lambda_l2,
                "bull_threshold": BULL_THR, "bear_threshold": BEAR_THR,
                "delta": DELTA, "min_baseline_weight": MIN_BASE_W,
                "universe_size": len(tickers),
                "signals_file": str(signals_path),
            },
            "baseline": {
                "expected_sharpe": baseline["sharpe"],
                "expected_return": baseline["return"],
                "expected_vol": baseline["vol"],
                "realized_sharpe": baseline_realized_sharpe,
                "realized_return": baseline_realized["realized_return"],
                "realized_vol": baseline_realized["realized_vol"],
                "realized_max_dd": baseline_realized["realized_max_dd"],
                "realized_n_days": baseline_realized["realized_n_days"],
            },
            "configs": {
                r["config"]: {
                    "description": r["description"],
                    "exp_sharpe_delta": r["exp_sharpe_delta"],
                    "real_sharpe_delta": r["real_sharpe_delta"],
                    "exp_return_delta": r["exp_return_delta"],
                    "exp_vol_delta": r["exp_vol_delta"],
                    "real_return_pct": r["real_return_pct"],
                    "real_vol_pct": r["real_vol_pct"],
                    "real_max_dd_pct": r["real_max_dd_pct"],
                    "turnover": r["turnover"],
                    "n_bullish": r["n_bullish"],
                    "n_bearish": r["n_bearish"],
                    "optimizer_success": r["optimizer_success"],
                }
                for r in rows
            },
        }
        json_path = OUT_DIR / "ablation_full_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Saved] {json_path}")

    return {"baseline": baseline, "rows": rows, "universe_size": len(tickers)}


if __name__ == "__main__":
    run_ablation_study(
        signals_path=SIGNALS_PATH,
        rf=RF,
        w_max=W_MAX,
        save_outputs=True,
    )