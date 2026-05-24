# ablation_study.py
# ============================================================
# Tez için standalone ablation study script.
# Dashboard'a hiç dokunmaz, sadece ayrı çalıştırılır.
# ============================================================
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from portfolio_prediction_core import run_portfolio_optimization_prediction_ablation

from agents_langgraph import (
    data_agent_get_mu_cov,
    optimization_agent_from_mu_cov,
    prediction_constrained_optimization_agent,
)
# ablation_study.py içinde şunu kullan:
from news_constraint_integration_ablation import (
    AblationConfig,
    ABLATION_CONFIG_DESCRIPTIONS,
    build_news_probability_constraints,
    run_ablation_study,
    print_ablation_summary,
)

# news_constraint_integration.py'ye dokunma

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIG
# ============================================================

SELECTED_TICKERS = ["AVGO", "GOOGL", "MU", "NVDA"]
RF = 0.02
W_MAX = 0.30
LAMBDA_L2 = 1e-3
LATEST_SIGNALS_PATH = Path("data/news_prediction/latest_news_prediction_signals.csv")
OUT_DIR = Path("data/ablation_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return None if not np.isfinite(v) else v
    except Exception:
        return None


def _fmt(x: Any, pct: bool = False, digits: int = 4) -> str:
    v = _safe_float(x)
    if v is None:
        return "–"
    if pct:
        return f"{v * 100:.{digits}f}%"
    return f"{v:.{digits}f}"


# ============================================================
# Core: run one ablation config end-to-end
# ============================================================

def run_single_config(
    config: AblationConfig,
    *,
    mu,
    cov,
    baseline_weights: Dict[str, float],
    latest_signals: pd.DataFrame,
    rf: float = RF,
    w_max: float = W_MAX,
    lambda_l2: float = LAMBDA_L2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs one full optimization with the given ablation config constraints.

    Returns a dict with:
        config, constraints, weights, metrics (return, vol, sharpe),
        baseline_weights, weight_deltas, constraint_debug
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Config {config}: {ABLATION_CONFIG_DESCRIPTIONS[config]}")
        print(f"{'='*60}")

    # ── Build constraints ──────────────────────────────────────
    constraints = build_news_probability_constraints(
        latest_signals=latest_signals,
        baseline_weights=baseline_weights,
        bullish_threshold=0.60,
        bearish_threshold=0.40,
        delta=0.02,
        w_max=w_max,
        config=config,
    )

    if verbose:
        print(f"Constraints ({len(constraints)}):")
        for t, c in constraints.items():
            if c.get("type") == "bullish":
                print(
                    f"  {t}: BULLISH p={c.get('probability', 0):.1%} "
                    f"base={c.get('baseline_weight', 0):.1%} → "
                    f"min_floor={c.get('min_weight', 0):.1%} "
                    f"delta={c.get('delta_used', 0):.4f}"
                    + (f" relaxed_cap={c.get('relaxed_w_max', 0):.3f}" if "relaxed_w_max" in c else "")
                )
            else:
                print(
                    f"  {t}: BEARISH p={c.get('probability', 0):.1%} "
                    f"base={c.get('baseline_weight', 0):.1%} → "
                    f"max_cap={c.get('max_weight', 0):.1%} "
                    f"delta={c.get('delta_used', 0):.4f}"
                )

    # ── Run optimizer ──────────────────────────────────────────
    t0 = time.time()
    result = run_portfolio_optimization_prediction_ablation(
    mu=mu, cov=cov, rf=rf, w_max=w_max,
    lambda_l2=lambda_l2, news_constraints=constraints,
    save_csv=False,
    )
    elapsed = time.time() - t0

    # ── Extract maxsharpe result ───────────────────────────────
    ms = result.get("maxsharpe") or {}
    final_weights = ms.get("weights") or {}
    ret = _safe_float(ms.get("return"))
    vol = _safe_float(ms.get("vol"))
    sharpe = _safe_float(ms.get("sharpe"))

    # ── Compute weight deltas vs baseline ─────────────────────
    all_tickers = list(baseline_weights.keys())
    weight_deltas = {
        t: float(final_weights.get(t, 0.0)) - float(baseline_weights.get(t, 0.0))
        for t in all_tickers
    }

    if verbose:
        print(f"\nResults (max-sharpe, elapsed={elapsed:.2f}s):")
        print(f"  Return:  {_fmt(ret, pct=True)}")
        print(f"  Vol:     {_fmt(vol, pct=True)}")
        print(f"  Sharpe:  {_fmt(sharpe)}")
        print(f"  Weights: { {t: f'{w:.1%}' for t, w in final_weights.items()} }")
        print(f"  Δ vs baseline: { {t: f'{d:+.1%}' for t, d in weight_deltas.items()} }")

    constraint_debug = result.get("constraint_debug") or []

    return {
        "config": config,
        "description": ABLATION_CONFIG_DESCRIPTIONS[config],
        "constraints": constraints,
        "n_constraints": len(constraints),
        "n_bullish": sum(1 for c in constraints.values() if c.get("type") == "bullish"),
        "n_bearish": sum(1 for c in constraints.values() if c.get("type") == "bearish"),
        "weights": final_weights,
        "baseline_weights": baseline_weights,
        "weight_deltas": weight_deltas,
        "return": ret,
        "vol": vol,
        "sharpe": sharpe,
        "optimizer_success": bool(ms.get("success", True)),
        "elapsed_s": round(elapsed, 3),
        "constraint_debug": constraint_debug,
    }


# ============================================================
# Full ablation study: all 6 configs
# ============================================================

def run_full_ablation_study(
    *,
    selected_tickers: List[str] = SELECTED_TICKERS,
    rf: float = RF,
    w_max: float = W_MAX,
    lambda_l2: float = LAMBDA_L2,
    signals_path: Path = LATEST_SIGNALS_PATH,
    save_outputs: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:

    print("\n" + "="*70)
    print("FULL ABLATION STUDY")
    print("="*70)
    print(f"Tickers: {selected_tickers}")
    print(f"rf={rf}, w_max={w_max}, lambda_l2={lambda_l2}")
    print(f"Signals: {signals_path}")

    # ── Load data ──────────────────────────────────────────────
    print("\n[Step 1] Loading mu/cov...")
    mu, cov = data_agent_get_mu_cov(selected_tickers)
    print(f"mu: {mu.to_dict()}")

    # ── Baseline (unconstrained max-sharpe) ───────────────────
    print("\n[Step 2] Computing baseline portfolio (unconstrained)...")
    baseline_res = optimization_agent_from_mu_cov(
        mu=mu,
        cov=cov,
        rf=rf,
        w_max=w_max,
        lambda_l2=lambda_l2,
    )
    baseline_weights = baseline_res.get("maxsharpe", {}).get("weights", {})
    baseline_ret = _safe_float(baseline_res.get("maxsharpe", {}).get("return"))
    baseline_vol = _safe_float(baseline_res.get("maxsharpe", {}).get("vol"))
    baseline_sharpe = _safe_float(baseline_res.get("maxsharpe", {}).get("sharpe"))

    print(f"Baseline weights: { {t: f'{w:.1%}' for t, w in baseline_weights.items()} }")
    print(f"Baseline: return={_fmt(baseline_ret, pct=True)} vol={_fmt(baseline_vol, pct=True)} sharpe={_fmt(baseline_sharpe)}")

    # ── Load prediction signals ────────────────────────────────
    print(f"\n[Step 3] Loading latest prediction signals from {signals_path}...")
    latest_signals = pd.read_csv(signals_path)
    print(f"Signals rows: {len(latest_signals)}")
    print(latest_signals[["ticker", "predicted_positive_probability", "signal_label"]].to_string(index=False))

    # ── Constraint summary (fast, no optimizer) ───────────────
    print("\n[Step 4] Constraint summary (fast pass, no optimizer)...")
    constraint_results = run_ablation_study(latest_signals, baseline_weights)
    print_ablation_summary(constraint_results, selected_tickers)

    # ── Full optimization for all 6 configs ───────────────────
    print("\n[Step 5] Running full optimization for all 6 configs...")

    config_results: Dict[str, Dict[str, Any]] = {}

    for cfg in ("A1", "A2", "B1", "B2", "C1", "C2"):
        res = run_single_config(
            cfg,
            mu=mu,
            cov=cov,
            baseline_weights=baseline_weights,
            latest_signals=latest_signals,
            rf=rf,
            w_max=w_max,
            lambda_l2=lambda_l2,
            verbose=verbose,
        )
        config_results[cfg] = res

    # ── Build comparison table ─────────────────────────────────
    print("\n\n" + "="*70)
    print("ABLATION STUDY COMPARISON TABLE")
    print("="*70)

    # Header
    col_w = 12
    header = (
        f"{'Config':<8}"
        f"{'Return':>{col_w}}"
        f"{'Vol':>{col_w}}"
        f"{'Sharpe':>{col_w}}"
        f"{'#Constraints':>{col_w}}"
        f"{'Turnover':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    # Baseline row
    print(
        f"{'Baseline':<8}"
        f"{_fmt(baseline_ret, pct=True, digits=2):>{col_w}}"
        f"{_fmt(baseline_vol, pct=True, digits=2):>{col_w}}"
        f"{_fmt(baseline_sharpe, digits=4):>{col_w}}"
        f"{'0':>{col_w}}"
        f"{'0.00%':>{col_w}}"
    )

    rows_for_csv = []
    rows_for_csv.append({
        "config": "Baseline",
        "description": "Unconstrained Max-Sharpe",
        "return": baseline_ret,
        "vol": baseline_vol,
        "sharpe": baseline_sharpe,
        "n_constraints": 0,
        "turnover": 0.0,
        **{f"weight_{t}": float(baseline_weights.get(t, 0.0)) for t in selected_tickers},
        **{f"delta_{t}": 0.0 for t in selected_tickers},
    })

    for cfg, res in config_results.items():
        # Turnover = sum of absolute weight changes / 2
        deltas = res.get("weight_deltas") or {}
        turnover = sum(abs(d) for d in deltas.values()) / 2

        print(
            f"{cfg:<8}"
            f"{_fmt(res.get('return'), pct=True, digits=2):>{col_w}}"
            f"{_fmt(res.get('vol'), pct=True, digits=2):>{col_w}}"
            f"{_fmt(res.get('sharpe'), digits=4):>{col_w}}"
            f"{str(res.get('n_constraints', 0)):>{col_w}}"
            f"{f'{turnover:.2%}':>{col_w}}"
        )

        row = {
            "config": cfg,
            "description": res.get("description", ""),
            "return": res.get("return"),
            "vol": res.get("vol"),
            "sharpe": res.get("sharpe"),
            "n_constraints": res.get("n_constraints", 0),
            "n_bullish": res.get("n_bullish", 0),
            "n_bearish": res.get("n_bearish", 0),
            "turnover": turnover,
            "elapsed_s": res.get("elapsed_s"),
            **{f"weight_{t}": float((res.get("weights") or {}).get(t, 0.0)) for t in selected_tickers},
            **{f"delta_{t}": float((res.get("weight_deltas") or {}).get(t, 0.0)) for t in selected_tickers},
        }
        rows_for_csv.append(row)

    # ── Delta vs Baseline ──────────────────────────────────────
    print("\n\nDelta vs Baseline:")
    print("-" * len(header))
    print(
        f"{'Config':<8}"
        f"{'Δ Return':>{col_w}}"
        f"{'Δ Vol':>{col_w}}"
        f"{'Δ Sharpe':>{col_w}}"
    )
    print("-" * len(header))

    for cfg, res in config_results.items():
        d_ret = _safe_float(res.get("return")) - _safe_float(baseline_ret) \
            if res.get("return") is not None and baseline_ret is not None else None
        d_vol = _safe_float(res.get("vol")) - _safe_float(baseline_vol) \
            if res.get("vol") is not None and baseline_vol is not None else None
        d_sh = _safe_float(res.get("sharpe")) - _safe_float(baseline_sharpe) \
            if res.get("sharpe") is not None and baseline_sharpe is not None else None

        d_ret_str = f"{d_ret*100:+.2f}%" if d_ret is not None else "–"
        d_vol_str = f"{d_vol*100:+.2f}%" if d_vol is not None else "–"
        d_sh_str = f"{d_sh:+.4f}" if d_sh is not None else "–"

        print(
            f"{cfg:<8}"
            f"{d_ret_str:>{col_w}}"
            f"{d_vol_str:>{col_w}}"
            f"{d_sh_str:>{col_w}}"
        )

    # ── Weight table per config ────────────────────────────────
    print("\n\nWeights per config:")
    print("-" * (8 + len(selected_tickers) * col_w))
    ticker_header = f"{'Config':<8}" + "".join(f"{t:>{col_w}}" for t in selected_tickers)
    print(ticker_header)
    print("-" * len(ticker_header))

    baseline_w_row = f"{'Baseline':<8}" + "".join(
        f"{float(baseline_weights.get(t, 0.0)):.1%}".rjust(col_w) for t in selected_tickers
    )
    print(baseline_w_row)

    for cfg, res in config_results.items():
        w = res.get("weights") or {}
        row_str = f"{cfg:<8}" + "".join(
            f"{float(w.get(t, 0.0)):.1%}".rjust(col_w) for t in selected_tickers
        )
        print(row_str)

    # ── Save outputs ───────────────────────────────────────────
    if save_outputs:
        csv_path = OUT_DIR / "ablation_comparison_table.csv"
        df_out = pd.DataFrame(rows_for_csv)
        df_out.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")

        json_path = OUT_DIR / "ablation_full_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tickers": selected_tickers,
                    "rf": rf,
                    "w_max": w_max,
                    "lambda_l2": lambda_l2,
                    "baseline": {
                        "weights": baseline_weights,
                        "return": baseline_ret,
                        "vol": baseline_vol,
                        "sharpe": baseline_sharpe,
                    },
                    "configs": {
                        cfg: {
                            k: v for k, v in res.items()
                            if k != "constraint_debug"  # CVXPY debug çıktısı çok uzun
                        }
                        for cfg, res in config_results.items()
                    },
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[Saved] {json_path}")

    return {
        "tickers": selected_tickers,
        "baseline_weights": baseline_weights,
        "baseline_return": baseline_ret,
        "baseline_vol": baseline_vol,
        "baseline_sharpe": baseline_sharpe,
        "config_results": config_results,
        "constraint_results": constraint_results,
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_full_ablation_study(
        selected_tickers=SELECTED_TICKERS,
        rf=RF,
        w_max=W_MAX,
        lambda_l2=LAMBDA_L2,
        signals_path=LATEST_SIGNALS_PATH,
        save_outputs=True,
        verbose=True,
    )