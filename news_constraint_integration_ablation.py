# news_constraint_integration_ablation.py
from __future__ import annotations
from typing import Dict, Any, Literal
import pandas as pd

# ============================================================
# Ablation config type
# ============================================================
AblationConfig = Literal["A1", "A2", "B1", "B2", "C1", "C2"]

# ============================================================
# Config açıklamaları (tez için referans)
# ============================================================
ABLATION_CONFIG_DESCRIPTIONS = {
    "A1": "Baseline: fixed delta=0.02, w_max=0.30, clip bullish at w_max (mevcut kod)",
    "A2": "Option 1: fixed delta=0.02, w_max=0.30, relax w_max+delta for bullish",
    "B1": "Option 2: prob-driven delta, w_max=0.30, skip already-capped bullish",
    "B2": "Fully model-driven: prob-driven delta, w_max=0.30, relax w_max+delta",
    "C1": "w_max sensitivity: fixed delta=0.02, w_max=0.25, clip bullish at w_max",
    "C2": "delta sensitivity: fixed delta=0.05, w_max=0.30, clip bullish at w_max",
}


# ============================================================
# Helper: probability-driven delta
# ============================================================
def _prob_driven_delta(
    prob: float,
    neutral: float = 0.50,
    scale: float = 0.20,
) -> float:
    """
    Model-driven delta: probability ne kadar uç değerdeyse
    constraint o kadar güçlü.

    Örnekler:
        prob=0.77 → delta = (0.77-0.50) × 0.20 = 0.054
        prob=0.34 → delta = (0.50-0.34) × 0.20 = 0.032
        prob=0.60 → delta = (0.60-0.50) × 0.20 = 0.020  (threshold'da sabit delta ile aynı)
        prob=0.40 → delta = (0.50-0.40) × 0.20 = 0.020
    """
    return abs(prob - neutral) * scale


# ============================================================
# Ana constraint builder
# ============================================================
def build_news_probability_constraints(
    latest_signals: pd.DataFrame,
    baseline_weights: Dict[str, float],
    bullish_threshold: float = 0.60,
    bearish_threshold: float = 0.40,
    delta: float = 0.02,
    w_max: float = 0.30,
    config: AblationConfig = "A1",
) -> Dict[str, Dict[str, Any]]:
    """
    Build portfolio weight constraints from news prediction probabilities.

    Ablation configs:
    - A1: Fixed delta=0.02, w_max=0.30, clip bullish at w_max         (mevcut kod / baseline)
    - A2: Fixed delta=0.02, w_max=0.30, relax w_max+delta for bullish  (Option 1)
    - B1: Prob-driven delta, w_max=0.30, skip already-capped bullish    (Option 2)
    - B2: Prob-driven delta, w_max=0.30, relax w_max+delta for bullish  (Option 1 + model delta)
    - C1: Fixed delta=0.02, w_max=0.25, clip bullish at w_max          (w_max sensitivity)
    - C2: Fixed delta=0.05, w_max=0.30, clip bullish at w_max          (delta sensitivity)

    Parameters
    ----------
    latest_signals : pd.DataFrame
        Output of latest_news_prediction_signals.csv
    baseline_weights : Dict[str, float]
        Baseline mean-variance portfolio weights (unconstrained).
    bullish_threshold : float
        Probability threshold for bullish signal (default 0.60).
    bearish_threshold : float
        Probability threshold for bearish signal (default 0.40).
    delta : float
        Base allocation adjustment for A1, A2, C1 configs (default 0.02).
        For C2 this is overridden to 0.05 internally.
    w_max : float
        Hard diversification cap for A1, A2, B1, B2, C2 configs (default 0.30).
        For C1 this is overridden to 0.25 internally.
    config : AblationConfig
        Which ablation configuration to use. Default "A1" (mevcut davranış).

    Returns
    -------
    constraints : Dict[str, Dict[str, Any]]
        Per-ticker constraint dicts. Keys vary by config but always include:
        - type: "bullish" | "bearish"
        - probability: float
        - baseline_weight: float
        - delta_used: float
        - config: str
        Plus type-specific keys:
        - bullish A1/B1/C1/C2: min_weight
        - bullish A2/B2: min_weight, relaxed_w_max
        - bearish all: max_weight
    """
    constraints: Dict[str, Dict[str, Any]] = {}

    for _, row in latest_signals.iterrows():
        ticker = str(row["ticker"]).upper().strip()

        if ticker not in baseline_weights:
            continue

        prob = float(row["predicted_positive_probability"])
        base_weight = float(baseline_weights[ticker])

        # ── Config'e göre effective delta ve effective w_max belirle ──
        if config == "C1":
            # w_max sensitivity: daha sıkı diversification cap
            effective_delta = float(delta)      # 0.02 (sabit)
            effective_w_max = 0.25              # ← w_max değiştiriliyor
        elif config == "C2":
            # delta sensitivity: daha agresif constraint
            effective_delta = 0.05              # ← delta değiştiriliyor
            effective_w_max = float(w_max)      # 0.30 (sabit)
        elif config in ("B1", "B2"):
            # model-driven delta
            effective_delta = _prob_driven_delta(prob)
            effective_w_max = float(w_max)      # 0.30 (sabit)
        else:
            # A1, A2: sabit delta
            effective_delta = float(delta)      # 0.02 (sabit)
            effective_w_max = float(w_max)      # 0.30 (sabit)

        # ── Bullish constraint ─────────────────────────────────────────
        if prob >= bullish_threshold:

            if config in ("A1", "C1", "C2"):
                # Mevcut davranış: floor = baseline + delta, clip at effective_w_max
                # Zaten w_max'ta olanlar için constraint etkisiz olur
                min_weight = base_weight + effective_delta
                min_weight = min(min_weight, effective_w_max - 1e-4)

                constraints[ticker] = {
                    "type": "bullish",
                    "probability": prob,
                    "baseline_weight": base_weight,
                    "min_weight": min_weight,
                    "delta_used": effective_delta,
                    "effective_w_max": effective_w_max,
                    "config": config,
                }

            elif config == "A2":
                # Option 1: w_max'ı gevşet → bullish sinyal her zaman etkili
                # AVGO 77.7% bullish, base=22.9% → min=24.9%, relaxed_cap=32%
                # GOOGL 65.3% bullish, base=30.0% → min=32.0%, relaxed_cap=32%  ← artık aktif!
                relaxed_cap = effective_w_max + effective_delta
                min_weight = base_weight + effective_delta
                # feasibility: min <= relaxed_cap
                min_weight = min(min_weight, relaxed_cap - 1e-4)

                constraints[ticker] = {
                    "type": "bullish",
                    "probability": prob,
                    "baseline_weight": base_weight,
                    "min_weight": min_weight,
                    "relaxed_w_max": relaxed_cap,
                    "delta_used": effective_delta,
                    "effective_w_max": effective_w_max,
                    "config": config,
                }

            elif config == "B1":
                # Option 2: zaten w_max'ta olan tickers'ı atla
                # Optimizer zaten maksimumda, bullish constraint gereksiz
                if base_weight >= effective_w_max - 0.01:
                    # Bu ticker için constraint üretme, devam et
                    continue

                min_weight = base_weight + effective_delta
                min_weight = min(min_weight, effective_w_max - 1e-4)

                constraints[ticker] = {
                    "type": "bullish",
                    "probability": prob,
                    "baseline_weight": base_weight,
                    "min_weight": min_weight,
                    "delta_used": effective_delta,
                    "effective_w_max": effective_w_max,
                    "config": config,
                    "skipped_if_capped": False,
                }

            elif config == "B2":
                # Fully model-driven: prob-delta + w_max gevşet
                # Hiçbir sayı keyfi değil
                relaxed_cap = effective_w_max + effective_delta
                min_weight = base_weight + effective_delta
                min_weight = min(min_weight, relaxed_cap - 1e-4)

                constraints[ticker] = {
                    "type": "bullish",
                    "probability": prob,
                    "baseline_weight": base_weight,
                    "min_weight": min_weight,
                    "relaxed_w_max": relaxed_cap,
                    "delta_used": effective_delta,
                    "effective_w_max": effective_w_max,
                    "config": config,
                }

        # ── Bearish constraint ─────────────────────────────────────────
        elif prob <= bearish_threshold:
            # Tüm config'lerde aynı mantık, sadece effective_delta değişir
            max_weight = max(0.0, base_weight - effective_delta)

            constraints[ticker] = {
                "type": "bearish",
                "probability": prob,
                "baseline_weight": base_weight,
                "max_weight": max_weight,
                "delta_used": effective_delta,
                "effective_w_max": effective_w_max,
                "config": config,
            }

    return constraints


# ============================================================
# Ablation study runner (tüm 6 config'i tek seferde çalıştır)
# ============================================================
def run_ablation_study(
    latest_signals: pd.DataFrame,
    baseline_weights: Dict[str, float],
    bullish_threshold: float = 0.60,
    bearish_threshold: float = 0.40,
    delta: float = 0.02,
    w_max: float = 0.30,
) -> Dict[str, Dict[str, Any]]:
    """
    Tüm 6 ablation config'ini çalıştır ve sonuçları döndür.

    Kullanım (tez için karşılaştırma tablosu):
    ─────────────────────────────────────────
    from news_constraint_integration import run_ablation_study

    results = run_ablation_study(
        latest_signals=latest_signals_df,
        baseline_weights={"AVGO": 0.209, "GOOGL": 0.30, "MU": 0.30, "NVDA": 0.191},
    )

    for cfg, constraints in results.items():
        print(f"Config {cfg}: {len(constraints)} constraints")
        for ticker, c in constraints.items():
            print(f"  {ticker}: {c}")
    ─────────────────────────────────────────
    """
    results: Dict[str, Dict[str, Any]] = {}

    for cfg in ("A1", "A2", "B1", "B2", "C1", "C2"):
        results[cfg] = build_news_probability_constraints(
            latest_signals=latest_signals,
            baseline_weights=baseline_weights,
            bullish_threshold=bullish_threshold,
            bearish_threshold=bearish_threshold,
            delta=delta,
            w_max=w_max,
            config=cfg,
        )

    return results


# ============================================================
# Ablation summary printer (tez tablosu için)
# ============================================================
def print_ablation_summary(
    ablation_results: Dict[str, Dict[str, Any]],
    selected_tickers: list,
) -> None:
    """
    6 config için karşılaştırma tablosu yazdır.
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    for cfg, desc in ABLATION_CONFIG_DESCRIPTIONS.items():
        constraints = ablation_results.get(cfg, {})
        print(f"\nConfig {cfg}: {desc}")
        print(f"  Active constraints: {len(constraints)}")

        for ticker in selected_tickers:
            t = str(ticker).upper().strip()
            c = constraints.get(t)
            if c is None:
                print(f"  {t}: NO CONSTRAINT (neutral signal)")
                continue

            c_type = c.get("type")
            prob = c.get("probability", 0.0)
            base_w = c.get("baseline_weight", 0.0)
            delta_used = c.get("delta_used", 0.0)

            if c_type == "bullish":
                min_w = c.get("min_weight", 0.0)
                relaxed = c.get("relaxed_w_max")
                eff_wmax = c.get("effective_w_max", 0.30)
                active = min_w > base_w + 1e-6
                relaxed_str = f" relaxed_cap={relaxed:.3f}" if relaxed else ""
                print(
                    f"  {t}: BULLISH p={prob:.1%} "
                    f"base={base_w:.1%} → min_floor={min_w:.1%} "
                    f"delta={delta_used:.4f}{relaxed_str} "
                    f"w_max_used={eff_wmax:.2f} "
                    f"binding={'YES' if active else 'INACTIVE (already capped)'}"
                )
            elif c_type == "bearish":
                max_w = c.get("max_weight", 0.0)
                eff_wmax = c.get("effective_w_max", 0.30)
                binding = max_w < base_w - 1e-6
                print(
                    f"  {t}: BEARISH p={prob:.1%} "
                    f"base={base_w:.1%} → max_cap={max_w:.1%} "
                    f"delta={delta_used:.4f} "
                    f"w_max_used={eff_wmax:.2f} "
                    f"binding={'YES' if binding else 'WEAK'}"
                )

    print("\n" + "=" * 80)