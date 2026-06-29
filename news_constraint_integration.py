# news_constraint_integration.py
from __future__ import annotations

from typing import Dict
import pandas as pd


def build_news_probability_constraints(
    latest_signals: pd.DataFrame,
    baseline_weights: Dict[str, float],
    bullish_threshold: float = 0.60,
    bearish_threshold: float = 0.40,
    delta: float = 0.02,
    w_max: float = 0.30,
    min_baseline_weight: float = 1e-3,
) -> Dict[str, Dict[str, float]]:

    constraints = {}

    for _, row in latest_signals.iterrows():

        ticker = str(row["ticker"]).upper().strip()

        if ticker not in baseline_weights:
            continue

        prob = float(row["predicted_positive_probability"])
        base_weight = float(baseline_weights[ticker])

        # ✅ IMPORTANT:
        # Do not create constraints for assets that are effectively not used
        # in the baseline portfolio.
        if base_weight < min_baseline_weight:
            continue

        # =====================================================
        # Bullish constraint: require slightly higher allocation
        # =====================================================
        if prob >= bullish_threshold:

            min_weight = base_weight + delta
            min_weight = min(min_weight, w_max - 1e-4)

            constraints[ticker] = {
                "type": "bullish",
                "probability": prob,
                "baseline_weight": base_weight,
                "min_weight": min_weight,
            }

        # =====================================================
        # Bearish constraint: require lower allocation
        # =====================================================
        elif prob <= bearish_threshold:

            max_weight = max(0.0, base_weight - delta)

            constraints[ticker] = {
                "type": "bearish",
                "probability": prob,
                "baseline_weight": base_weight,
                "max_weight": max_weight,
            }

    return constraints