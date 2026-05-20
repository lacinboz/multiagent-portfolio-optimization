# news_constraint_integration.py
from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def build_news_probability_constraints(
    latest_signals: pd.DataFrame,
    baseline_weights: Dict[str, float],
    bullish_threshold: float = 0.60,
    bearish_threshold: float = 0.40,
    delta: float = 0.02,
    w_max: float = 0.30,
) -> Dict[str, Dict[str, float]]:
    """
    Build portfolio weight constraints from news probabilities.

    This does NOT modify expected returns or covariance.
    It only constrains feasible portfolio allocations.

    Parameters
    ----------
    latest_signals : pd.DataFrame
        Output of latest_news_prediction_signals.csv

    baseline_weights : Dict[str, float]
        Baseline mean-variance portfolio weights.

    bullish_threshold : float
        Probability threshold for bullish signal.

    bearish_threshold : float
        Probability threshold for bearish signal.

    delta : float
        Minimum allocation adjustment.

    Returns
    -------
    constraints : dict

        Example:

        {
            "AAPL": {"min_weight": 0.10},
            "TSLA": {"max_weight": 0.03},
        }
    """

    constraints = {}

    for _, row in latest_signals.iterrows():

        ticker = str(row["ticker"]).upper().strip()

        if ticker not in baseline_weights:
            continue

        prob = float(row["predicted_positive_probability"])
        base_weight = float(baseline_weights[ticker])

        # =====================================================
        # Bullish constraint
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
        # Bearish constraint
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