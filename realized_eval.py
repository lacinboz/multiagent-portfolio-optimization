# realized_eval.py
# ============================================================
# Shared utility: realized portfolio metrics from test returns.
#
# Usage in any ablation script:
#   from realized_eval import compute_realized_metrics, load_test_returns
#
# All ablation scripts use this instead of w @ mu / sqrt(w @ cov @ w).
# ============================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

RETURNS_TEST_PATH = Path("data/processed_yahoo/returns_test.csv")
DAYS_PER_YEAR = 252
RF = 0.02


def load_test_returns(path: Path = RETURNS_TEST_PATH) -> pd.DataFrame:
    """
    Load test period daily returns.
    Index: timestamp (date), Columns: tickers.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.astype(float)


def compute_realized_metrics(
    weights: Dict[str, float],
    returns_test: pd.DataFrame,
    rf: float = RF,
) -> Dict[str, float]:
    """
    Compute realized annualized metrics from test period daily returns.

    Parameters
    ----------
    weights : dict
        Portfolio weights {ticker: weight}. Need not sum to 1 — will be normalized.
    returns_test : pd.DataFrame
        Daily returns for test period. Rows = trading days, cols = tickers.
    rf : float
        Annual risk-free rate.

    Returns
    -------
    dict with keys:
        realized_return   annualized return
        realized_vol      annualized volatility
        realized_sharpe   annualized Sharpe ratio
        realized_max_dd   maximum drawdown
        realized_n_days   number of trading days used
    """
    # Only keep tickers present in both weights and returns
    common = [
        t for t in weights
        if t in returns_test.columns and abs(float(weights[t])) > 1e-9
    ]

    if not common:
        nan = float("nan")
        return {
            "realized_return": nan,
            "realized_vol": nan,
            "realized_sharpe": nan,
            "realized_max_dd": nan,
            "realized_n_days": 0,
        }

    w = np.array([float(weights[t]) for t in common])
    w = w / w.sum()

    daily_ret = returns_test[common].dropna(how="any").values @ w

    n = len(daily_ret)
    ann_return = float(np.mean(daily_ret) * DAYS_PER_YEAR)
    ann_vol    = float(np.std(daily_ret, ddof=1) * np.sqrt(DAYS_PER_YEAR))
    sharpe     = float((ann_return - rf) / ann_vol) if ann_vol > 0 else float("nan")

    cum = np.cumprod(1.0 + daily_ret)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum / running_max - 1.0))

    return {
        "realized_return": ann_return,
        "realized_vol":    ann_vol,
        "realized_sharpe": sharpe,
        "realized_max_dd": max_dd,
        "realized_n_days": n,
    }