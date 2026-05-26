import pandas as pd
import numpy as np

# 1. Test start tarihini bul
from news_return_predictor import build_ticker_date_prediction_dataset_v2, ALL_TICKERS_RAW_PATH, HORIZON_DAYS

dataset = build_ticker_date_prediction_dataset_v2(ALL_TICKERS_RAW_PATH, min_abs_return_for_signal=0.02)
dataset = dataset.sort_values("news_date_dt").reset_index(drop=True)
split_idx = int(len(dataset) * 0.70)
test_start = dataset.iloc[split_idx]["news_date_dt"]
print(f"Test start: {test_start.date()}")

# 2. SPY Sharpe hesapla
spy = pd.read_csv("data/raw/daily_yahoo/SPY_daily.csv")
spy["timestamp"] = pd.to_datetime(spy["timestamp"])
spy = spy.set_index("timestamp")["close"].pct_change().dropna()
spy.index = spy.index.normalize()

spy_test = spy[spy.index >= test_start]
rf_daily = 0.02 / 252
er = spy_test - rf_daily
spy_sharpe = float(er.mean() / er.std() * np.sqrt(252))
spy_return = float((1 + spy_test).prod() ** (252 / len(spy_test)) - 1)

print(f"SPY test period: {spy_test.index[0].date()} → {spy_test.index[-1].date()}")
print(f"SPY trading days: {len(spy_test)}")
print(f"SPY Sharpe: {spy_sharpe:.4f}")
print(f"SPY Annualized Return: {spy_return*100:.2f}%")