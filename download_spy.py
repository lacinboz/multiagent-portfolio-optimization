"""
download_spy.py
SPY_daily.csv dosyasini data/raw/daily_yahoo/ altina indirir.
Mevcut download_yahoo.py'a dokunmadan ayri calistirilabilir.
Cikti: data/raw/daily_yahoo/SPY_daily.csv
"""

import os
import pandas as pd
import yfinance as yf

START_DATE = "2020-01-01"
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")
OUT_DIR    = "data/raw/daily_yahoo"
os.makedirs(OUT_DIR, exist_ok=True)

out_path = os.path.join(OUT_DIR, "SPY_daily.csv")

print(f"Downloading SPY {START_DATE} -> {END_DATE} ...")

df = yf.download("SPY", start=START_DATE, end=END_DATE,
                 interval="1d", auto_adjust=False, progress=False)

if df.empty:
    print("ERROR: No data returned for SPY.")
else:
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Date": "timestamp", "Close": "close",
                             "Open": "open", "High": "high",
                             "Low": "low", "Volume": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)")
    print(df.tail(3).to_string())