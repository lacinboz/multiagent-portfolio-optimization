import os
import glob
import pandas as pd
import numpy as np

# Conversion to yearly factor ~252 days * 6.5 hours ≈ 1638 hours/year
HOURS_PER_YEAR = 1638

RAW_DIR = "data/raw/intraday_60min"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


price_frames = []
for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    fname = os.path.basename(path)
    ticker = fname.split("_")[0]          
    df = pd.read_csv(path)
  
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
 
    df = df[["timestamp", "close"]].rename(columns={"close": ticker})
    price_frames.append(df)

from functools import reduce
prices = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="inner"), price_frames)
prices = prices.sort_values("timestamp").reset_index(drop=True)

print("Prices shape:", prices.shape)
print(prices.head())

returns = prices.set_index("timestamp").pct_change().dropna(how="any")
print("Returns shape:", returns.shape)
print(returns.head())

mu_hourly = returns.mean()            
sigma_hourly = returns.std()          
mu_annual = mu_hourly * HOURS_PER_YEAR
sigma_annual = sigma_hourly * np.sqrt(HOURS_PER_YEAR)


sharpe = (mu_annual) / sigma_annual

summary = pd.DataFrame({
    "mu_hourly": mu_hourly,
    "sigma_hourly": sigma_hourly,
    "mu_annual": mu_annual,
    "sigma_annual": sigma_annual,
    "sharpe": sharpe
}).sort_values("sharpe", ascending=False)

print("\n=== Per-Asset Summary (annualized) ===")
print(summary.round(4))


cov_hourly = returns.cov()
cov_annual = cov_hourly * HOURS_PER_YEAR  


prices.to_csv(os.path.join(OUT_DIR, "prices_hourly.csv"), index=False)
returns.to_csv(os.path.join(OUT_DIR, "returns_hourly.csv"))
summary.to_csv(os.path.join(OUT_DIR, "summary_per_asset_annual.csv"))
cov_hourly.to_csv(os.path.join(OUT_DIR, "cov_hourly.csv"))
cov_annual.to_csv(os.path.join(OUT_DIR, "cov_annual.csv"))

print("\nSaved processed files under:", OUT_DIR)
