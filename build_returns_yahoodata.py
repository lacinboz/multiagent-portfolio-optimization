import os
import glob
import pandas as pd
import numpy as np


DAYS_PER_YEAR = 252

RAW_DIR = "data/raw/daily_yahoo"
OUT_DIR = "data/processed_yahoo"
os.makedirs(OUT_DIR, exist_ok=True)

price_frames = []


for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_daily.csv"))):
    fname = os.path.basename(path)
    ticker = fname.split("_")[0]   # "AAPL_daily.csv" -> "AAPL"

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")


  
    df = df[["timestamp", "close"]].rename(columns={"close": ticker})
    price_frames.append(df)
if len(price_frames) == 0:
    raise FileNotFoundError(f"No usable *_daily.csv found under {RAW_DIR}")

from functools import reduce


prices = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="inner"),
                price_frames)
prices = prices.sort_values("timestamp").reset_index(drop=True)
print("Date range:", prices["timestamp"].min(), "->", prices["timestamp"].max())

print("Prices shape:", prices.shape)
print(prices.head())
effective_start = prices["timestamp"].min()
effective_end = prices["timestamp"].max()
print(f"Effective analysis period: {effective_start.date()} – {effective_end.date()}")


# Daily Returns 
returns = prices.set_index("timestamp").pct_change().dropna(how="any")
print("Returns shape:", returns.shape)
print(returns.head())
mu_daily = returns.mean()
sigma_daily = returns.std()


mu_annual = mu_daily * DAYS_PER_YEAR
sigma_annual = sigma_daily * np.sqrt(DAYS_PER_YEAR)
debug_mu = pd.DataFrame({
    "mu_daily": mu_daily,
    "mu_daily_pct": mu_daily * 100,
    "mu_annual": mu_annual,
    "mu_annual_pct": mu_annual * 100,
})

print("\n=== EXPECTED RETURN DEBUG ===")
print(debug_mu.round(6))

sharpe = (mu_annual) / sigma_annual

summary = pd.DataFrame({
    "mu_daily": mu_daily,
    "sigma_daily": sigma_daily,
    "mu_annual": mu_annual,
    "sigma_annual": sigma_annual,
    "sharpe": sharpe
}).sort_values("sharpe", ascending=False)

debug_returns = summary.copy()

debug_returns["mu_daily_pct"] = debug_returns["mu_daily"] * 100
debug_returns["mu_annual_pct"] = debug_returns["mu_annual"] * 100
debug_returns["sigma_daily_pct"] = debug_returns["sigma_daily"] * 100
debug_returns["sigma_annual_pct"] = debug_returns["sigma_annual"] * 100

print("\n=== DEBUG: Daily vs Annualized Expected Returns ===")
print(
    debug_returns[
        ["mu_daily_pct", "mu_annual_pct", "sigma_daily_pct", "sigma_annual_pct", "sharpe"]
    ].round(4)
)

print("\n=== Per-Asset Summary (annualized, daily-based) ===")
print(summary.round(4))


cov_daily = returns.cov()
cov_annual = cov_daily * DAYS_PER_YEAR


prices.to_csv(os.path.join(OUT_DIR, "prices_daily.csv"), index=False)
returns.to_csv(os.path.join(OUT_DIR, "returns_daily.csv"))
summary.to_csv(os.path.join(OUT_DIR, "summary_per_asset_annual.csv"))
cov_daily.to_csv(os.path.join(OUT_DIR, "cov_daily.csv"))
cov_annual.to_csv(os.path.join(OUT_DIR, "cov_annual.csv"))
debug_returns.to_csv(os.path.join(OUT_DIR, "debug_daily_vs_annual_returns.csv"))

print("\nSaved processed files under:", OUT_DIR)
