import os
import glob
import pandas as pd
import numpy as np

# Yıllık iş gün sayısı (finansta klasik varsayım)
DAYS_PER_YEAR = 252

RAW_DIR = "data/raw/daily_yahoo"
OUT_DIR = "data/processed_yahoo"
os.makedirs(OUT_DIR, exist_ok=True)

price_frames = []

# Her hisse için günlük close fiyatlarını okuyup merge edeceğiz
for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_daily.csv"))):
    fname = os.path.basename(path)
    ticker = fname.split("_")[0]   # "AAPL_daily.csv" -> "AAPL"

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")


    # Sadece timestamp + close kullanıyoruz
    df = df[["timestamp", "close"]].rename(columns={"close": ticker})
    price_frames.append(df)
if len(price_frames) == 0:
    raise FileNotFoundError(f"No usable *_daily.csv found under {RAW_DIR}")

from functools import reduce

# Tüm hisseleri timestamp üzerinden merge et
prices = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="inner"),
                price_frames)
prices = prices.sort_values("timestamp").reset_index(drop=True)
print("Date range:", prices["timestamp"].min(), "->", prices["timestamp"].max())

print("Prices shape:", prices.shape)
print(prices.head())
effective_start = prices["timestamp"].min()
effective_end = prices["timestamp"].max()
print(f"Effective analysis period: {effective_start.date()} – {effective_end.date()}")


# Günlük basit getiriler (% değişim)
returns = prices.set_index("timestamp").pct_change().dropna(how="any")
print("Returns shape:", returns.shape)
print(returns.head())

# Günlük ortalama getiri ve volatilite
mu_daily = returns.mean()
sigma_daily = returns.std()

# Yıllığa çevirme
mu_annual = mu_daily * DAYS_PER_YEAR
sigma_annual = sigma_daily * np.sqrt(DAYS_PER_YEAR)

sharpe = (mu_annual) / sigma_annual

summary = pd.DataFrame({
    "mu_daily": mu_daily,
    "sigma_daily": sigma_daily,
    "mu_annual": mu_annual,
    "sigma_annual": sigma_annual,
    "sharpe": sharpe
}).sort_values("sharpe", ascending=False)

print("\n=== Per-Asset Summary (annualized, daily-based) ===")
print(summary.round(4))

# Kovaryans matrisi (günlük ve yıllık)
cov_daily = returns.cov()
cov_annual = cov_daily * DAYS_PER_YEAR

# Kaydet
prices.to_csv(os.path.join(OUT_DIR, "prices_daily.csv"), index=False)
returns.to_csv(os.path.join(OUT_DIR, "returns_daily.csv"))
summary.to_csv(os.path.join(OUT_DIR, "summary_per_asset_annual.csv"))
cov_daily.to_csv(os.path.join(OUT_DIR, "cov_daily.csv"))
cov_annual.to_csv(os.path.join(OUT_DIR, "cov_annual.csv"))

print("\nSaved processed files under:", OUT_DIR)
