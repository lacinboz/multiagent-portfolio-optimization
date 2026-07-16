# build_returns_yahoodata.py
import os
import glob
import pandas as pd
import numpy as np

DAYS_PER_YEAR = 252
RAW_DIR = "data/raw/daily_yahoo"
OUT_DIR = "data/processed_yahoo"
os.makedirs(OUT_DIR, exist_ok=True)


TRAIN_END_DATE = "2026-01-14"   # training inclusive end
TEST_START_DATE = "2026-01-15"  # test starts here
TEST_END_DATE   = "2026-05-22"  # test ends here

price_frames = []
for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_daily.csv"))):
    fname = os.path.basename(path)
    ticker = fname.split("_")[0]
    if ticker == "SPY":
        continue
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    df = df[["timestamp", "close"]].rename(columns={"close": ticker})
    price_frames.append(df)

if len(price_frames) == 0:
    raise FileNotFoundError(f"No usable *_daily.csv found under {RAW_DIR}")

from functools import reduce
prices_all = reduce(
    lambda left, right: pd.merge(left, right, on="timestamp", how="inner"),
    price_frames
)
prices_all = prices_all.sort_values("timestamp").reset_index(drop=True)

print("Full date range:", prices_all["timestamp"].min(), "->", prices_all["timestamp"].max())
print("Full prices shape:", prices_all.shape)

# ============================================================
# TRAINING DATA: strictly before test period
# ============================================================
prices_train = prices_all[
    prices_all["timestamp"] <= TRAIN_END_DATE
].copy().reset_index(drop=True)

# ============================================================
# TEST DATA: held-out period
# ============================================================
prices_test = prices_all[
    (prices_all["timestamp"] >= TEST_START_DATE) &
    (prices_all["timestamp"] <= TEST_END_DATE)
].copy().reset_index(drop=True)

print(f"\nTraining period: {prices_train['timestamp'].min().date()} "
      f"→ {prices_train['timestamp'].max().date()} "
      f"({len(prices_train)} days)")
print(f"Test period:     {prices_test['timestamp'].min().date()} "
      f"→ {prices_test['timestamp'].max().date()} "
      f"({len(prices_test)} days)")

# ============================================================
# mu and cov: TRAINING DATA ONLY
# ============================================================
returns_train = (
    prices_train.set_index("timestamp")
    .pct_change()
    .dropna(how="any")
)

mu_daily    = returns_train.mean()
sigma_daily = returns_train.std()
mu_annual   = mu_daily   * DAYS_PER_YEAR
sigma_annual= sigma_daily * np.sqrt(DAYS_PER_YEAR)

summary = pd.DataFrame({
    "mu_daily":    mu_daily,
    "sigma_daily": sigma_daily,
    "mu_annual":   mu_annual,
    "sigma_annual":sigma_annual,
    "sharpe":      mu_annual / sigma_annual,
}).sort_values("sharpe", ascending=False)

cov_daily  = returns_train.cov()
cov_annual = cov_daily * DAYS_PER_YEAR

# ============================================================
# TEST RETURNS: for realized performance evaluation
# ============================================================
returns_test = (
    prices_test.set_index("timestamp")
    .pct_change()
    .dropna(how="any")
)

# ============================================================
# Debug
# ============================================================
debug_returns = summary.copy()
debug_returns["mu_daily_pct"]    = debug_returns["mu_daily"]    * 100
debug_returns["mu_annual_pct"]   = debug_returns["mu_annual"]   * 100
debug_returns["sigma_daily_pct"] = debug_returns["sigma_daily"] * 100
debug_returns["sigma_annual_pct"]= debug_returns["sigma_annual"] * 100

print("\n=== Per-Asset Summary (training data only, annualized) ===")
print(summary.round(4))
print(f"\nTraining returns shape: {returns_train.shape}")
print(f"Test returns shape:     {returns_test.shape}")

# ============================================================
# SAVE
# ============================================================
prices_all.to_csv(  os.path.join(OUT_DIR, "prices_daily.csv"),       index=False)
prices_train.to_csv(os.path.join(OUT_DIR, "prices_train.csv"),       index=False)
prices_test.to_csv( os.path.join(OUT_DIR, "prices_test.csv"),        index=False)
returns_train.to_csv(os.path.join(OUT_DIR, "returns_daily.csv"))     
returns_test.to_csv( os.path.join(OUT_DIR, "returns_test.csv"))     
summary.to_csv(     os.path.join(OUT_DIR, "summary_per_asset_annual.csv"))
cov_daily.to_csv(   os.path.join(OUT_DIR, "cov_daily.csv"))
cov_annual.to_csv(  os.path.join(OUT_DIR, "cov_annual.csv"))
debug_returns.to_csv(os.path.join(OUT_DIR, "debug_daily_vs_annual_returns.csv"))

print("\nSaved all processed files under:", OUT_DIR)
print("IMPORTANT: mu/cov computed from TRAINING data only (no look-ahead bias)")
print(f"  Training: up to {TRAIN_END_DATE}")
print(f"  Test:     {TEST_START_DATE} to {TEST_END_DATE}")