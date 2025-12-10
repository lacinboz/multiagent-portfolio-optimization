import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

BASE = "https://www.alphavantage.co/query"
INTERVAL = "60min"

# Hisse listemiz (senin universe'in)
TICKERS = ["AAPL", "AMZN", "GOOGL", "JPM",
           "META", "MSFT", "NVDA", "TSLA", "UNH", "XOM"]

# Bugün hangi ay(lar)ı indirmek istiyorsun?
# Örneğin bugün sadece 1 ek ay:
MONTHS = ["2025-09"]      # istersen ["2025-09","2025-08"] da yapabilirsin

os.makedirs("data/raw/intraday_60min", exist_ok=True)

max_calls_per_day = 25
call_count = 0

for symbol in TICKERS:
    for month in MONTHS:
        if call_count >= max_calls_per_day:
            print("API limitine yaklaşmamak için durduruyorum.")
            break

        out_path = f"data/raw/intraday_60min/{symbol}_{month}.csv"
        if os.path.exists(out_path):
            print(f"Zaten var, atlıyorum: {out_path}")
            continue

        params = (
            f"function=TIME_SERIES_INTRADAY&symbol={symbol}"
            f"&interval={INTERVAL}&adjusted=true&extended_hours=false"
            f"&month={month}&outputsize=full&datatype=csv&apikey={API_KEY}"
        )
        url = f"{BASE}?{params}"

        print(f"Downloading {symbol} {month} -> {out_path}")
        df = pd.read_csv(url)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        call_count += 1

    if call_count >= max_calls_per_day:
        break

print(f"Toplam API çağrısı: {call_count}")
