import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

BASE = "https://www.alphavantage.co/query"

# Hisse listesi (senin universe'in)
TICKERS = ["AAPL", "AMZN", "GOOGL", "JPM",
           "META", "MSFT", "NVDA", "TSLA", "UNH", "XOM"]

FUNCTION = "TIME_SERIES_DAILY"          # daily raw OHLCV
OUTPUTSIZE = "compact"                  # free: last 100 days
DATATYPE = "csv"

OUT_DIR = "data/raw/daily"
os.makedirs(OUT_DIR, exist_ok=True)

for symbol in TICKERS:
    print(f"Downloading daily data for: {symbol}")

    url = (
        f"{BASE}"
        f"?function={FUNCTION}"
        f"&symbol={symbol}"
        f"&outputsize={OUTPUTSIZE}"
        f"&datatype={DATATYPE}"
        f"&apikey={API_KEY}"
    )

    out_path = os.path.join(OUT_DIR, f"{symbol}_daily.csv")

    # Dosya zaten varsa tekrar indirmek istemezsen:
    if os.path.exists(out_path):
        print(f"File already exists, skipping: {out_path}")
        continue

    df = pd.read_csv(url)
    # daily endpoint için kolon adı 'timestamp' olacak zaten
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

print("DONE ✓")
