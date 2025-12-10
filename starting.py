import os
import pandas as pd

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

SYMBOL = "AAPL"           
MONTH = "2025-09"        
INTERVAL = "60min"

BASE = "https://www.alphavantage.co/query"
params = (
    f"function=TIME_SERIES_INTRADAY&symbol={SYMBOL}"
    f"&interval={INTERVAL}&adjusted=true&extended_hours=false"
    f"&month={MONTH}&outputsize=full&datatype=csv&apikey={API_KEY}"
)
url = f"{BASE}?{params}"


os.makedirs("data/raw/intraday_60min", exist_ok=True)
out_path = f"data/raw/intraday_60min/{SYMBOL}_{MONTH}.csv"


df = pd.read_csv(url)

df = df.sort_values("timestamp").reset_index(drop=True)

df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(df.head())
