import os
import io
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

BASE_URL = "https://www.alphavantage.co/query"
params = {
    "function": "LISTING_STATUS",
    "state": "active",     
    "apikey": API_KEY
}


response = requests.get(BASE_URL, params=params)


decoded_content = response.content.decode("utf-8")
df = pd.read_csv(io.StringIO(decoded_content))


os.makedirs("data/raw", exist_ok=True)
out_path = "data/raw/all_active_stocks.csv"
df.to_csv(out_path, index=False)
print(f"Saved all active stocks to: {out_path}")


print(df.head(10))
