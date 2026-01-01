import yfinance as yf
import pandas as pd
import os

tickers = [
    "EXC","FAST","CDNS","MRVL","MSFT","MSTR","MU","CPRT","CSGP","CSX","CTAS","CTSH",
    "GILD","GOOGL","MNST","IDXX","INTC","LRCX","MAR","AMZN","ASML","AAPL","ADBE",
    "ADI","ADP","ADSK","AEP","AZN","BKR","BIIB","AMAT","AMD","AMGN","PANW","KHC",
    "PYPL","SHOP","CCEP","AVGO","TEAM","TTD","GOOG","CDW","DXCM","CSCO","PCAR",
    "BKNG","PEP","TSLA","NXPI","KDP","META","WDAY","FANG","CHTR","VRSK","FTNT",
    "VRTX","LULU","REGN","ROP","ROST","SBUX","AXON","MELI","TMUS","NVDA","ODFL",
    "ON","ORLY","PAYX","TRI","XEL","TTWO","TXN","SNPS","QCOM","MDLZ","KLAC","EA",
    "INTU","ISRG","MCHP","HON","NFLX","CMCSA","COST","ABNB","DASH","APP","WBD",
    "DDOG","ZS","GFS","CEG","CRWD","PLTR","PDD","LIN","ARM","GEHC"
]

START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
OUT_DIR = "data/raw/daily_yahoo"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Toplu indir
data = yf.download(
    tickers=tickers,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    group_by="ticker",
    auto_adjust=False,
    threads=True,
    progress=False
)

failed = []

# 2) Her ticker için ayrı CSV
for t in tickers:
    if t not in data.columns.get_level_values(0):
        failed.append(t)
        continue

    df = data[t].dropna(how="all").copy()
    if df.empty:
        failed.append(t)
        continue

    df.reset_index(inplace=True)

    # Senin formatın
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    df.to_csv(os.path.join(OUT_DIR, f"{t}_daily.csv"), index=False)

print("✅ Completed.")
if failed:
    print("⚠️ Failed / missing:", failed)
