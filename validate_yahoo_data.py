import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw/daily_yahoo"
OUT_DIR = "data/processed_yahoo"
os.makedirs(OUT_DIR, exist_ok=True)

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

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

def to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

rows = []

for t in tickers:
    path = os.path.join(RAW_DIR, f"{t}_daily.csv")

    if not os.path.exists(path):
        rows.append({
            "ticker": t,
            "file_exists": False,
            "rows": 0,
            "cols_present": None,
            "missing_cols": ",".join(REQUIRED_COLS),
            "timestamp_parse_fail_rows": None,
            "nan_cells_total_required": None,
            "nan_rows_any_required": None,
            "duplicate_timestamps": None,
            "is_monotonic_increasing": None,
            "start_date": None,
            "end_date": None,
            "max_gap_days": None,
            "num_gaps_ge_7d": None,
            "ohlc_invalid_rows": None,
            "volume_nonpositive_rows": None
        })
        continue

    # READ ONLY: sadece oku
    df = pd.read_csv(path)

    cols_present = list(df.columns)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

    if missing_cols:
        rows.append({
            "ticker": t,
            "file_exists": True,
            "rows": len(df),
            "cols_present": ",".join(cols_present),
            "missing_cols": ",".join(missing_cols),
            "timestamp_parse_fail_rows": None,
            "nan_cells_total_required": None,
            "nan_rows_any_required": None,
            "duplicate_timestamps": None,
            "is_monotonic_increasing": None,
            "start_date": None,
            "end_date": None,
            "max_gap_days": None,
            "num_gaps_ge_7d": None,
            "ohlc_invalid_rows": None,
            "volume_nonpositive_rows": None
        })
        continue

    # Buradan sonrası da READ ONLY: sadece analiz için kopya seriler üretir, df'yi "temizlemez"
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")

    timestamp_parse_fail_rows = int(ts.isna().sum())

    # NaN metrikleri (required cols üzerinde)
    required_block = pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol
    })
    nan_cells_total_required = int(required_block.isna().sum().sum())
    nan_rows_any_required = int(required_block.isna().any(axis=1).sum())

    # duplicate timestamps (parse edilen timestamp üzerinden)
    duplicate_timestamps = int(ts.duplicated().sum())

    # sıralı mı? (parse edilmiş ts üzerinden)
    is_mono = bool(ts.is_monotonic_increasing)

    # start/end (parse edilmiş ts)
    start = ts.min()
    end = ts.max()
    start_date = str(start.date()) if pd.notna(start) else None
    end_date = str(end.date()) if pd.notna(end) else None

    # gaps (parse + sort edilmiş unique timestamp üzerinden)
    max_gap_days = None
    num_gaps_ge_7d = None
    ts_valid_sorted = np.sort(ts.dropna().unique())
    if len(ts_valid_sorted) >= 2:
        diffs = np.diff(ts_valid_sorted).astype("timedelta64[D]").astype(int)
        max_gap_days = int(np.max(diffs))
        num_gaps_ge_7d = int(np.sum(diffs >= 7))
    else:
        max_gap_days = np.nan
        num_gaps_ge_7d = np.nan

    # OHLC mantık kontrolü (NaN olmayan satırlar üzerinden)
    # high >= max(open, close), low <= min(open, close), high >= low
    valid_mask = ~(open_.isna() | high.isna() | low.isna() | close.isna())
    ohlc_invalid_rows = int((
        (high[valid_mask] < pd.concat([open_[valid_mask], close[valid_mask]], axis=1).max(axis=1)) |
        (low[valid_mask]  > pd.concat([open_[valid_mask], close[valid_mask]], axis=1).min(axis=1)) |
        (high[valid_mask] < low[valid_mask])
    ).sum())

    # volume <= 0 kontrol (NaN olmayanlar üzerinden)
    vol_valid = vol.dropna()
    volume_nonpositive_rows = int((vol_valid <= 0).sum())

    rows.append({
        "ticker": t,
        "file_exists": True,
        "rows": int(len(df)),
        "cols_present": ",".join(cols_present),
        "missing_cols": "",
        "timestamp_parse_fail_rows": timestamp_parse_fail_rows,
        "nan_cells_total_required": nan_cells_total_required,
        "nan_rows_any_required": nan_rows_any_required,
        "duplicate_timestamps": duplicate_timestamps,
        "is_monotonic_increasing": is_mono,
        "start_date": start_date,
        "end_date": end_date,
        "max_gap_days": max_gap_days,
        "num_gaps_ge_7d": num_gaps_ge_7d,
        "ohlc_invalid_rows": ohlc_invalid_rows,
        "volume_nonpositive_rows": volume_nonpositive_rows
    })

report = pd.DataFrame(rows)

# hızlı flag: problem var mı?
report["has_issue"] = (
    (report["file_exists"] == False) |
    (report["missing_cols"].fillna("") != "") |
    (report["timestamp_parse_fail_rows"].fillna(0) > 0) |
    (report["nan_rows_any_required"].fillna(0) > 0) |
    (report["duplicate_timestamps"].fillna(0) > 0) |
    (report["ohlc_invalid_rows"].fillna(0) > 0) |
    (report["volume_nonpositive_rows"].fillna(0) > 0)
)

out_path = os.path.join(OUT_DIR, "audit_report_readonly.csv")
report.to_csv(out_path, index=False)

print("✅ READ-ONLY audit report saved to:", out_path)
print("\n=== Summary ===")
print("Expected tickers:", len(tickers))
print("Files found:", int(report["file_exists"].sum()))
print("Tickers with issues:", int(report["has_issue"].sum()))

print("\n=== First issues (up to 25) ===")
print(report[report["has_issue"]].head(25).to_string(index=False))
