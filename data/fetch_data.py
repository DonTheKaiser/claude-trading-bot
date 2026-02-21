# data/fetch_data.py
import yfinance as yf
import pandas as pd
import os
from typing import Optional

def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    print(f"📥 Lade Daten für {ticker} ...")
    
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    
    if df is None or df.empty:
        print(f"❌ Keine Daten für {ticker} gefunden!")
        return None
    
    # Spaltennamen vereinfachen
    new_cols = []
    for c in df.columns:
        new_cols.append(c[0] if isinstance(c, tuple) else c)
    df.columns = new_cols
    df.index.name = "Date"
    
    # Lokal speichern
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{ticker}_{interval}.csv"
    df.to_csv(filepath)
    