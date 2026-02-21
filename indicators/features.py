# indicators/features.py
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from typing import Optional

def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # EMA
    df["EMA_9"]  = EMAIndicator(close, window=9).ema_indicator()
    df["EMA_21"] = EMAIndicator(close, window=21).ema_indicator()
    df["EMA_50"] = EMAIndicator(close, window=50).ema_indicator()

    # RSI
    df["RSI_14"] = RSIIndicator(close, window=14).rsi()

    # MACD
    macd = MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"]   = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close, window=20)
    df["BB_High"]   = bb.bollinger_hband()
    df["BB_Low"]    = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()

    # ATR
    df["ATR_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # NaN Zeilen entfernen
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    from data.fetch_data import fetch_stock_data
    df = fetch_stock_data("TSLA", period="2y", interval="1d")
    if df is not None:
        result = add_indicators(df)
        if result is not None:
            print(result.tail(5))
            print(f"\n✅ Spalten: {list(result.columns)}")