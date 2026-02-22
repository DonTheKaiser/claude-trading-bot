# backtest/backtest.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from config.settings import TICKER, PERIOD, INTERVAL

def backtest():
    # 1. Daten & Indikatoren laden
    print("📥 Lade Daten...")
    df = fetch_stock_data(TICKER, period=PERIOD, interval=INTERVAL)
    if df is None:
        return
    df = add_indicators(df)
    if df is None:
        return

    # 2. Modell laden
    print("🧠 Lade Modell...")
    with open("model/saved/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 3. Features & Vorhersagen
    feature_cols = ["EMA_9", "EMA_21", "EMA_50", "RSI_14",
                    "MACD", "MACD_Signal", "MACD_Diff",
                    "BB_High", "BB_Low", "BB_Middle", "ATR_14"]

    df["Signal"] = model.predict(df[feature_cols])
    # 0=Sell, 1=Hold, 2=Buy

    # 4. Backtest Simulation
    capital = 10000.0
    position = 0.0
    buy_price = 0.0
    trades = []

    for i, row in df.iterrows():
        price = row["Close"]
        signal = row["Signal"]

        if signal == 2 and position == 0:
            # Buy
            position = capital / price
            buy_price = price
            capital = 0
            trades.append({"Date": i, "Action": "BUY", "Price": price})

        elif signal == 0 and position > 0:
            # Sell
            capital = position * price
            profit = (price - buy_price) / buy_price * 100
            trades.append({"Date": i, "Action": "SELL", "Price": price, "Profit%": profit})
            position = 0

    # Falls noch offen
    if position > 0:
        capital = position * df["Close"].iloc[-1]

    # 5. Ergebnisse
    total_return = (capital - 10000) / 10000 * 100
    trades_df = pd.DataFrame(trades)

    print(f"\n📊 Backtest Ergebnis für {TICKER}:")
    print(f"   Startkapital:  $10,000")
    print(f"   Endkapital:    ${capital:.2f}")
    print(f"   Gesamtrendite: {total_return:.2f}%")
    print(f"   Anzahl Trades: {len([t for t in trades if t['Action'] == 'BUY'])}")

    if not trades_df.empty:
        print(f"\n📋 Letzte 5 Trades:")
        print(trades_df.tail(5).to_string(index=False))

if __name__ == "__main__":
    backtest()