# model/predict.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
from datetime import datetime
from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from config.settings import TICKER, INTERVAL

def add_extra_features(df):
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_3d"]  = df["Close"].pct_change(3)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Volume_MA10"]    = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"]   = df["Volume"] / df["Volume_MA10"]
    df["RSI_Overbought"] = (df["RSI_14"] > 70).astype(int)
    df["RSI_Oversold"]   = (df["RSI_14"] < 30).astype(int)
    df["EMA_Cross_9_21"]  = (df["EMA_9"]  > df["EMA_21"]).astype(int)
    df["EMA_Cross_21_50"] = (df["EMA_21"] > df["EMA_50"]).astype(int)
    df["BB_Position"]    = (df["Close"] - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"])
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df.dropna(inplace=True)
    return df

def predict_signal():
    print(f"🤖 Trading Bot Signal — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 50)

    # 1. Modell & Features laden
    with open("model/saved/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/saved/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # 2. Aktuelle Daten laden
    df = fetch_stock_data(TICKER, period="6mo", interval=INTERVAL)
    if df is None:
        return
    df = add_indicators(df)
    if df is None:
        return
    df = add_extra_features(df)

    # 3. Letzter bekannter Tag
    latest = df.iloc[-1]
    X = df[feature_cols].iloc[-1:]

    # 4. Signal
    signal_num = model.predict(X)[0]
    signal_map = {0: "🔴 SELL", 1: "🟡 HOLD", 2: "🟢 BUY"}
    signal = signal_map[signal_num]

    print(f"📈 Ticker:        {TICKER}")
    print(f"💰 Aktueller Kurs: ${latest['Close']:.2f}")
    print(f"📊 RSI:           {latest['RSI_14']:.1f}")
    print(f"📉 MACD:          {latest['MACD']:.3f}")
    print(f"📍 BB Position:   {latest['BB_Position']:.2f}")
    print(f"\n🎯 Signal:        {signal}")
    print("=" * 50)

    # Stop-Loss & Take-Profit Empfehlung
    atr = latest["ATR_14"]
    price = latest["Close"]
    print(f"🛡️  Stop-Loss:     ${price - 2 * atr:.2f} (-{2 * atr / price * 100:.1f}%)")
    print(f"🎯 Take-Profit:   ${price + 3 * atr:.2f} (+{3 * atr / price * 100:.1f}%)")

if __name__ == "__main__":
    predict_signal()    