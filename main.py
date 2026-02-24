# main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from model.predict import predict_signal
from broker.execute import execute_signal
from config.settings import TICKER, INTERVAL
import pickle

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

def main():
    print(f"🚀 Claude Trading Bot — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 50)

    # 1. Modell prüfen
    if not os.path.exists("model/saved/xgboost_model.pkl"):
        print("⚠️  Kein Modell! Erst trainieren: python -m model.train")
        return

    # 2. Daten laden
    df = fetch_stock_data(TICKER, period="6mo", interval=INTERVAL)
    if df is None:
        return

    # 3. Indikatoren & Features
    df = add_indicators(df)
    if df is None:
        return
    df = add_extra_features(df)

    # 4. Modell & Features laden
    with open("model/saved/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/saved/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # 5. Signal generieren
    latest = df.iloc[-1]
    X = df[feature_cols].iloc[-1:]
    signal = model.predict(X)[0]
    price = float(latest["Close"])

    signal_map = {0: "🔴 SELL", 1: "🟡 HOLD", 2: "🟢 BUY"}
    print(f"📈 {TICKER} @ ${price:.2f}")
    print(f"🎯 Signal: {signal_map[signal]}")
    print("=" * 50)

    # 6. Order ausführen
    execute_signal(signal, price)

    # 7. Log schreiben
    os.makedirs("logs", exist_ok=True)
    with open("logs/trades.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {TICKER} | ${price:.2f} | {signal_map[signal]}\n")

if __name__ == "__main__":
    main()