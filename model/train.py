# model/train.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import pickle

from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from config.settings import TICKER, INTERVAL, RANDOM_SEED

def create_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    def label(r):
        if r > threshold: return 2
        elif r < -threshold: return 0
        else: return 1
    df["Label"] = df["Future_Return"].apply(label)
    df.dropna(inplace=True)
    return df

def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
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

if __name__ == "__main__":
    print("📥 Lade Daten...")
    df = fetch_stock_data(TICKER, period="2y", interval=INTERVAL)
    if df is None:
        exit()

    print("📊 Berechne Indikatoren...")
    df = add_indicators(df)
    if df is None:
        exit()

    print("🔧 Erstelle Extra Features...")
    df = add_extra_features(df)

    print("🏷️  Erstelle Labels...")
    df = create_labels(df)

    feature_cols = [
        "EMA_9", "EMA_21", "EMA_50",
        "RSI_14", "RSI_Overbought", "RSI_Oversold",
        "MACD", "MACD_Signal", "MACD_Diff",
        "BB_High", "BB_Low", "BB_Middle", "BB_Position",
        "ATR_14", "Volatility_10d",
        "Return_1d", "Return_3d", "Return_5d", "Return_10d",
        "Volume_Ratio", "EMA_Cross_9_21", "EMA_Cross_21_50"
    ]

    X = df[feature_cols]
    y = df["Label"]
    X_train, y_train = X.iloc[:-63], y.iloc[:-63]
    X_test,  y_test  = X.iloc[-63:], y.iloc[-63:]

    print(f"\n📅 Training: {len(X_train)} Tage | Test (letzte 3 Monate): {len(X_test)} Tage")

    print("\n🧠 Trainiere Modell...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss"
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("\n📊 Ergebnis auf letzten 3 Monaten:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"]))

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print("🔑 Top 5 wichtigste Features:")
    print(importance.nlargest(5).to_string())

    os.makedirs("model/saved", exist_ok=True)
    with open("model/saved/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/saved/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("\n✅ Modell gespeichert!")