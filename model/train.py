# model/train.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from config.settings import TICKER, PERIOD, INTERVAL, TEST_SIZE, RANDOM_SEED

def create_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Erstellt Buy/Sell/Hold Labels basierend auf zukünftiger Rendite.
    +1 = Buy  (Kurs steigt >1% am nächsten Tag)
    -1 = Sell (Kurs fällt >1% am nächsten Tag)
     0 = Hold (Kurs bleibt in ±1% Range)
    """
    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1

    def label(r):
        if r > threshold:
            return 2    # Buy
        elif r < -threshold:
            return 0   # Sell
        else:
            return 1    # Hold

    df["Label"] = df["Future_Return"].apply(label)
    df.dropna(inplace=True)
    return df

def train_model():
    # 1. Daten laden
    print("📥 Lade Daten...")
    df = fetch_stock_data(TICKER, period=PERIOD, interval=INTERVAL)
    if df is None:
        return

    # 2. Indikatoren berechnen
    print("📊 Berechne Indikatoren...")
    df = add_indicators(df)
    if df is None:
        return

    # 3. Labels erstellen
    print("🏷️  Erstelle Labels...")
    df = create_labels(df)

    # 4. Features & Labels trennen
    feature_cols = ["EMA_9", "EMA_21", "EMA_50", "RSI_14",
                    "MACD", "MACD_Signal", "MACD_Diff",
                    "BB_High", "BB_Low", "BB_Middle", "ATR_14"]

    X = df[feature_cols]
    y = df["Label"]

    print(f"📈 Label Verteilung:\n{y.value_counts()}")

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=False
    )

    # 6. Modell trainieren
    print("\n🧠 Trainiere XGBoost Modell...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # 7. Auswerten
    print("\n📊 Ergebnis:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"]))

    # 8. Modell speichern
    os.makedirs("model/saved", exist_ok=True)
    with open("model/saved/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Modell gespeichert: model/saved/xgboost_model.pkl")

if __name__ == "__main__":
    train_model()