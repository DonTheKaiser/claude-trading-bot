# model/evaluate.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from config.settings import TICKER, PERIOD, INTERVAL, RANDOM_SEED

def create_labels(df, threshold=0.01):
    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    def label(r):
        if r > threshold: return 2
        elif r < -threshold: return 0
        else: return 1
    df["Label"] = df["Future_Return"].apply(label)
    df.dropna(inplace=True)
    return df

def walk_forward_test():
    """
    Walk-Forward Testing — realistischerer Backtest.
    Trainiere auf Vergangenheit, teste auf Zukunft.
    """
    print("📥 Lade Daten...")
    df = fetch_stock_data(TICKER, period=PERIOD, interval=INTERVAL)
    if df is None:
        return
    df = add_indicators(df)
    if df is None:
        return
    df = create_labels(df)

    feature_cols = ["EMA_9", "EMA_21", "EMA_50", "RSI_14",
                    "MACD", "MACD_Signal", "MACD_Diff",
                    "BB_High", "BB_Low", "BB_Middle", "ATR_14"]

    X = df[feature_cols].values
    y = df["Label"].values

    # Walk-Forward: 70% train, 30% test — kein shuffle!
    split = int(len(df) * 0.7)

    results = []
    window = 50  # Trainiere auf 50 Tage, teste auf nächsten Tag

    print(f"🔄 Walk-Forward Test ({len(df) - split} Perioden)...")

    for i in range(split, len(df)):
        train_start = max(0, i - 200)
        X_train = X[train_start:i]
        y_train = y[train_start:i]
        X_test  = X[i:i+1]
        y_test  = y[i:i+1]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            eval_metric="mlogloss",
            verbosity=0
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({"actual": y_test[0], "predicted": pred[0]})

    results_df = pd.DataFrame(results)
    accuracy = (results_df["actual"] == results_df["predicted"]).mean()

    print(f"\n📊 Walk-Forward Ergebnis:")
    print(f"   Perioden getestet: {len(results_df)}")
    print(f"   Accuracy:          {accuracy:.2%}")
    print(f"\n{classification_report(results_df['actual'], results_df['predicted'], target_names=['Sell', 'Hold', 'Buy'])}")

if __name__ == "__main__":
    walk_forward_test()