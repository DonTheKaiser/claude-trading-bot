# model/train_drl.py
import torch    
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import pickle

from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from model.trading_env import TradingEnv
from config.settings import TICKER, INTERVAL

def add_extra_features(df):
    df["Return_1d"]       = df["Close"].pct_change(1)
    df["Return_3d"]       = df["Close"].pct_change(3)
    df["Return_5d"]       = df["Close"].pct_change(5)
    df["Return_10d"]      = df["Close"].pct_change(10)
    df["Volume_MA10"]     = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"]    = df["Volume"] / df["Volume_MA10"]
    df["RSI_Overbought"]  = (df["RSI_14"] > 70).astype(int)
    df["RSI_Oversold"]    = (df["RSI_14"] < 30).astype(int)
    df["EMA_Cross_9_21"]  = (df["EMA_9"]  > df["EMA_21"]).astype(int)
    df["EMA_Cross_21_50"] = (df["EMA_21"] > df["EMA_50"]).astype(int)
    df["BB_Position"]     = (df["Close"] - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"])
    df["Volatility_10d"]  = df["Return_1d"].rolling(10).std()
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    print("📥 Lade Daten...")
    df = fetch_stock_data(TICKER, period="5y", interval=INTERVAL)
    if df is None:
        exit()

    df = add_indicators(df)
    df = add_extra_features(df)
    df.dropna(inplace=True)

    feature_cols = [
        "EMA_9", "EMA_21", "EMA_50",
        "RSI_14", "RSI_Overbought", "RSI_Oversold",
        "MACD", "MACD_Signal", "MACD_Diff",
        "BB_High", "BB_Low", "BB_Middle", "BB_Position",
        "ATR_14", "Volatility_10d",
        "Return_1d", "Return_3d", "Return_5d", "Return_10d",
        "Volume_Ratio", "EMA_Cross_9_21", "EMA_Cross_21_50"
    ]

    # Train/Test Split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df  = df.iloc[split:].reset_index(drop=True)

    print(f"📅 Training: {len(train_df)} Tage | Test: {len(test_df)} Tage")

    # Environment erstellen
    # Environment erstellen
    train_env = TradingEnv(train_df, feature_cols, initial_capital=10000)
    test_env  = TradingEnv(test_df,  feature_cols, initial_capital=10000)

    # PPO Agent — Deep Reinforcement Learning
    print("\n🧠 Trainiere PPO Agent (Deep RL)...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=0.0003,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",
        verbose=1
    )

    # Training — 200.000 Schritte
    model.learn(total_timesteps=50000)

    # Test
    print("\n📊 Teste Agent auf ungesehenen Daten...")
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)

    final_value = info["portfolio_value"]
    total_return = (final_value - 10000) / 10000 * 100
    print(f"\n💰 Startkapital:  $10,000")
    print(f"💰 Endkapital:    ${final_value:.2f}")
    print(f"📈 Rendite:       {total_return:.2f}%")
    print(f"🔄 Trades:        {info['total_trades']}")

    # Modell speichern
    os.makedirs("model/saved", exist_ok=True)
    model.save("model/saved/ppo_trading_agent")
    with open("model/saved/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("\n✅ DRL Agent gespeichert!")