# main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetch_data import fetch_stock_data
from indicators.features import add_indicators
from model.predict import predict_signal
from config.settings import TICKER

def main():
    print("🚀 Claude Trading Bot gestartet!")
    print("=" * 50)

    # 1. Prüfen ob Modell vorhanden
    if not os.path.exists("model/saved/xgboost_model.pkl"):
        print("⚠️  Kein Modell gefunden! Trainiere zuerst:")
        print("   python -m model.train")
        return

    # 2. Signal generieren
    predict_signal()

if __name__ == "__main__":
    main()