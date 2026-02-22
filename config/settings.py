# config/settings.py

# Aktie
TICKER   = "TSLA"
PERIOD   = "2y"
INTERVAL = "1d"

# Indikatoren
EMA_WINDOWS = [9, 21, 50]
RSI_WINDOW  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_WINDOW   = 20
ATR_WINDOW  = 14

# ML Modell
TEST_SIZE   = 0.2      # 20% der Daten zum Testen
RANDOM_SEED = 42

# Risiko
MAX_RISK_PER_TRADE = 0.02   # max 2% des Kapitals pro Trade
STOP_LOSS_ATR_MULT = 2.0    # Stop-Loss = 2x ATR