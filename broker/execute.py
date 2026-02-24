# broker/execute.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.alpaca_api import get_client, get_position
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config.settings import TICKER, MAX_RISK_PER_TRADE

def execute_signal(signal: int, current_price: float):
    """
    signal: 0=Sell, 1=Hold, 2=Buy
    """
    client = get_client()
    account = client.get_account()
    cash = float(account.cash)
    portfolio = float(account.portfolio_value)

    print(f"💰 Cash: ${cash:.2f} | Portfolio: ${portfolio:.2f}")

    position = get_position(TICKER)

    if signal == 2:  # BUY
        if position is not None:
            print("📭 Position bereits offen — kein Kauf")
            return

        # Max 2% Risiko pro Trade
        risk_amount = portfolio * MAX_RISK_PER_TRADE
        qty = int(risk_amount / current_price)

        if qty < 1:
            print("⚠️  Zu wenig Kapital für eine Aktie")
            return

        order = MarketOrderRequest(
            symbol=TICKER,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        client.submit_order(order)
        print(f"✅ BUY: {qty}x {TICKER} @ ~${current_price:.2f}")

    elif signal == 0:  # SELL
        if position is None:
            print("📭 Keine Position zum Verkaufen")
            return

        order = MarketOrderRequest(
            symbol=TICKER,
            qty=int(float(position.qty)),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        client.submit_order(order)
        print(f"✅ SELL: {position.qty}x {TICKER} @ ~${current_price:.2f}")

    else:  # HOLD
        print("🟡 HOLD — keine Aktion")

if __name__ == "__main__":
    from model.predict import predict_signal
    predict_signal()