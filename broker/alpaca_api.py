# broker/alpaca_api.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

load_dotenv()

def get_client():
    return TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        paper=True
    )

def get_account():
    client = get_client()
    account = client.get_account()
    print(f"💰 Kontostand:  ${float(account.cash):.2f}")
    print(f"📊 Portfolio:   ${float(account.portfolio_value):.2f}")
    return account

def place_order(ticker: str, qty: int, side: str):
    client = get_client()
    order_data = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    order = client.submit_order(order_data)
    print(f"✅ Order: {side.upper()} {qty}x {ticker}")
    return order

def get_position(ticker: str):
    client = get_client()
    try:
        position = client.get_open_position(ticker)
        print(f"📈 {ticker}: {position.qty} Aktien @ ${float(position.avg_entry_price):.2f}")
        return position
    except:
        print(f"📭 Keine offene Position in {ticker}")
        return None

if __name__ == "__main__":
    get_account()