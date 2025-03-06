import time
import numpy as np
from data_pipeline.bybit_api import BybitAPI
from analysis.ofi_analysis import compute_order_flow_imbalance

api = BybitAPI()

class MarketMaker:
    def __init__(self, symbol="BTCUSDT", spread=0.05, size=0.01, max_position=0.1):
        """
        Initializes the market-making engine.
        :param symbol: Trading pair
        :param spread: Target bid-ask spread in percentage
        :param size: Order size
        :param max_position: Maximum position size to prevent excessive exposure
        """
        self.symbol = symbol
        self.spread = spread
        self.size = size
        self.max_position = max_position
        self.position = 0  # Current market exposure

    def place_orders(self):
        """
        Dynamically places bid and ask orders to capture the spread.
        """
        book = api.get_order_book(self.symbol)
        if not book:
            return

        best_bid = max(book["bids"], key=lambda x: x[0])[0]
        best_ask = min(book["asks"], key=lambda x: x[0])[0]

        # Calculate optimal bid/ask prices
        bid_price = round(best_bid * (1 - self.spread / 100), 2)
        ask_price = round(best_ask * (1 + self.spread / 100), 2)

        # Check market order flow to adjust pricing
        ofi = compute_order_flow_imbalance(self.symbol)
        if ofi > 0:
            bid_price += 0.1  # Adjust bid up if buying pressure is high
        elif ofi < 0:
            ask_price -= 0.1  # Adjust ask down if selling pressure is high

        # Risk Management: Limit position size
        if self.position + self.size > self.max_position:
            print("⚠️ Position limit reached, skipping new bids")
            return

        # Place bid and ask orders
        api.place_limit_order(self.symbol, "buy", bid_price, self.size)
        api.place_limit_order(self.symbol, "sell", ask_price, self.size)

        print(f"✅ Placed orders: BUY @ {bid_price}, SELL @ {ask_price}")

    def monitor_orders(self):
        """
        Continuously monitors the market and adjusts orders.
        """
        while True:
            self.place_orders()
            time.sleep(1)  # Adjust frequency as needed

if __name__ == "__main__":
    maker = MarketMaker()
    maker.monitor_orders()

