import time
import numpy as np
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

class Scalper:
    def __init__(self, symbol="BTCUSDT", spread=0.02, size=0.01):
        """
        Implements an AI-powered scalping strategy.
        :param symbol: Trading pair
        :param spread: Target bid-ask spread in percentage
        :param size: Order size
        """
        self.symbol = symbol
        self.spread = spread
        self.size = size

    def execute_scalping(self):
        """
        Places rapid bid-ask orders to capture small profits frequently.
        """
        book = api.get_order_book(self.symbol)
        if not book:
            return

        best_bid = max(book["bids"], key=lambda x: x[0])[0]
        best_ask = min(book["asks"], key=lambda x: x[0])[0]

        bid_price = round(best_bid * (1 - self.spread / 100), 2)
        ask_price = round(best_ask * (1 + self.spread / 100), 2)

        # Check for high-frequency trading conditions
        tick_speed = api.get_market_latency(self.symbol)
        if tick_speed > 1:  # If execution delay is high, skip placing new orders
            print("⚠️ High latency detected, skipping trades.")
            return

        api.place_limit_order(self.symbol, "buy", bid_price, self.size)
        api.place_limit_order(self.symbol, "sell", ask_price, self.size)

        print(f"✅ Scalping: BUY @ {bid_price}, SELL @ {ask_price}")

    def run(self):
        while True:
            self.execute_scalping()
            time.sleep(0.5)  # Fast execution loop

if __name__ == "__main__":
    scalper = Scalper()
    scalper.run()

