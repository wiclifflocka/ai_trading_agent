import time
import numpy as np
from data_pipeline.bybit_api import BybitAPI

class ScalpingStrategy:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT", spread: float = 0.02, size: float = 0.01):
        """
        Implements an AI-powered scalping strategy.

        Args:
            api (BybitAPI): Instance of BybitAPI for placing orders and fetching data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            spread (float): Target bid-ask spread in percentage. Defaults to 0.02%.
            size (float): Order size in units. Defaults to 0.01.
        """
        self.api = api
        self.symbol = symbol
        self.spread = spread
        self.size = size

    def execute_scalping(self):
        """
        Places rapid bid-ask orders to capture small profits frequently.
        """
        book = self.api.get_order_book(self.symbol)
        if not book or 'bids' not in book or 'asks' not in book:
            print(f"Failed to fetch order book for {self.symbol}")
            return

        best_bid = float(max(book["bids"], key=lambda x: float(x[0]))[0])
        best_ask = float(min(book["asks"], key=lambda x: float(x[0]))[0])

        bid_price = round(best_bid * (1 - self.spread / 100), 2)
        ask_price = round(best_ask * (1 + self.spread / 100), 2)

        # Check for high-frequency trading conditions
        tick_speed = self.api.get_market_latency(self.symbol)
        if tick_speed is None or tick_speed > 1:  # If execution delay is high or unavailable, skip
            print("⚠️ High latency detected or latency data unavailable, skipping trades.")
            return

        self.api.place_limit_order(self.symbol, "buy", bid_price, self.size)
        self.api.place_limit_order(self.symbol, "sell", ask_price, self.size)

        print(f"✅ Scalping: BUY @ {bid_price}, SELL @ {ask_price}")

    def run(self):
        """
        Runs the scalping strategy in a continuous loop.
        """
        while True:
            self.execute_scalping()
            time.sleep(0.5)  # Fast execution loop

# Example usage
if __name__ == "__main__":
    api = BybitAPI()
    scalper = ScalpingStrategy(api)
    scalper.run()
