# execution/scalping_strategy.py
import time
import numpy as np
import threading

class ScalpingStrategy:
    def __init__(self, client, symbol: str = "BTCUSDT", spread: float = 0.02, size: float = 0.01, position_info=None, risk_components=None):
        """
        Implements an AI-powered scalping strategy.

        Args:
            client: BybitClient instance for placing orders and fetching data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            spread (float): Target bid-ask spread in percentage. Defaults to 0.02%.
            size (float): Order size in units. Defaults to 0.01.
            position_info: Shared position info from TradingSystem.
            risk_components: Shared risk management components from TradingSystem.
        """
        self.client = client
        self.symbol = symbol
        self.spread = spread
        self.size = size
        self.running = False
        self.thread = None
        self.position_info = position_info or {}
        self.risk_components = risk_components or {}

    def start(self):
        """Start the scalping strategy thread."""
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        print(f"ScalpingStrategy started for {self.symbol}")

    def execute_scalping(self):
        """
        Places rapid bid-ask orders to capture small profits frequently.
        """
        if self.position_info.get('size', 0):  # Skip if position exists
            return

        book = self.client.get_order_book(self.symbol)
        if not book or 'bids' not in book or 'asks' not in book:
            print(f"Failed to fetch order book for {self.symbol}")
            return

        best_bid = float(max(book["bids"], key=lambda x: float(x[0]))[0])
        best_ask = float(min(book["asks"], key=lambda x: float(x[0]))[0])

        bid_price = round(best_bid * (1 - self.spread / 100), 2)
        ask_price = round(best_ask * (1 + self.spread / 100), 2)

        self.client.place_order(self.symbol, "Buy", self.size, "Limit", price=bid_price)
        self.client.place_order(self.symbol, "Sell", self.size, "Limit", price=ask_price)
        print(f"✅ Scalping: BUY @ {bid_price}, SELL @ {ask_price}")

    def run(self):
        """
        Runs the scalping strategy in a continuous loop.
        """
        while self.running:
            self.execute_scalping()
            time.sleep(0.5)  # Fast execution loop

    def stop(self):
        """Stop the scalping strategy thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print(f"ScalpingStrategy stopped for {self.symbol}")

if __name__ == "__main__":
    from bybit_client import BybitClient
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    scalper = ScalpingStrategy(client)
    scalper.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scalper.stop()
