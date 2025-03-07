import time
import numpy as np
from data_pipeline.bybit_api import BybitAPI
from analysis.ofi_analysis import OFIAnalysis  # Import the class instead

class MarketMaker:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT", spread: float = 0.05, size: float = 0.01, max_position: float = 0.1):
        """
        Initializes the market-making engine.

        Args:
            api (BybitAPI): Instance of BybitAPI for placing orders and fetching data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            spread (float): Target bid-ask spread in percentage. Defaults to 0.05%.
            size (float): Order size in units. Defaults to 0.01.
            max_position (float): Maximum position size to prevent excessive exposure. Defaults to 0.1.
        """
        self.api = api
        self.symbol = symbol
        self.spread = spread
        self.size = size
        self.max_position = max_position
        self.position = 0  # Current market exposure
        self.ofi_analyzer = OFIAnalysis(api, symbol)  # Initialize OFIAnalysis with api and symbol

    def place_orders(self):
        """
        Dynamically places bid and ask orders to capture the spread.
        """
        book = self.api.get_order_book(self.symbol)
        if not book or 'bids' not in book or 'asks' not in book:
            print(f"Failed to fetch order book for {self.symbol}")
            return

        best_bid = float(max(book["bids"], key=lambda x: float(x[0]))[0])
        best_ask = float(min(book["asks"], key=lambda x: float(x[0]))[0])

        # Calculate optimal bid/ask prices
        bid_price = round(best_bid * (1 - self.spread / 100), 2)
        ask_price = round(best_ask * (1 + self.spread / 100), 2)

        # Check market order flow to adjust pricing
        ofi = self.ofi_analyzer.compute_order_flow_imbalance()
        if ofi is not None:
            if ofi > 0:
                bid_price += 0.1  # Adjust bid up if buying pressure is high
            elif ofi < 0:
                ask_price -= 0.1  # Adjust ask down if selling pressure is high

        # Risk Management: Limit position size
        if self.position + self.size > self.max_position:
            print("⚠️ Position limit reached, skipping new bids")
            return

        # Place bid and ask orders
        self.api.place_limit_order(self.symbol, "buy", bid_price, self.size)
        self.api.place_limit_order(self.symbol, "sell", ask_price, self.size)

        print(f"✅ Placed orders: BUY @ {bid_price}, SELL @ {ask_price}")

    def monitor_orders(self):
        """
        Continuously monitors the market and adjusts orders.
        """
        while True:
            self.place_orders()
            time.sleep(1)  # Adjust frequency as needed

# Example usage
if __name__ == "__main__":
    api = BybitAPI()
    maker = MarketMaker(api)
    maker.monitor_orders()
