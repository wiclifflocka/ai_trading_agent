# execution/market_maker.py
import time
import threading
import logging

logger = logging.getLogger(__name__)

class MarketMaker:
    def __init__(self, client, symbol: str = "BTCUSDT", spread: float = 0.05, size: float = 0.01, position_info=None, risk_components=None):
        """
        Implements a market making strategy.

        Args:
            client: BybitClient instance for placing orders and fetching data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            spread (float): Target bid-ask spread in percentage. Defaults to 0.05%.
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
        """Start the market making strategy thread."""
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"MarketMaker started for {self.symbol}")

    def run(self):
        """Runs the market making strategy in a continuous loop."""
        while self.running:
            self.execute_market_making()
            time.sleep(1)  # Adjust timing as needed

    def execute_market_making(self):
        """Places bid and ask orders around the market price."""
        if self.position_info.get('size', 0):  # Skip if position exists
            return

        try:
            book = self.client.get_order_book(self.symbol)
            if not book or 'bids' not in book or 'asks' not in book:
                logger.warning(f"Failed to fetch order book for {self.symbol}")
                return

            best_bid = float(max(book["bids"], key=lambda x: float(x[0]))[0])
            best_ask = float(min(book["asks"], key=lambda x: float(x[0]))[0])
            mid_price = (best_bid + best_ask) / 2

            bid_price = round(mid_price * (1 - self.spread / 100), 2)
            ask_price = round(mid_price * (1 + self.spread / 100), 2)

            # Apply risk management if available
            if 'position_sizing' in self.risk_components:
                size = self.risk_components['position_sizing'].calculate_position_size(self.size)
            else:
                size = self.size

            self.client.place_order(self.symbol, "Buy", size, "Limit", price=bid_price)
            self.client.place_order(self.symbol, "Sell", size, "Limit", price=ask_price)
            logger.info(f"Market Making: BUY @ {bid_price}, SELL @ {ask_price}")
        except Exception as e:
            logger.error(f"Market making execution failed: {str(e)}")

    def stop(self):
        """Stop the market making strategy thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        logger.info(f"MarketMaker stopped for {self.symbol}")

if __name__ == "__main__":
    from bybit_client import BybitClient
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    mm = MarketMaker(client)
    mm.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mm.stop()
