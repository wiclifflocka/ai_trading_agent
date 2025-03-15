# analysis/order_timing.py
import numpy as np
import logging
from typing import Optional
from bybit_client import BybitClient

# Set up logging consistent with other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderTimingOptimizer:
    def __init__(self, client: BybitClient, symbol: str = "BTCUSDT", threshold: float = 5.0):
        """
        Detects large institutional orders by monitoring order flow imbalance (OFI).

        Args:
            client (BybitClient): Instance of BybitClient for API access.
            symbol (str): Trading pair (default: "BTCUSDT").
            threshold (float): Minimum imbalance to trigger action (default: 5.0).
        """
        self.client = client
        self.symbol = symbol
        self.threshold = threshold
        logger.info(f"OrderTimingOptimizer initialized for {symbol} with threshold: {threshold}")

    def detect_large_orders(self) -> Optional[str]:
        """
        Identifies large trades and order flow imbalances to anticipate big moves.

        Returns:
            str or None: "BUY" or "SELL" if large order detected, None otherwise.
        """
        try:
            trades = self.client.get_recent_trades(self.symbol, limit=50)
            if not trades:
                logger.warning(f"No recent trades fetched for {self.symbol}")
                return None

            # Use 'qty' instead of 'size' based on Bybit API response
            volumes = [float(trade.get('qty', 0)) for trade in trades]  # Safely get 'qty', default to 0 if missing
            buy_vol = sum(v for v, t in zip(volumes, trades) if t.get('side', '').capitalize() == "Buy")
            sell_vol = sum(v for v, t in zip(volumes, trades) if t.get('side', '').capitalize() == "Sell")

            ofi = buy_vol - sell_vol  # Order flow imbalance
            if abs(ofi) > self.threshold:
                direction = "BUY" if ofi > 0 else "SELL"
                logger.info(f"Large {direction} order detected! OFI: {ofi:.2f} (Buy Vol: {buy_vol:.2f}, Sell Vol: {sell_vol:.2f})")
                return direction
            else:
                logger.debug(f"No large orders detected. OFI: {ofi:.2f} (Buy Vol: {buy_vol:.2f}, Sell Vol: {sell_vol:.2f})")
                return None

        except Exception as e:
            logger.error(f"Failed to detect large orders: {str(e)}", exc_info=True)
            return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    client = BybitClient(api_key, api_secret, testnet=True)
    timing_optimizer = OrderTimingOptimizer(client, "BTCUSDT", threshold=5.0)
    result = timing_optimizer.detect_large_orders()
    print(f"Detected direction: {result}")
