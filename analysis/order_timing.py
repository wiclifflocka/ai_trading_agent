# analysis/order_timing.py
import numpy as np

class OrderTimingOptimizer:
    def __init__(self, client, symbol="BTCUSDT", threshold=5):
        """
        Detects large institutional orders by monitoring order flow imbalance (OFI).
        :param client: BybitClient instance for API access
        :param symbol: Trading pair
        :param threshold: Minimum imbalance to trigger action
        """
        self.client = client
        self.symbol = symbol
        self.threshold = threshold
        print(f"OrderTimingOptimizer initialized for symbol: {self.symbol} with threshold: {self.threshold}")

    def detect_large_orders(self):
        """
        Identifies large trades and order flow imbalances to anticipate big moves.
        Returns:
            str or None: "BUY" or "SELL" if large order detected, None otherwise
        """
        trades = self.client.get_recent_trades(self.symbol, limit=50)  # Increased limit for better analysis
        if not trades:
            return None

        volumes = [float(trade["size"]) for trade in trades]  # Convert to float for calculation
        buy_vol = sum(v for v, t in zip(volumes, trades) if t["side"] == "Buy")  # Capitalized "Buy"
        sell_vol = sum(v for v, t in zip(volumes, trades) if t["side"] == "Sell")  # Capitalized "Sell"

        ofi = buy_vol - sell_vol  # Order flow imbalance
        if abs(ofi) > self.threshold:
            direction = "BUY" if ofi > 0 else "SELL"
            print(f"🚨 Large {direction} order detected! OFI: {ofi}")
            return direction
        return None

if __name__ == "__main__":
    from bybit_client import BybitClient
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    timing_optimizer = OrderTimingOptimizer(client)
    result = timing_optimizer.detect_large_orders()
    print(f"Detected direction: {result}")
