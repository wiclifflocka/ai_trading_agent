import numpy as np
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

class OrderTimingOptimizer:
    def __init__(self, symbol="BTCUSDT", threshold=5):
        """
        Detects large institutional orders by monitoring order flow imbalance (OFI).
        :param symbol: Trading pair
        :param threshold: Minimum imbalance to trigger action
        """
        self.symbol = symbol
        self.threshold = threshold

    def detect_large_orders(self):
        """
        Identifies large trades and order flow imbalances to anticipate big moves.
        """
        trades = api.get_recent_trades(self.symbol)
        volumes = [trade["size"] for trade in trades]
        buy_vol = sum(v for v, t in zip(volumes, trades) if t["side"] == "buy")
        sell_vol = sum(v for v, t in zip(volumes, trades) if t["side"] == "sell")

        ofi = buy_vol - sell_vol  # Order flow imbalance
        if abs(ofi) > self.threshold:
            direction = "BUY" if ofi > 0 else "SELL"
            print(f"🚨 Large {direction} order detected! OFI: {ofi}")
            return direction
        return None

if __name__ == "__main__":
    timing_optimizer = OrderTimingOptimizer()
    timing_optimizer.detect_large_orders()

