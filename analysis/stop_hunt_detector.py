import numpy as np
import pandas as pd
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

def detect_stop_hunt_zones(symbol="BTCUSDT"):
    """
    Identifies stop-loss hunting zones where price is likely to be manipulated.
    :param symbol: Trading pair
    :return: Stop-hunt levels
    """
    data = api.get_order_book(symbol)

    if not data:
        return None

    bids = sorted(data['bids'], key=lambda x: x[0], reverse=True)[:5]
    asks = sorted(data['asks'], key=lambda x: x[0])[:5]

    bid_sizes = np.array([b[1] for b in bids])
    ask_sizes = np.array([a[1] for a in asks])

    # Large liquidity clusters indicate stop-loss levels
    large_bid_zone = bids[np.argmax(bid_sizes)][0]
    large_ask_zone = asks[np.argmax(ask_sizes)][0]

    print(f"Potential Stop-Hunt Levels: Buy Stops at {large_ask_zone}, Sell Stops at {large_bid_zone}")
    return large_ask_zone, large_bid_zone

# Example usage
if __name__ == "__main__":
    detect_stop_hunt_zones()

