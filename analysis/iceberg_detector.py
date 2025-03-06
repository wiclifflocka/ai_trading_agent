import numpy as np
import pandas as pd
from data_pipeline.bybit_api import BybitAPI
import time

api = BybitAPI()

def detect_iceberg_orders(symbol="BTCUSDT", refresh_threshold=3):
    """
    Detects iceberg orders by identifying repetitive refreshes at the same price level.
    :param symbol: Trading pair
    :param refresh_threshold: Number of times an order at the same price level refreshes
    :return: Iceberg detection result
    """
    iceberg_data = {}

    for _ in range(10):  # Check for iceberg behavior over 10 cycles
        data = api.get_order_book(symbol)
        if not data:
            continue

        bids = sorted(data['bids'], key=lambda x: x[0], reverse=True)[:5]
        asks = sorted(data['asks'], key=lambda x: x[0])[:5]

        for level in bids + asks:
            price, size = level
            if price in iceberg_data:
                iceberg_data[price].append(size)
            else:
                iceberg_data[price] = [size]

        time.sleep(1)  # Short delay before next cycle

    detected_icebergs = [price for price, sizes in iceberg_data.items() if len(set(sizes)) >= refresh_threshold]

    print(f"Detected Iceberg Orders at: {detected_icebergs}")
    return detected_icebergs

# Example usage
if __name__ == "__main__":
    detect_iceberg_orders()

