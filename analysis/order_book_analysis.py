import pandas as pd
import numpy as np
from data_pipeline.bybit_api import BybitAPI

# Initialize API
api = BybitAPI()

def calculate_order_flow_imbalance(symbol="BTCUSDT"):
    """
    Calculates Order Flow Imbalance (OFI) to detect buying/selling pressure.
    :param symbol: Trading pair (e.g., "BTCUSDT")
    :return: OFI value
    """
    data = api.get_order_book(symbol)
    
    if not data:
        return None
    
    bids = sorted(data['bids'], key=lambda x: x[0], reverse=True)[:5]  # Top 5 bid levels
    asks = sorted(data['asks'], key=lambda x: x[0])[:5]  # Top 5 ask levels

    bid_volumes = sum([b[1] for b in bids])
    ask_volumes = sum([a[1] for a in asks])

    ofi = bid_volumes - ask_volumes  # Positive: Buy pressure, Negative: Sell pressure
    
    print(f"Order Flow Imbalance: {ofi}")
    return ofi

# Example usage
if __name__ == "__main__":
    ofi = calculate_order_flow_imbalance()

