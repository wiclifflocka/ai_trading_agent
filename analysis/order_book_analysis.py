import pandas as pd
import numpy as np
from data_pipeline.bybit_api import BybitAPI

class OrderBookAnalysis:
    """
    A class to analyze order book data from Bybit and compute trading signals.
    """
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT"):
        """
        Initialize the OrderBookAnalysis with a BybitAPI instance and trading symbol.

        Args:
            api (BybitAPI): Instance of BybitAPI for fetching order book data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
        """
        self.api = api
        self.symbol = symbol

    def calculate_order_flow_imbalance(self, levels: int = 5) -> float | None:
        """
        Calculates Order Flow Imbalance (OFI) to detect buying/selling pressure.

        Args:
            levels (int): Number of top bid/ask levels to consider. Defaults to 5.

        Returns:
            float | None: OFI value (positive = buy pressure, negative = sell pressure),
                         or None if data fetch fails.
        """
        data = self.api.get_order_book(self.symbol)

        if not data or 'bids' not in data or 'asks' not in data:
            print(f"Failed to fetch order book data for {self.symbol}")
            return None

        # Top N bid levels (sorted by price, descending)
        bids = sorted(data['bids'], key=lambda x: float(x[0]), reverse=True)[:levels]
        # Top N ask levels (sorted by price, ascending)
        asks = sorted(data['asks'], key=lambda x: float(x[0]))[:levels]

        # Calculate total volumes
        bid_volumes = sum(float(b[1]) for b in bids)
        ask_volumes = sum(float(a[1]) for a in asks)

        # Compute Order Flow Imbalance (OFI)
        ofi = bid_volumes - ask_volumes  # Positive: Buy pressure, Negative: Sell pressure

        print(f"Order Flow Imbalance for {self.symbol}: {ofi}")
        return ofi

# Example usage
if __name__ == "__main__":
    api = BybitAPI()  # Initialize API
    analysis = OrderBookAnalysis(api)  # Create instance
    ofi = analysis.calculate_order_flow_imbalance()  # Calculate OFI
