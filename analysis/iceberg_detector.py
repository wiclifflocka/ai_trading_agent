import time
from data_pipeline.bybit_api import BybitAPI

class IcebergDetector:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT"):
        """
        Initialize the IcebergDetector with a BybitAPI instance and trading symbol.

        Args:
            api (BybitAPI): Instance of BybitAPI for fetching order book data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
        """
        self.api = api
        self.symbol = symbol

    def detect_iceberg_orders(self, cycles: int = 10, refresh_threshold: int = 3) -> list:
        """
        Detects iceberg orders by identifying repetitive refreshes at the same price level.

        Args:
            cycles (int): Number of times to check the order book. Defaults to 10.
            refresh_threshold (int): Minimum number of refreshes at the same price level to consider it an iceberg. Defaults to 3.

        Returns:
            list: List of prices where iceberg orders are detected.
        """
        iceberg_data = {}

        for _ in range(cycles):
            data = self.api.get_order_book(self.symbol)
            if not data or 'bids' not in data or 'asks' not in data:
                print(f"Failed to fetch order book data for {self.symbol}")
                continue

            # Get top 5 bids and asks
            bids = sorted(data['bids'], key=lambda x: float(x[0]), reverse=True)[:5]
            asks = sorted(data['asks'], key=lambda x: float(x[0]))[:5]

            # Track price levels and their sizes
            for level in bids + asks:
                price, size = level
                price = float(price)  # Convert to float for consistency
                if price in iceberg_data:
                    iceberg_data[price].append(float(size))
                else:
                    iceberg_data[price] = [float(size)]

            time.sleep(1)  # Short delay before the next cycle

        # Detect iceberg orders: prices with at least 'refresh_threshold' different sizes
        detected_icebergs = [
            price for price, sizes in iceberg_data.items() if len(set(sizes)) >= refresh_threshold
        ]

        print(f"Detected Iceberg Orders at: {detected_icebergs}")
        return detected_icebergs

# Example usage
if __name__ == "__main__":
    api = BybitAPI()  # Initialize API
    detector = IcebergDetector(api)  # Create instance
    detector.detect_iceberg_orders()  # Detect iceberg orders
