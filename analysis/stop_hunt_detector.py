import time
from data_pipeline.bybit_api import BybitAPI

class StopHuntDetector:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT", lookback_period: int = 60):
        """
        Initialize the StopHuntDetector with a BybitAPI instance, trading symbol, and lookback period.

        Args:
            api (BybitAPI): Instance of BybitAPI for fetching market data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            lookback_period (int): Number of seconds to look back for price movements. Defaults to 60.
        """
        self.api = api
        self.symbol = symbol
        self.lookback_period = lookback_period

    def detect_stop_hunts(self, threshold: float = 0.005) -> bool:
        """
        Detects potential stop hunts by checking for rapid price movements beyond a threshold.

        Args:
            threshold (float): Percentage price change threshold to consider a stop hunt. Defaults to 0.5%.

        Returns:
            bool: True if a stop hunt is detected, False otherwise.
        """
        # Fetch recent price data (e.g., last 2 one-minute candles)
        recent_data = self.api.get_recent_price_data(self.symbol, interval="1m", limit=2)
        if not recent_data or len(recent_data) < 2:
            print(f"Failed to fetch price data for {self.symbol}")
            return False

        # Extract closing prices from the two most recent candles
        prev_close = float(recent_data[0]['close'])
        current_close = float(recent_data[1]['close'])

        # Calculate percentage price change
        price_change = (current_close - prev_close) / prev_close

        # Check if the absolute price change exceeds the threshold
        if abs(price_change) > threshold:
            print(f"Potential stop hunt detected: {price_change*100:.2f}% change")
            return True
        else:
            print(f"No stop hunt detected: {price_change*100:.2f}% change")
            return False

# Example usage
if __name__ == "__main__":
    api = BybitAPI()  # Initialize API
    detector = StopHuntDetector(api)  # Create instance
    detector.detect_stop_hunts()  # Detect stop hunts
