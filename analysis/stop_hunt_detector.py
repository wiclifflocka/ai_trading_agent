# analysis/stop_hunt_detector.py
import pandas as pd
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

class StopHuntDetector:
    def __init__(self, client: BybitClient, symbol: str = "BTCUSDT", lookback_period: int = 60):
        """
        Initialize the StopHuntDetector with a BybitClient instance, trading symbol, and lookback period.

        Args:
            client (BybitClient): Instance of BybitClient for fetching market data.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            lookback_period (int): Number of seconds to look back for price movements. Defaults to 60.
        """
        self.client = client
        self.symbol = symbol
        self.lookback_period = lookback_period
        logger.info(f"StopHuntDetector initialized for {symbol} with lookback period {lookback_period}s")

    def detect_stop_hunts(self, recent_data: pd.DataFrame, threshold: float = 0.005) -> bool:
        """
        Detects potential stop hunts by checking for rapid price movements beyond a threshold.

        Args:
            recent_data (pd.DataFrame): OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            threshold (float): Percentage price change threshold to consider a stop hunt. Defaults to 0.5%.

        Returns:
            bool: True if a stop hunt is detected, False otherwise.
        """
        try:
            if not isinstance(recent_data, pd.DataFrame):
                logger.error("Input must be a pandas DataFrame")
                return False

            if len(recent_data) < 2:
                logger.warning(f"Insufficient data for stop hunt detection: {len(recent_data)} rows")
                return False

            # Ensure data is numeric
            recent_data = recent_data.astype(float)

            # Extract closing prices from the first and last rows
            prev_close = recent_data['close'].iloc[0]
            current_close = recent_data['close'].iloc[-1]

            # Calculate percentage price change
            price_change = (current_close - prev_close) / prev_close

            # Check if the absolute price change exceeds the threshold
            if abs(price_change) > threshold:
                logger.info(f"Potential stop hunt detected: {price_change*100:.2f}% change")
                return True
            else:
                logger.debug(f"No stop hunt detected: {price_change*100:.2f}% change")
                return False
        except Exception as e:
            logger.error(f"Stop hunt detection failed: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    client = BybitClient(api_key, api_secret, testnet=True)
    detector = StopHuntDetector(client, "BTCUSDT")
    
    # Fetch sample data for testing
    ohlcv_data = client.get_historical_data("BTCUSDT", interval="1m", limit=10)
    if ohlcv_data:
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = detector.detect_stop_hunts(df)
        print(f"Stop hunt detected: {result}")
    else:
        print("Failed to fetch sample data")
