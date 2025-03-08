# strategies/hold_strategy.py
from strategies.trading_strategy import TradingStrategy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HoldStrategy(TradingStrategy):
    def __init__(self, client=None, symbol=None):
        """
        Initialize the HoldStrategy.

        Args:
            client: An instance of BybitClient for API interactions (optional for hold strategy).
            symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        """
        super().__init__(client=client, symbol=symbol)
        logger.info(f"HoldStrategy initialized for {self.symbol}")

    def execute_trade(self):
        """
        Execute the hold strategy: no trade is placed, just log the decision.

        This method logs that the position is being held without taking any trading action.
        """
        logger.info(f"Holding position for {self.symbol}, no trade executed.")

# Example usage (optional, for testing)
if __name__ == "__main__":
    from bybit_client import BybitClient
    
    # Replace with your actual API credentials
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    
    # Initialize BybitClient
    client = BybitClient(api_key, api_secret, testnet=True)
    
    # Create and test HoldStrategy
    strategy = HoldStrategy(client=client, symbol="BTCUSDT")
    strategy.execute_trade()
