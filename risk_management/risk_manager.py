import numpy as np
import os
from dotenv import load_dotenv
from data_pipeline.bybit_api import BybitAPI
from bybit_client import BybitClient

# Load API keys from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# Initialize API
api = BybitAPI(API_KEY, API_SECRET, testnet=USE_TESTNET)


class RiskManager:
    def __init__(self, symbol="BTCUSDT", max_loss=0.02, volatility_threshold=0.5):
        """
        :param symbol: Trading pair
        :param max_loss: Maximum percentage loss before exiting a trade
        :param volatility_threshold: Adjust spread based on volatility
        """
        self.symbol = symbol
        self.max_loss = max_loss
        self.volatility_threshold = volatility_threshold

    def check_volatility(self):
        """
        Measures market volatility and adjusts the spread dynamically.
        """
        price_data = api.get_recent_trades(self.symbol)
        returns = np.diff([trade["price"] for trade in price_data])

        volatility = np.std(returns)
        if volatility > self.volatility_threshold:
            print("⚠️ High volatility detected, widening spread")
            return True
        return False

    def apply_stop_loss(self, entry_price, position_type):
        """
        Implements stop-loss protection.
        """
        current_price = api.get_current_price(self.symbol)

        if position_type == "long" and (current_price < entry_price * (1 - self.max_loss)):
            print("🚨 Stop-loss triggered! Closing long position.")
            api.close_position(self.symbol)
        elif position_type == "short" and (current_price > entry_price * (1 + self.max_loss)):
            print("🚨 Stop-loss triggered! Closing short position.")
            api.close_position(self.symbol)

if __name__ == "__main__":
    risk_manager = RiskManager()
    risk_manager.check_volatility()

