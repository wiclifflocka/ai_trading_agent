# risk_management/risk_manager.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, client, symbol: str = "BTCUSDT", max_loss: float = 0.02, volatility_threshold: float = 0.5):
        """
        :param client: BybitClient instance for API access
        :param symbol: Trading pair
        :param max_loss: Maximum percentage loss before exiting a trade
        :param volatility_threshold: Adjust spread based on volatility
        """
        self.client = client
        self.symbol = symbol
        self.max_loss = max_loss
        self.volatility_threshold = volatility_threshold
        self.current_volatility = 0.0
        logger.info(f"RiskManager initialized for {symbol}")

    def check_volatility(self) -> bool:
        """
        Measures market volatility and adjusts the spread dynamically.

        Returns:
            bool: True if high volatility detected, False otherwise
        """
        try:
            price_data = self.client.get_recent_trades(self.symbol, limit=100)
            if not price_data or len(price_data) < 2:
                logger.warning(f"Insufficient trade data for {self.symbol} volatility calculation")
                self.current_volatility = 0.0
                return False

            prices = [float(trade["price"]) for trade in price_data]
            returns = np.diff(prices) / prices[:-1]
            self.current_volatility = np.std(returns)
            logger.debug(f"Volatility for {self.symbol}: {self.current_volatility:.4f}")

            if self.current_volatility > self.volatility_threshold:
                logger.info("⚠️ High volatility detected, widening spread")
                return True
            return False
        except Exception as e:
            logger.error(f"Volatility check failed for {self.symbol}: {str(e)}")
            self.current_volatility = 0.0
            return False

    def apply_stop_loss(self, entry_price: float, position_type: str):
        """
        Implements stop-loss protection.

        Args:
            entry_price (float): Entry price of the position
            position_type (str): "long" or "short"
        """
        try:
            current_price = self.client.get_market_price(self.symbol)

            if position_type == "long" and (current_price < entry_price * (1 - self.max_loss)):
                logger.info("🚨 Stop-loss triggered! Closing long position.")
                self.client.close_position(self.symbol)
            elif position_type == "short" and (current_price > entry_price * (1 + self.max_loss)):
                logger.info("🚨 Stop-loss triggered! Closing short position.")
                self.client.close_position(self.symbol)
        except Exception as e:
            logger.error(f"Stop-loss check failed for {self.symbol}: {str(e)}")

if __name__ == "__main__":
    from bybit_client import BybitClient
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    testnet = os.getenv("USE_TESTNET", "True").lower() == "true"

    client = BybitClient(api_key, api_secret, testnet=testnet)
    risk_manager = RiskManager(client)
    volatility_high = risk_manager.check_volatility()
    print(f"High volatility: {volatility_high}")
