# risk_management/position_sizing.py
from __future__ import annotations
import logging

# Configure logging to match main.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from bybit_client import BybitClient

class RiskManagement:
    def __init__(self, client: BybitClient, account_balance: float, symbol: str = "BTCUSDT"):
        """
        Initialize the RiskManagement class.

        Args:
            client (BybitClient): Instance of BybitClient
            account_balance (float): Current account balance in USDT
            symbol (str): Trading pair (default: "BTCUSDT")
        """
        self.client = client
        self.account_balance = account_balance
        self.symbol = symbol
        self.min_position_size = 0.001  # Bybit minimum precision for BTCUSDT

    def calculate_position_size(self, risk_percentage: float = 1.0) -> float:
        """
        Calculates position size based on balance and risk percentage, enforcing minimum size.

        Args:
            risk_percentage (float): Percentage of balance to risk per trade (default: 1%)

        Returns:
            float: Position size in terms of the base currency (BTC)
        """
        try:
            risk_amount = self.account_balance * (risk_percentage / 100)
            stop_loss_distance = self.calculate_stop_loss_distance()
            if stop_loss_distance <= 0:
                logger.warning("Stop-loss distance is zero or negative, returning minimum size.")
                return self.min_position_size

            position_size = risk_amount / stop_loss_distance
            if position_size < self.min_position_size:
                logger.info(f"Position size {position_size} below minimum {self.min_position_size}. Adjusting to minimum.")
                position_size = self.min_position_size

            logger.info(f"Calculated position size: {position_size} for risk {risk_amount}")
            return position_size
        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            return self.min_position_size  # Fallback to minimum on error

    def calculate_stop_loss_distance(self) -> float:
        """
        Calculate the stop-loss distance using a static percentage of the current price.

        Returns:
            float: Stop loss distance in terms of price
        """
        try:
            current_price = self.client.get_current_price(self.symbol)
            if current_price <= 0:
                logger.warning("Current price is zero or negative, returning 0.")
                return 0.0
            stop_loss_percentage = 2.0  # Static 2% stop-loss
            stop_loss_distance = current_price * (stop_loss_percentage / 100)
            return stop_loss_distance
        except Exception as e:
            logger.error(f"Stop-loss distance calculation failed: {str(e)}")
            return 0.0
