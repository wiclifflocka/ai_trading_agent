# risk_management/position_sizing.py
from __future__ import annotations
import logging

# Configure logger
logger = logging.getLogger(__name__)

class RiskManagement:
    def __init__(self, client: BybitClient, account_balance):
        """
        Initialize the RiskManagement class.
        
        :param client: Instance of BybitClient to interact with the exchange
        :param account_balance: Current account balance in the base currency
        """
        self.client = client
        self.account_balance = account_balance  # Current account balance

    def calculate_position_size(self, risk_percentage=1):
        """
        Calculates position size based on the available balance and risk percentage.
        
        :param risk_percentage: Percentage of balance to risk per trade (default: 1%)
        :return: Position size in terms of the base currency
        """
        risk_amount = self.account_balance * (risk_percentage / 100)
        stop_loss_distance = self.calculate_stop_loss_distance()
        if stop_loss_distance <= 0:
            logger.warning("Stop-loss distance is zero or negative, cannot calculate position size.")
            return 0
        position_size = risk_amount / stop_loss_distance
        return position_size

    def calculate_stop_loss_distance(self):
        """
        Calculate the stop-loss distance for the trade using a static percentage of the current price.
        
        :return: Stop loss distance in terms of price
        """
        current_price = self.client.get_current_price("BTCUSD")  # Fetch current price from client
        stop_loss_percentage = 2  # Static stop-loss distance of 2%
        stop_loss_distance = current_price * (stop_loss_percentage / 100)
        return stop_loss_distance
