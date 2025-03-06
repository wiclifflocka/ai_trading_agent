# risk_management/max_drawdown.py
from __future__ import annotations

class MaxDrawdown:
    def __init__(self, client: BybitClient, initial_balance):
        self.client = client
        self.initial_balance = initial_balance
        self.max_drawdown_threshold = 0.2  # e.g., 20% max drawdown

    def calculate_drawdown(self, current_balance):
        """
        Calculates the drawdown based on the current balance.
        :param current_balance: Current balance of the account
        :return: The drawdown percentage
        """
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        return drawdown

    def check_drawdown(self, current_balance):
        """
        Checks if the current drawdown exceeds the threshold.
        :param current_balance: Current balance of the account
        :return: True if drawdown exceeds threshold, False otherwise
        """
        drawdown = self.calculate_drawdown(current_balance)
        if drawdown >= self.max_drawdown_threshold:
            logger.warning(f"Max Drawdown Exceeded: {drawdown*100:.2f}%")
            return True
        return False

