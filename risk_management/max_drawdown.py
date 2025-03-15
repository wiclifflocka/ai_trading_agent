# risk_management/max_drawdown.py
import logging
from bybit_client import BybitClient

logger = logging.getLogger(__name__)

class MaxDrawdown:
    """
    Monitor and enforce a maximum drawdown limit based on account balance.

    Attributes:
        client (BybitClient): Bybit API client instance
        initial_balance (float): Starting balance of the account
        max_drawdown (float): Maximum allowable drawdown percentage (default 20%)
    """

    def __init__(self, client: BybitClient, initial_balance: float, max_drawdown: float = 0.2):
        self.client = client
        self.initial_balance = initial_balance
        self.max_drawdown = max_drawdown
        self.peak_balance = initial_balance
        logger.info(f"MaxDrawdown initialized: initial_balance={initial_balance}, threshold={max_drawdown}")

    def check_drawdown(self, current_balance: float) -> bool:
        """
        Check if the current drawdown is within the threshold.

        Args:
            current_balance (float): Current balance of the account

        Returns:
            bool: True if drawdown is within threshold (safe), False if exceeded
        """
        # Update peak balance
        self.peak_balance = max(self.peak_balance, current_balance)
        # Calculate drawdown
        if self.peak_balance <= 0:
            logger.warning("Peak balance is zero or negative, returning True to avoid division by zero")
            return True
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        drawdown = max(drawdown, 0)  # Ensure non-negative
        is_within_limit = drawdown <= self.max_drawdown
        if not is_within_limit:
            logger.warning(f"Max Drawdown Exceeded: {drawdown:.2%} vs threshold {self.max_drawdown:.2%}")
        else:
            logger.debug(f"Drawdown check: peak={self.peak_balance}, current={current_balance}, drawdown={drawdown:.2%}")
        return is_within_limit
