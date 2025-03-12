# risk_management/max_drawdown.py
import logging

logger = logging.getLogger(__name__)

class MaxDrawdown:
    def __init__(self, client, initial_balance: float, max_drawdown: float = 0.2):
        """
        Initialize the MaxDrawdown monitor.

        :param client: BybitClient instance (for future use)
        :param initial_balance: The starting balance of the account
        :param max_drawdown: Maximum allowable drawdown percentage (default 20%)
        """
        self.client = client
        self.peak_balance = initial_balance  # Set peak balance to initial balance
        self.max_drawdown_threshold = max_drawdown  # Configurable threshold
        logger.info(f"MaxDrawdown initialized: initial_balance={initial_balance}, threshold={max_drawdown}")

    def update_peak_balance(self, current_balance: float) -> None:
        """
        Update the peak balance if the current balance exceeds it.

        :param current_balance: Current balance of the account
        """
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            logger.debug(f"Peak balance updated to {self.peak_balance}")

    def calculate_drawdown(self, current_balance: float) -> float:
        """
        Calculate the drawdown based on the peak balance.

        :param current_balance: Current balance of the account
        :return: Drawdown percentage (0 to 1), or 0 if peak_balance is invalid
        """
        if self.peak_balance <= 0:
            logger.warning("Peak balance is zero or negative, returning drawdown as 0")
            return 0
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        return max(drawdown, 0)  # Ensure drawdown is non-negative

    def check_drawdown(self, current_balance: float) -> bool:
        """
        Check if the current drawdown is within the threshold.

        :param current_balance: Current balance of the account
        :return: True if drawdown is within threshold (safe), False if exceeded
        """
        self.update_peak_balance(current_balance)
        drawdown = self.calculate_drawdown(current_balance)
        logger.debug(f"Drawdown check: peak={self.peak_balance}, current={current_balance}, drawdown={drawdown:.2%}")
        if drawdown >= self.max_drawdown_threshold:
            logger.warning(f"Max Drawdown Exceeded: {drawdown*100:.2f}% vs threshold {self.max_drawdown_threshold*100:.2f}%")
            return False
        return True
