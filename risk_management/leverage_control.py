# risk_management/leverage_control.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

class LeverageControl:
    def __init__(self, client):  # Type hint removed since BybitClient isn’t imported
        self.client = client
        self.max_leverage = 10  # Maximum leverage allowed
        logger.info(f"LeverageControl initialized with max_leverage={self.max_leverage}")

    def check_and_set_leverage(self, pair: str) -> bool:
        """
        Checks the leverage for the current position and adjusts it to the safe limit.

        :param pair: The trading pair (e.g., 'BTCUSDT')
        :return: True if leverage is safe or adjusted successfully, False if an error occurs
        """
        try:
            # Fetch current leverage
            current_leverage_response = self.client.get_leverage(pair)
            logger.debug(f"Raw leverage response for {pair}: {current_leverage_response}")

            # Handle string response (e.g., "5x")
            if isinstance(current_leverage_response, str):
                current_leverage = int(current_leverage_response.replace('x', ''))
            else:
                current_leverage = int(current_leverage_response)

            logger.info(f"Leverage for {pair}: {current_leverage}x")

            # Check and adjust leverage if necessary
            if current_leverage > self.max_leverage:
                logger.info(f"Current leverage {current_leverage}x exceeds max {self.max_leverage}x, adjusting...")
                self.client.set_leverage(pair, self.max_leverage)
                logger.info(f"Leverage reduced to {self.max_leverage}x for {pair}")
            else:
                logger.debug(f"Leverage {current_leverage}x is within limit {self.max_leverage}x")

            return True
        except Exception as e:
            logger.error(f"Leverage adjustment failed for {pair}: {str(e)}")
            return False
