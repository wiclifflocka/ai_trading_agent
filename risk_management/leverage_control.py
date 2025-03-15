# risk_management/leverage_control.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

class LeverageControl:
    def __init__(self, client):
        self.client = client
        self.max_leverage = 10  # Maximum leverage allowed
        self.default_leverage = 1  # Fallback leverage value
        logger.info(f"LeverageControl initialized with max_leverage={self.max_leverage}, default_leverage={self.default_leverage}")

    def check_and_set_leverage(self, pair: str, category: str = "linear") -> bool:
        """
        Checks the leverage for the current position and adjusts it to the safe limit if applicable.

        :param pair: The trading pair (e.g., 'BTCUSDT')
        :param category: Market category ('linear', 'inverse', or 'spot')
        :return: True if leverage is safe or adjusted successfully, False if an error occurs
        """
        # Skip leverage adjustment for spot markets
        if category not in ["linear", "inverse"]:
            logger.info(f"Skipping leverage adjustment for {pair}: {category} market does not support leverage")
            return True

        try:
            # Fetch current leverage
            current_leverage_response = self.client.get_leverage(pair)
            logger.debug(f"Raw leverage response for {pair}: {current_leverage_response}")

            # Handle case where no leverage data is returned
            if current_leverage_response is None:
                logger.warning(f"No leverage data available for {pair}. Using default leverage: {self.default_leverage}x")
                self.client.set_leverage(pair, self.default_leverage, category=category)
                return True

            # Handle string or numeric response
            if isinstance(current_leverage_response, str):
                current_leverage = int(current_leverage_response.replace('x', ''))
            else:
                current_leverage = int(current_leverage_response)

            logger.info(f"Leverage for {pair}: {current_leverage}x")

            # Check and adjust leverage if necessary
            if current_leverage > self.max_leverage:
                logger.info(f"Current leverage {current_leverage}x exceeds max {self.max_leverage}x, adjusting...")
                self.client.set_leverage(pair, self.max_leverage, category=category)
                logger.info(f"Leverage reduced to {self.max_leverage}x for {pair}")
            else:
                logger.debug(f"Leverage {current_leverage}x is within limit {self.max_leverage}x")

            return True
        except ValueError as ve:
            logger.error(f"Invalid leverage value for {pair}: {str(ve)}. Setting to default {self.default_leverage}x")
            try:
                self.client.set_leverage(pair, self.default_leverage, category=category)
                return True
            except Exception as set_e:
                logger.error(f"Failed to set default leverage: {str(set_e)}")
                return False
        except Exception as e:
            logger.error(f"Leverage adjustment failed for {pair}: {str(e)}")
            return False
