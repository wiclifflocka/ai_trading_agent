# risk_management/risk_manager.py
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, client, symbol: str = "BTCUSDT", max_loss: float = 0.02, volatility_threshold: float = 0.5):
        """
        Initialize the Risk Manager.

        Args:
            client: BybitClient instance for API access
            symbol (str): Trading pair
            max_loss (float): Maximum percentage loss before exiting a trade
            volatility_threshold (float): Adjust spread based on volatility
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
            # Try fetching recent trades first
            price_data = self.client.get_recent_trades(self.symbol, limit=100)
            if not price_data or len(price_data) < 2:
                logger.warning(f"Insufficient trade data for {self.symbol} volatility calculation. Falling back to OHLCV.")
                # Fallback to OHLCV data
                ohlcv_data = self.client.get_historical_data(self.symbol, interval='1m', limit=100)
                if not ohlcv_data or len(ohlcv_data) < 2:
                    logger.warning(f"Insufficient OHLCV data for {self.symbol} volatility calculation.")
                    self.current_volatility = 0.0
                    return False
                prices = [float(candle[4]) for candle in ohlcv_data]  # Use closing prices
            else:
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
            current_price = self.client.get_current_price(self.symbol)
            if current_price == 0.0:
                logger.error(f"Failed to fetch current price for {self.symbol}. Skipping stop-loss check.")
                return

            if position_type.lower() == "long" and (current_price < entry_price * (1 - self.max_loss)):
                logger.info("🚨 Stop-loss triggered! Closing long position.")
                self._close_position()
            elif position_type.lower() == "short" and (current_price > entry_price * (1 + self.max_loss)):
                logger.info("🚨 Stop-loss triggered! Closing short position.")
                self._close_position()
        except Exception as e:
            logger.error(f"Stop-loss check failed for {self.symbol}: {str(e)}")

    def _close_position(self):
        """Close the current position on the exchange."""
        try:
            positions = self.client.get_positions(self.symbol)
            if not positions or float(positions[0]['contracts']) == 0:
                logger.info(f"No open position to close for {self.symbol}")
                return

            position = positions[0]
            side_to_close = 'Sell' if position['side'].lower() == 'buy' else 'Buy'
            qty = float(position['contracts'])
            self.client.place_order(
                symbol=self.symbol,
                side=side_to_close,
                qty=qty,
                order_type="Market",
                reduce_only=True
            )
            logger.info(f"Closed {position['side']} position of {qty} for {self.symbol}")
        except Exception as e:
            logger.error(f"Failed to close position for {self.symbol}: {str(e)}")

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
