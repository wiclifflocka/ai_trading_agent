# risk_management/stop_loss_take_profit.py
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

class StopLossTakeProfit:
    def __init__(self, client: BybitClient):
        self.client = client

    def set_levels(self, symbol: str, entry_price: float, stop_loss_factor: float, take_profit_factor: float):
        """
        Set stop-loss and take-profit levels for a position.
        
        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            entry_price (float): Entry price of the trade
            stop_loss_factor (float): Factor below entry price for stop-loss (e.g., 0.02 for 2%)
            take_profit_factor (float): Factor above entry price for take-profit (e.g., 0.04 for 4%)
        """
        try:
            stop_loss_price = entry_price * (1 - stop_loss_factor)
            take_profit_price = entry_price * (1 + take_profit_factor)
            
            # Bybit requires conditional orders for SL/TP; simulate with market orders for now
            logger.info(f"Setting stop-loss at {stop_loss_price} and take-profit at {take_profit_price} for {symbol}")
            # Note: Actual SL/TP orders require Bybit's conditional order API, not fully supported here
            # Placeholder: Log intent; implement with client.place_order if conditional orders are added
        except Exception as e:
            logger.error(f"Failed to set stop-loss/take-profit levels for {symbol}: {str(e)}")

    def place_stop_loss(self, pair: str, entry_price: float, stop_loss_percentage: float):
        """
        Places a stop-loss order based on a percentage of the entry price.
        
        Args:
            pair (str): The trading pair (e.g., "BTCUSDT")
            entry_price (float): The entry price of the trade
            stop_loss_percentage (float): Percentage of entry price for stop-loss
        """
        try:
            stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
            # Corrected to match BybitClient.place_order parameters
            self.client.place_order(pair, stop_loss_price, "Sell", "Market")
            logger.info(f"Stop-loss placed at {stop_loss_price} for {pair}")
        except Exception as e:
            logger.error(f"Failed to place stop-loss for {pair}: {str(e)}")

    def place_take_profit(self, pair: str, entry_price: float, take_profit_percentage: float):
        """
        Places a take-profit order based on a percentage of the entry price.
        
        Args:
            pair (str): The trading pair (e.g., "BTCUSDT")
            entry_price (float): The entry price of the trade
            take_profit_percentage (float): Percentage of entry price for take-profit
        """
        try:
            take_profit_price = entry_price * (1 + take_profit_percentage / 100)
            # Corrected to match BybitClient.place_order parameters
            self.client.place_order(pair, take_profit_price, "Sell", "Market")
            logger.info(f"Take-profit placed at {take_profit_price} for {pair}")
        except Exception as e:
            logger.error(f"Failed to place take-profit for {pair}: {str(e)}")
