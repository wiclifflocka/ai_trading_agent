# strategies/sell_strategy.py
from strategies.trading_strategy import TradingStrategy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SellStrategy(TradingStrategy):
    def __init__(self, client=None, symbol=None):
        super().__init__(client=client, symbol=symbol)
        logger.info(f"SellStrategy initialized for {self.symbol}")

    def execute_trade(self):
        logger.info(f"Executing Sell trade for {self.symbol}")
        if self.client:
            self.client.place_order(self.symbol, qty=0.001, side="Sell", order_type="Market")
