# risk_management/trailing_stop.py

class TrailingStopLoss:
    def __init__(self, client: BybitClient):
        self.client = client

    def place_trailing_stop(self, pair, entry_price, trail_percentage):
        """
        Places a trailing stop loss order.
        :param pair: The trading pair (e.g., 'BTCUSD')
        :param entry_price: The entry price of the trade
        :param trail_percentage: Percentage for trailing stop
        """
        trailing_stop_price = entry_price * (1 - trail_percentage / 100)
        # Simulating placing a trailing stop-loss order
        self.client.place_order(pair, "SELL", trailing_stop_price, "TRAILING_STOP")
        logger.info(f"Trailing stop-loss placed at {trailing_stop_price} for {pair}")

