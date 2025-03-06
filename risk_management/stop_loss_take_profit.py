# risk_management/stop_loss_take_profit.py

class StopLossTakeProfit:
    def __init__(self, client: BybitClient):
        self.client = client

    def place_stop_loss(self, pair, entry_price, stop_loss_percentage):
        """
        Places a stop-loss order based on a percentage of the entry price.
        :param pair: The trading pair (e.g., 'BTCUSD')
        :param entry_price: The entry price of the trade
        :param stop_loss_percentage: Percentage of entry price for stop-loss
        """
        stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
        # Place a stop-loss order (simulating for now)
        self.client.place_order(pair, "SELL", stop_loss_price, "LIMIT")
        logger.info(f"Stop-loss placed at {stop_loss_price} for {pair}")

    def place_take_profit(self, pair, entry_price, take_profit_percentage):
        """
        Places a take-profit order based on a percentage of the entry price.
        :param pair: The trading pair (e.g., 'BTCUSD')
        :param entry_price: The entry price of the trade
        :param take_profit_percentage: Percentage of entry price for take-profit
        """
        take_profit_price = entry_price * (1 + take_profit_percentage / 100)
        # Place a take-profit order (simulating for now)
        self.client.place_order(pair, "SELL", take_profit_price, "LIMIT")
        logger.info(f"Take-profit placed at {take_profit_price} for {pair}")

