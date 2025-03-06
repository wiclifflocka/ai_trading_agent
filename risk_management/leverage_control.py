# risk_management/leverage_control.py

class LeverageControl:
    def __init__(self, client: BybitClient):
        self.client = client
        self.max_leverage = 10  # Maximum leverage allowed

    def check_and_set_leverage(self, pair):
        """
        Checks the leverage for the current position and adjusts it to the safe limit.
        :param pair: The trading pair (e.g., 'BTCUSD')
        """
        current_leverage = self.client.get_leverage(pair)
        if current_leverage > self.max_leverage:
            self.client.set_leverage(pair, self.max_leverage)
            logger.info(f"Leverage reduced to {self.max_leverage} for {pair}")

