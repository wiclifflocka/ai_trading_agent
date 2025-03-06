# risk_management/max_loss.py

class MaxLossPerTrade:
    def __init__(self, client: BybitClient, account_balance):
        self.client = client
        self.account_balance = account_balance
        self.max_loss_percentage = 2  # Maximum loss per trade in %

    def calculate_max_loss(self):
        """
        Calculates the maximum loss allowed per trade.
        :return: Maximum loss value in USD (or base currency)
        """
        max_loss_value = self.account_balance * (self.max_loss_percentage / 100)
        return max_loss_value

