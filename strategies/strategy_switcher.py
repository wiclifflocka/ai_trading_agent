from strategies.trading_strategy import TradingStrategy

class StrategySwitcher:
    def __init__(self, strategy=None):
        """
        Initializes the StrategySwitcher with a default strategy.

        :param strategy: An instance of a trading strategy.
        """
        self.strategy = strategy if strategy else TradingStrategy()
        print("StrategySwitcher initialized with", self.strategy.__class__.__name__)

    def set_strategy(self, new_strategy):
        """
        Change the trading strategy dynamically.

        :param new_strategy: An instance of a trading strategy.
        """
        self.strategy = new_strategy
        print(f"Strategy switched to {self.strategy.__class__.__name__}")

    def execute(self):
        """
        Execute the current strategy's trade logic.
        """
        if self.strategy:
            self.strategy.execute_trade()
        else:
            print("No strategy set.")

# Example usage (optional, for testing)
if __name__ == "__main__":
    switcher = StrategySwitcher()
    switcher.execute()

