from strategies.trading_strategy import TradingStrategy
from strategies.buy_strategy import BuyStrategy  # Ensure these classes exist
from strategies.sell_strategy import SellStrategy
from strategies.hold_strategy import HoldStrategy

class StrategySwitcher:
    def __init__(self, client, strategy=None):
        """
        Initializes the StrategySwitcher with a BybitClient and an optional initial strategy.

        :param client: An instance of BybitClient for fetching market data.
        :param strategy: An instance of a trading strategy (default: TradingStrategy).
        """
        self.client = client
        self.strategy = strategy if strategy else TradingStrategy()
        print("StrategySwitcher initialized with", self.strategy.__class__.__name__)

    def set_strategy(self, new_strategy):
        """
        Change the trading strategy dynamically.

        :param new_strategy: An instance of a trading strategy.
        """
        self.strategy = new_strategy
        print(f"Strategy switched to {self.strategy.__class__.__name__}")

    def switch_strategy_based_on_market_conditions(self, symbol: str):
        """
        Switch trading strategy based on current market conditions.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT").

        This method fetches historical data, calculates a 20-period Simple Moving Average (SMA),
        and decides the strategy based on the current price relative to the SMA.
        """
        # Fetch historical data (e.g., last 50 candles with 15-minute interval)
        market_data = self.client.get_historical_data(symbol, interval="15", limit=50)
        if not market_data:
            print(f"No market data available for {symbol}, keeping current strategy")
            return

        # Extract closing prices (assuming close price is at index 4 in candlestick data)
        closes = [float(candle[4]) for candle in market_data]
        if len(closes) < 20:
            print(f"Insufficient data for {symbol}, keeping current strategy")
            return

        # Calculate 20-period SMA
        sma_20 = sum(closes[-20:]) / 20
        current_price = closes[-1]

        # Decide strategy based on current price vs. SMA
        if current_price > sma_20:
            new_strategy = BuyStrategy()
        elif current_price < sma_20:
            new_strategy = SellStrategy()
        else:
            new_strategy = HoldStrategy()

        # Set the new strategy
        self.set_strategy(new_strategy)

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
    from bybit_client import BybitClient  # Assuming BybitClient is defined elsewhere
    api_key = "YOUR_API_KEY"  # Replace with your actual API key
    api_secret = "YOUR_API_SECRET"  # Replace with your actual API secret
    client = BybitClient(api_key, api_secret, testnet=True)
    switcher = StrategySwitcher(client)
    switcher.switch_strategy_based_on_market_conditions("BTCUSDT")
    switcher.execute()
