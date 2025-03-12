from strategies.trading_strategy import TradingStrategy
from strategies.buy_strategy import BuyStrategy
from strategies.sell_strategy import SellStrategy
from strategies.hold_strategy import HoldStrategy

class StrategySwitcher:
    def __init__(self, client, symbol, strategy=None):
        """
        Initializes the StrategySwitcher with a BybitClient, symbol, and an optional initial strategy.

        :param client: An instance of BybitClient for fetching market data and placing orders.
        :param symbol: The trading symbol (e.g., "BTCUSDT").
        :param strategy: An instance of a trading strategy (default: TradingStrategy).
        """
        self.client = client
        self.symbol = symbol
        self.strategy = strategy if strategy else TradingStrategy(client=client, symbol=symbol)
        print("StrategySwitcher initialized with", self.strategy.__class__.__name__)

    def set_strategy(self, new_strategy):
        """
        Change the trading strategy dynamically.

        :param new_strategy: An instance of a trading strategy.
        """
        self.strategy = new_strategy
        print(f"Strategy switched to {self.strategy.__class__.__name__}")

    def switch_strategy_based_on_market_conditions(self):
        """
        Switch trading strategy based on current market conditions.

        This method fetches historical data, calculates a 20-period Simple Moving Average (SMA),
        and decides the strategy based on the current price relative to the SMA.
        """
        # Fetch historical data (e.g., last 50 candles with 15-minute interval)
        market_data = self.client.get_historical_data(self.symbol, interval="15", limit=50)
        if not market_data:
            print(f"No market data available for {self.symbol}, keeping current strategy")
            return

        # Extract closing prices (assuming close price is at index 4 in candlestick data)
        closes = [float(candle[4]) for candle in market_data]
        if len(closes) < 20:
            print(f"Insufficient data for {self.symbol}, keeping current strategy")
            return

        # Calculate 20-period SMA
        sma_20 = sum(closes[-20:]) / 20
        current_price = closes[-1]

        # Decide strategy based on current price vs. SMA
        if current_price > sma_20:
            new_strategy = BuyStrategy(client=self.client, symbol=self.symbol)
        elif current_price < sma_20:
            new_strategy = SellStrategy(client=self.client, symbol=self.symbol)
        else:
            new_strategy = HoldStrategy(client=self.client, symbol=self.symbol)

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
