from __future__ import annotations

class MarketInsights:
    """
    A class to provide market insights based on data fetched from the Bybit API.

    Attributes:
        client (BybitClient): An instance of BybitClient to interact with the API.
        symbols (list): List of trading symbols to analyze (e.g., ['BTCUSDT', 'ETHUSDT']).
        timeframe (str): The timeframe for candlestick data (e.g., '1m', '1h', '1d').
    """

    def __init__(self, client: BybitClient, symbols: list, timeframe: str = '1h'):
        """
        Initializes the MarketInsights with a BybitClient instance, list of symbols, and timeframe.

        Args:
            client (BybitClient): The Bybit API client instance used to fetch market data.
            symbols (list): List of symbols to analyze.
            timeframe (str, optional): The candlestick timeframe. Defaults to '1h'.
        """
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe

    def get_latest_data(self):
        """
        Fetches the latest candlestick data for the specified symbols and timeframe.

        Returns:
            dict: A dictionary with symbols as keys and lists of candlestick data as values.
                  Each candlestick is assumed to be a dict with keys like 'close', etc.

        Note:
            Assumes the BybitClient has a method `get_candlestick(symbol, timeframe, limit)`.
            Adjust the method call based on the actual BybitClient implementation.
        """
        data = {}
        for symbol in self.symbols:
            # Fetch candlestick data; limit=100 gets the last 100 periods
            candles = self.client.get_candlestick(symbol, self.timeframe, limit=100)
            data[symbol] = candles
        return data

    def analyze_market(self):
        """
        Analyzes the market data and provides basic insights.

        Returns:
            dict: A dictionary with symbols as keys and analysis results as values.
                  Each value is a dict containing metrics like 'average_close'.

        Note:
            Currently calculates the average closing price as a simple example.
            Expand this method with additional technical indicators as needed.
        """
        data = self.get_latest_data()
        insights = {}
        for symbol, candles in data.items():
            if candles and len(candles) > 0:
                # Extract closing prices and convert to float
                closes = [float(c['close']) for c in candles]
                average_close = sum(closes) / len(closes)
                insights[symbol] = {'average_close': average_close}
            else:
                # Handle case where no data is returned
                insights[symbol] = {'average_close': None}
        return insights
