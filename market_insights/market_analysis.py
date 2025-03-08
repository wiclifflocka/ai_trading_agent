# market_insights/market_analysis.py
"""
Market Insights Module

Analyzes market data to provide trading insights using candlestick data from Bybit API
"""
from __future__ import annotations
import logging
from typing import Dict, List

# Assuming BybitClient is imported from your bybit_client module
from bybit_client import BybitClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketInsights:
    """
    A class to provide market insights based on data fetched from the Bybit API.

    Attributes:
        client (BybitClient): An instance of BybitClient to interact with the API.
        symbols (list): List of trading symbols to analyze (e.g., ['BTCUSDT', 'ETHUSDT']).
        timeframe (str): The timeframe for candlestick data (e.g., '1m', '1h', '1d').
    """

    def __init__(self, client: BybitClient, symbols: List[str], timeframe: str = '1h'):
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

    def get_latest_data(self) -> Dict[str, List[Dict]]:
        """
        Fetches the latest candlestick data for the specified symbols and timeframe.

        Returns:
            dict: A dictionary with symbols as keys and lists of candlestick data as values.
                  Each candlestick is a dict with keys like 'open', 'high', 'low', 'close', etc.
        """
        data = {}
        for symbol in self.symbols:
            try:
                # Fetch candlestick data; limit=10 gets the last 10 periods
                candles = self.client.get_candlestick(symbol, self.timeframe, limit=10)
                if candles and isinstance(candles, list):
                    data[symbol] = candles
                    logger.info(f"Fetched {len(candles)} candles for {symbol}")
                else:
                    logger.warning(f"No candlestick data returned for {symbol}")
                    data[symbol] = []
            except Exception as e:
                logger.error(f"Error fetching candlestick data for {symbol}: {e}")
                data[symbol] = []
        return data

    def analyze_market(self) -> Dict[str, Dict[str, float]]:
        """
        Analyzes the market data and provides trading insights.

        Returns:
            dict: A dictionary with symbols as keys and analysis results as values.
                  Each value is a dict containing:
                  - 'entry_price': Suggested entry price (latest close).
                  - 'stop_loss_percentage': Suggested stop-loss percentage.
                  - 'take_profit_percentage': Suggested take-profit percentage.
                  - 'average_close': Average closing price over the period.
        """
        data = self.get_latest_data()
        insights = {}

        for symbol, candles in data.items():
            if candles and len(candles) > 0:
                try:
                    # Extract prices and convert to float
                    closes = [float(c['close']) for c in candles]
                    highs = [float(c['high']) for c in candles]
                    lows = [float(c['low']) for c in candles]

                    # Basic analysis
                    average_close = sum(closes) / len(closes)
                    latest_close = closes[0]  # Most recent candle
                    avg_range = (sum(highs) / len(highs)) - (sum(lows) / len(lows))  # Average true range

                    # Simple strategy: Buy at latest close
                    entry_price = latest_close

                    # Stop-loss: 1% below entry (adjustable)
                    stop_loss_percentage = 1.0  # 1% loss
                    stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)

                    # Take-profit: 2% above entry (adjustable, 2:1 risk-reward ratio)
                    take_profit_percentage = 2.0  # 2% gain
                    take_profit_price = entry_price * (1 + take_profit_percentage / 100)

                    insights[symbol] = {
                        'entry_price': entry_price,
                        'stop_loss_percentage': stop_loss_percentage,
                        'take_profit_percentage': take_profit_percentage,
                        'average_close': average_close
                    }
                    logger.info(f"Analysis for {symbol}: Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}")
                except (KeyError, ValueError) as e:
                    logger.error(f"Error analyzing data for {symbol}: {e}")
                    insights[symbol] = {'average_close': None}
            else:
                insights[symbol] = {
                    'entry_price': None,
                    'stop_loss_percentage': None,
                    'take_profit_percentage': None,
                    'average_close': None
                }
                logger.warning(f"No valid data to analyze for {symbol}")
        return insights

if __name__ == "__main__":
    # Example usage (assuming BybitClient is available)
    from bybit_client import BybitClient
    client = BybitClient("your_api_key", "your_api_secret", testnet=True)
    symbols = ["BTCUSDT", "ETHUSDT"]
    market_insights = MarketInsights(client, symbols, timeframe="1h")
    insights = market_insights.analyze_market()
    print(insights)
