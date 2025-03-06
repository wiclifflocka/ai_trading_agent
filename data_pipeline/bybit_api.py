import requests
import time
import json
from bybit import bybit

# Configuration for API keys (store securely in environment variables for production)
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

# Bybit API Client
client = bybit(test=True, api_key=API_KEY, api_secret=API_SECRET)  # Use testnet=True for sandbox testing

class BybitAPI:
    """
    Handles API requests to fetch market data from Bybit.
    """

    def __init__(self):
        self.client = client

    def get_order_book(self, symbol="BTCUSDT", depth=50):
        """
        Fetch the order book for a given trading pair.
        :param symbol: Trading pair (e.g., "BTCUSDT")
        :param depth: Number of order levels to fetch
        :return: Order book data (bids and asks)
        """
        try:
            response = self.client.Market.Market_orderbook(symbol=symbol).result()
            return response[0]['result']
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None

    def get_recent_trades(self, symbol="BTCUSDT"):
        """
        Fetch recent market trades for a given trading pair.
        :param symbol: Trading pair (e.g., "BTCUSDT")
        :return: List of recent trades
        """
        try:
            response = self.client.Market.Market_tradingRecords(symbol=symbol).result()
            return response[0]['result']
        except Exception as e:
            print(f"Error fetching recent trades: {e}")
            return None

# Example usage
if __name__ == "__main__":
    api = BybitAPI()
    order_book = api.get_order_book()
    print(json.dumps(order_book, indent=4))  # Pretty-print the order book data

