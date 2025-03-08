# data_pipeline/bybit_api.py
"""
Bybit API Interface Module

Provides methods to interact with Bybit's API for trading and data retrieval
"""

import time
import logging
import requests
from pybit.unified_trading import HTTP
from typing import List, Dict, Any

# Set up logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Running updated BybitAPI from data_pipeline/bybit_api.py")  # Debug marker

class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize BybitAPI with API credentials.

        Args:
            api_key (str): Bybit API key
            api_secret (str): Bybit API secret
            testnet (bool): Use testnet if True, mainnet if False (default: True)
        """
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        self.testnet = testnet
        logger.info("BybitAPI initialized")

    def get_btc_price(self) -> float | None:
        """Fetch the current BTC price."""
        try:
            response = self.client.get_tickers(category="linear", symbol="BTCUSDT")
            if response.get("retCode") != 0:
                logger.error(f"API error fetching BTC price: {response.get('retMsg')}")
                return None
            price = float(response["result"]["list"][0]["lastPrice"])
            logger.info(f"BTC Price: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching BTC price: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a given symbol using the Bybit API.

        Args:
            symbol (str): Trading symbol (e.g., "BTCUSDT")
            limit (int): Number of trades to fetch (default: 10)

        Returns:
            List[Dict[str, Any]]: List of trade dictionaries or empty list on failure
        """
        base_url = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        endpoint = "/v5/market/recent-trade"
        url = f"{base_url}{endpoint}"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") != 0:
                logger.error(f"API error fetching recent trades for {symbol}: {data.get('retMsg')}")
                return []
            trades = data.get("result", {}).get("list", [])
            logger.info(f"Fetched {len(trades)} recent trades for {symbol}")
            return trades
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching recent trades for {symbol}: {e}")
            return []
        except ValueError as e:
            logger.error(f"Error parsing JSON response for {symbol}: {e}")
            return []

    def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch open positions for a given symbol."""
        try:
            response = self.client.get_positions(category="linear", symbol=symbol)
            if response.get("retCode") != 0:
                logger.error(f"API error fetching positions for {symbol}: {response.get('retMsg')}")
                return []
            positions = response["result"]["list"]
            logger.info(f"Fetched {len(positions)} open positions for {symbol}")
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions for {symbol}: {e}")
            return []

    def close_position(self, symbol: str):
        """Close an open position for a given symbol."""
        positions = self.get_open_positions(symbol)
        position = next((p for p in positions if p["symbol"] == symbol and float(p["size"]) > 0), None)

        if position:
            try:
                close_order = self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side="Sell" if position["side"] == "Buy" else "Buy",
                    orderType="Market",
                    qty=str(position["size"]),
                    reduceOnly=True
                )
                logger.info(f"Close Position Response: {close_order}")
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
        else:
            logger.info(f"No open position to close for {symbol}")

    def place_order(self, symbol: str, side: str, qty: float):
        """Place a market order for a given symbol."""
        try:
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side.capitalize(),
                orderType="Market",
                qty=str(qty)
            )
            if response.get("retCode") != 0:
                logger.error(f"API error placing order for {symbol}: {response.get('retMsg')}")
                raise Exception(f"Order failed: {response.get('retMsg')}")
            logger.info(f"Order placed successfully: {response}")
            return response
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise

    def get_order_book(self, symbol: str) -> Dict[str, List] | None:
        """Fetch the order book for a given symbol."""
        try:
            response = self.client.get_orderbook(category="linear", symbol=symbol, limit=5)
            if response.get("retCode") != 0:
                logger.error(f"API error fetching order book for {symbol}: {response.get('retMsg')}")
                return None
            result = response['result']
            order_book = {
                'bids': [(str(bid[0]), str(bid[1])) for bid in result['b']],
                'asks': [(str(ask[0]), str(ask[1])) for ask in result['a']]
            }
            logger.info(f"Fetched order book for {symbol}")
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, interval: str = "60", limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch historical Kline data for a given symbol."""
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            if response.get("retCode") != 0:
                logger.error(f"API error fetching historical data for {symbol}: {response.get('retMsg')}")
                return []
            data = response["result"]["list"]
            logger.info(f"Fetched {len(data)} historical data points for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

def main():
    """Example usage of BybitAPI."""
    api_key = "05EqRWk80CvjiSto64"
    api_secret = "6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH"
    bybit_api = BybitAPI(api_key, api_secret, testnet=True)

    try:
        price = bybit_api.get_btc_price()
        if price:
            logger.info(f"BTC price: {price}")

        trades = bybit_api.get_recent_trades("BTCUSDT")
        logger.info(f"Recent Trades: {trades}")

        order_book = bybit_api.get_order_book("BTCUSDT")
        if order_book:
            logger.info(f"Order Book: {order_book}")

        historical_data = bybit_api.get_historical_data("BTCUSDT")
        logger.info(f"Historical Data (first 5): {historical_data[:5]}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

    time.sleep(10)

if __name__ == "__main__":
    main()
