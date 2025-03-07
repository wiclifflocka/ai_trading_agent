# data_pipeline/bybit_api.py
"""
Bybit API Interface Module

Provides methods to interact with Bybit's API for trading and data retrieval
"""

import time
import logging
from pybit.unified_trading import HTTP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def get_btc_price(self) -> float | None:
        """
        Fetch the latest BTC price.

        Returns:
            float | None: Last price of BTCUSDT if successful, None otherwise
        """
        try:
            response = self.client.get_tickers(category="linear", symbol="BTCUSDT")
            price = float(response["result"]["list"][0]["lastPrice"])
            logger.info(f"BTC Price: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching BTC price: {e}")
            return None

    def get_recent_trades(self, symbol: str) -> list:
        """
        Fetch recent trading records for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            list: List of recent trades if successful, empty list otherwise
        """
        try:
            response = self.client.get_recent_trades(category="linear", symbol=symbol, limit=10)
            return response["result"]["list"]
        except AttributeError:
            # Fallback for older pybit versions or method name changes
            try:
                response = self.client.get_public_trade_records(category="linear", symbol=symbol, limit=10)
                return response["result"]["list"]
            except Exception as e:
                logger.error(f"Error fetching recent trades for {symbol}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return []

    def get_open_positions(self, symbol: str) -> list:
        """
        Fetch open positions for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            list: List of position details if successful, empty list otherwise
        """
        try:
            response = self.client.get_positions(category="linear", symbol=symbol)
            return response["result"]["list"]
        except Exception as e:
            logger.error(f"Error fetching positions for {symbol}: {e}")
            return []

    def close_position(self, symbol: str):
        """
        Close an open position for a given symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
        """
        positions = self.get_open_positions(symbol)
        position = next((p for p in positions if p["symbol"] == symbol and float(p["size"]) > 0), None)

        if position:
            try:
                close_order = self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side="Sell" if position["side"] == "Buy" else "Buy",
                    order_type="Market",
                    qty=str(position["size"]),
                    reduce_only=True
                )
                logger.info(f"Close Position Response: {close_order}")
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
        else:
            logger.info(f"No open position to close for {symbol}")

    def place_order(self, symbol: str, side: str, qty: float):
        """
        Place a market order.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            side (str): "Buy" or "Sell"
            qty (float): Order quantity
        """
        try:
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side.capitalize(),
                order_type="Market",
                qty=str(qty)
            )
            logger.info(f"Order placed successfully: {response}")
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")

    def get_order_book(self, symbol: str) -> dict | None:
        """
        Fetch the order book for a given symbol from Bybit.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            dict | None: Order book data with 'bids' and 'asks' if successful, None otherwise
        """
        try:
            response = self.client.get_orderbook(category="linear", symbol=symbol, limit=5)
            if response and 'result' in response:
                result = response['result']
                return {
                    'bids': [(str(bid[0]), str(bid[1])) for bid in result['b']],
                    'asks': [(str(ask[0]), str(ask[1])) for ask in result['a']]
                }
            logger.error(f"Failed to fetch order book data for {symbol}: Invalid response")
            return None
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, interval: str = "60", limit: int = 200) -> list:
        """
        Fetch historical candlestick data for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            interval (str): Candlestick interval (e.g., "1", "5", "15", "60")
            limit (int): Number of candles to fetch (max 200)

        Returns:
            list: List of OHLCV data if successful, empty list otherwise
        """
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return response["result"]["list"]
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

def main():
    # Example usage
    api_key = "your_api_key_here"  # Replace with actual key
    api_secret = "your_api_secret_here"  # Replace with actual secret
    bybit_api = BybitAPI(api_key, api_secret, testnet=True)

    # Test BTC price
    price = bybit_api.get_btc_price()
    if price:
        logger.info(f"BTC price: {price}")

    # Test recent trades
    trades = bybit_api.get_recent_trades("BTCUSDT")
    logger.info(f"Recent Trades: {trades}")

    # Test order book
    order_book = bybit_api.get_order_book("BTCUSDT")
    if order_book:
        logger.info(f"Order Book: {order_book}")

    # Test historical data
    historical_data = bybit_api.get_historical_data("BTCUSDT")
    logger.info(f"Historical Data (first 5): {historical_data[:5]}")

    # Wait before next cycle
    time.sleep(10)

if __name__ == "__main__":
    main()
