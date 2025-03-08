# bybit_client.py
"""
Bybit API Client for Spot and Derivatives Trading

A comprehensive interface for interacting with Bybit's API, supporting trading operations
and data retrieval. Configured for testnet by default.
"""

import logging
import time
from pybit.unified_trading import HTTP

# Set up logging with a detailed format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BybitClient:
    """
    Features:
    - Account balance checks (USDT equity)
    - Order placement (market and limit)
    - Market price retrieval
    - Historical kline/candlestick data with retry mechanism
    - Position management (view and close)
    - Leverage control
    - Order book data
    - Recent trade records
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialize the Bybit API client with credentials."""
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        logger.info("BybitClient initialized")

    def get_balance(self) -> float:
        """
        Get total equity balance for USDT in the unified account.

        Returns:
            float: Total equity in USDT, or 10000.0 if retrieval fails
        """
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            logger.info(f"API Response from get_wallet_balance: {response}")
            balance_list = response.get('result', {}).get('list', [])
            if not balance_list or 'coin' not in balance_list[0]:
                logger.warning("No balance data found, returning fallback value 10000.0")
                return 10000.0
            for coin_data in balance_list[0]['coin']:
                if coin_data['coin'] == "USDT":
                    balance = float(coin_data['equity'])
                    logger.info(f"Fetched USDT Equity Balance: {balance} USD")
                    return balance
            logger.warning("No USDT balance found, returning fallback value 10000.0")
            return 10000.0
        except Exception as e:
            logger.warning(f"Balance check failed: {str(e)}, returning fallback value 10000.0")
            return 10000.0

    def place_order(self, symbol: str, qty: float, side: str, order_type: str = "Market") -> dict:
        """
        Place a new order (market or limit).

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            qty (float): Order quantity
            side (str): "Buy" or "Sell"
            order_type (str): "Market" or "Limit" (default: "Market")

        Returns:
            dict: Order response from Bybit
        """
        try:
            order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side.capitalize(),
                orderType=order_type,
                qty=str(qty)
            )
            logger.info(f"Order placed successfully: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise

    def get_market_price(self, symbol: str) -> float:
        """
        Get the latest market price for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            float: Last traded price
        """
        try:
            response = self.client.get_tickers(category="linear", symbol=symbol)
            price = float(response['result']['list'][0]['lastPrice'])
            logger.info(f"Market price for {symbol}: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching market price for {symbol}: {e}")
            raise

    def get_historical_data(self, symbol: str, interval: str = "15", limit: int = 200) -> list:
        """
        Get historical kline/candlestick data with retry mechanism.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            interval (str): Candle interval in minutes (e.g., "1", "5", "15", "60")
            limit (int): Number of candles to retrieve (max 1000, default: 200)

        Returns:
            list: Array of [timestamp, open, high, low, close, volume] as floats
        """
        for attempt in range(3):
            try:
                response = self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                data = [
                    [float(item[0]), float(item[1]), float(item[2]),
                     float(item[3]), float(item[4]), float(item[5])]
                    for item in response['result']['list']
                ]
                logger.info(f"Fetched {len(data)} historical data points for {symbol}")
                return data
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to fetch historical data for {symbol}: {e}")
                if attempt < 2:
                    time.sleep(2)  # Wait before retrying
                else:
                    logger.warning(f"All attempts failed for {symbol}, returning empty list")
                    return []
        return []

    def get_candlestick(self, symbol: str, interval: str = "15", limit: int = 200) -> list:
        """
        Get historical candlestick data for a symbol (alias for get_historical_data).

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            interval (str): Candle interval in minutes (e.g., "1", "5", "15", "60")
            limit (int): Number of candles to retrieve (max 1000, default: 200)

        Returns:
            list: Array of [timestamp, open, high, low, close, volume] as floats
        """
        return self.get_historical_data(symbol, interval, limit)

    def close_position(self, symbol: str) -> dict | None:
        """
        Close the open position for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            dict | None: API response if position closed, None if no position
        """
        try:
            positions = self.get_positions(symbol)
            if not positions:
                logger.info(f"No open positions to close for {symbol}")
                return None
            position = next((p for p in positions if float(p["size"]) > 0), None)
            if position:
                close_order = self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side="Sell" if position["side"] == "Buy" else "Buy",
                    orderType="Market",
                    qty=str(position["size"]),
                    reduceOnly=True
                )
                logger.info(f"Position closed for {symbol}: {close_order}")
                return close_order
            logger.info(f"No non-zero size positions to close for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            raise

    def get_positions(self, symbol: str) -> list:
        """
        Get current open positions for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            list: List of position details
        """
        try:
            response = self.client.get_positions(category="linear", symbol=symbol)
            positions = response['result']['list']
            logger.info(f"Fetched {len(positions)} positions for {symbol}")
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions for {symbol}: {e}")
            raise

    def set_leverage(self, symbol: str, leverage: int = 10):
        """
        Set leverage for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            leverage (int): Desired leverage (default: 10)
        """
        try:
            self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            raise

    def get_leverage(self, symbol: str) -> float | None:
        """
        Fetch the current leverage for a specific trading symbol.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT")

        Returns:
            float | None: The leverage as a float if successful, otherwise None
        """
        try:
            response = self.client.get_positions(category="linear", symbol=symbol)
            if response['retCode'] == 0:
                positions = response.get('result', {}).get('list', [])
                if positions:
                    leverage = positions[0].get('leverage')
                    if leverage is not None:
                        logger.info(f"Leverage for {symbol}: {leverage}x")
                        return float(leverage)
                    logger.warning(f"No leverage data found for symbol {symbol}")
                    return None
                logger.info(f"No positions found for symbol {symbol}")
                return None
            logger.error(f"API error for symbol {symbol}: {response.get('retMsg', 'Unknown error')}")
            return None
        except Exception as e:
            logger.error(f"Exception while fetching leverage for {symbol}: {e}")
            return None

    def get_order_book(self, symbol: str) -> dict:
        """
        Get order book data for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            dict: Order book with 'bids' and 'asks' lists (price, quantity as strings)
        """
        try:
            response = self.client.get_orderbook(category="linear", symbol=symbol, limit=5)
            result = response['result']
            order_book = {
                'bids': [(str(bid[0]), str(bid[1])) for bid in result['b']],
                'asks': [(str(ask[0]), str(ask[1])) for ask in result['a']]
            }
            logger.info(f"Fetched order book for {symbol}")
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise

    def get_recent_trades(self, symbol: str, limit: int = 10) -> list:
        """
        Fetch recent trading records for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            limit (int): Number of trades to fetch (default: 10)

        Returns:
            list: List of recent trades, empty list if retrieval fails
        """
        try:
            response = self.client.get_public_trade(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            trades = response.get("result", {}).get("list", [])
            logger.info(f"Fetched {len(trades)} recent trades for {symbol}")
            return trades
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return []

# Example Usage
if __name__ == "__main__":
    api_key = "05EqRWk80CvjiSto64"
    api_secret = "6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH"
    client = BybitClient(api_key, api_secret, testnet=True)

    balance = client.get_balance()
    print(f"Unified USDT Balance: {balance} USD")

    price = client.get_market_price("BTCUSDT")
    print(f"BTCUSDT Market Price: {price}")

    order_book = client.get_order_book("BTCUSDT")
    print(f"Order Book (first bid/ask): {order_book['bids'][0]}, {order_book['asks'][0]}")

    historical_data = client.get_historical_data("BTCUSDT")
    print(f"Historical Data (first entry): {historical_data[0]}")
