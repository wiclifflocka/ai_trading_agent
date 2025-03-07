"""
Bybit API Client for Spot and Derivatives Trading

Handles authenticated requests to Bybit's API using the official pybit library.
Configured for testnet/demo accounts by default.

Features:
- Account balance checks
- Order placement (limit/market)
- Market price data
- Historical kline data
- Position management
- Leverage control
- Order book retrieval

Requirements:
- pybit library (install with `pip install pybit`)
- Valid Bybit testnet API keys (from https://testnet.bybit.com)
"""

import logging
from pybit.unified_trading import HTTP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BybitClient:
    """
    Bybit API client for trading operations

    Args:
        api_key (str): Bybit API key
        api_secret (str): Bybit API secret
        testnet (bool): True for demo account, False for live trading
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )

    def get_balance(self) -> float:
        """
        Get total equity balance for USDT in the unified account.

        Returns:
            float: Total equity in USDT, or 10000.0 if retrieval fails

        Notes:
            - Returns a fallback value to prevent invalid drawdown calculations
        """
        try:
            response = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
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

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "Market"
    ) -> dict:
        """
        Place a new order (market order by default)

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            qty (float): Order quantity
            side (str): 'Buy' or 'Sell'
            order_type (str): 'Market' or 'Limit'

        Returns:
            dict: Order response from Bybit

        Raises:
            Exception: On placement failure
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
            logger.error(f"Order failed: {str(e)}")
            raise

    def get_market_price(self, symbol: str) -> float:
        """
        Get latest market price for a symbol

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')

        Returns:
            float: Last traded price
        """
        try:
            response = self.client.get_tickers(
                category="linear",
                symbol=symbol
            )
            return float(response['result']['list'][0]['lastPrice'])
        except Exception as e:
            logger.error(f"Price check failed: {str(e)}")
            raise

    def get_historical_data(
        self,
        symbol: str,
        interval: int = 15,
        limit: int = 200
    ) -> list:
        """
        Get historical kline/candlestick data

        Args:
            symbol (str): Trading pair
            interval (int): Minutes per candle (1, 3, 5, 15, 30, 60, 120, 240, 360, 720)
            limit (int): Number of candles to retrieve (max 1000)

        Returns:
            list: Array of kline data in [timestamp, open, high, low, close, volume] format
        """
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=str(interval),
                limit=limit
            )
            return [
                [float(item[0]), float(item[1]), float(item[2]),
                 float(item[3]), float(item[4]), float(item[5])]
                for item in response['result']['list']
            ]
        except Exception as e:
            logger.error(f"Historical data failed: {str(e)}")
            raise

    def close_all_positions(self, symbol: str) -> dict | None:
        """
        Close all open positions for a symbol

        Args:
            symbol (str): Trading pair to close

        Returns:
            dict | None: API response if positions closed, None if no positions
        """
        try:
            positions = self.get_positions(symbol)
            if not positions:
                logger.info(f"No open positions to close for {symbol}")
                return None
            position = positions[0]
            if float(position["size"]) > 0:
                return self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side="Sell" if position["side"] == "Buy" else "Buy",
                    orderType="Market",
                    qty=str(position["size"]),
                    reduceOnly=True
                )
            logger.info(f"No non-zero size positions to close for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Position closure failed: {str(e)}")
            raise

    def get_positions(self, symbol: str) -> list:
        """
        Get current open positions for a symbol

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')

        Returns:
            list: List of position details
        """
        try:
            response = self.client.get_positions(
                category="linear",
                symbol=symbol
            )
            return response['result']['list']
        except Exception as e:
            logger.error(f"Position fetch failed: {str(e)}")
            raise

    def check_and_set_leverage(self, symbol: str, leverage: int = 10):
        """
        Check and set leverage for a symbol

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
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
            logger.error(f"Leverage setting failed: {str(e)}")
            raise

    def get_order_book(self, symbol: str) -> dict:
        """
        Get order book data for a symbol

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')

        Returns:
            dict: Order book with 'bids' and 'asks' lists
        """
        try:
            response = self.client.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=5
            )
            result = response['result']
            return {
                'bids': [(str(bid[0]), str(bid[1])) for bid in result['b']],
                'asks': [(str(ask[0]), str(ask[1])) for ask in result['a']]
            }
        except Exception as e:
            logger.error(f"Order book fetch failed: {str(e)}")
            raise

# Example Usage
if __name__ == "__main__":
    client = BybitClient("YOUR_TESTNET_API_KEY", "YOUR_TESTNET_API_SECRET", testnet=True)
    balance = client.get_balance()
    print(f"Unified USDT Balance: {balance} USD")
