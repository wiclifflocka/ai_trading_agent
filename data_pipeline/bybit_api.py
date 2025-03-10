import logging
from typing import Dict, Any
from pybit.unified_trading import HTTP
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize BybitAPI with API credentials.

        Args:
            api_key (str): Bybit API key from environment variables.
            api_secret (str): Bybit API secret from environment variables.
            testnet (bool): Use testnet if True, mainnet if False.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        try:
            self.client = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret
            )
            logger.info("BybitAPI initialized with testnet=%s", testnet)
        except Exception as e:
            logger.error("Failed to initialize BybitAPI: %s", e)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_historical_klines(self, symbol: str, interval: str = "1", limit: int = 500) -> Dict[str, Any]:
        """
        Fetch historical Kline (OHLCV) data from Bybit for futures trading with retry logic.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").
            interval (str): Time interval (e.g., "1", "5", "60", "D").
            limit (int): Number of data points to fetch (default 500).

        Returns:
            Dict: API response with OHLCV data, or empty dict on failure.
        """
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            logger.debug("Klines response: %s", response)
            if response.get('retCode') == 0:
                logger.info("Fetched historical Kline data for %s", symbol)
                return response
            else:
                logger.error("Failed to fetch Kline data: %s (retCode: %s)",
                             response.get('retMsg'), response.get('retCode'))
                return {}
        except Exception as e:
            logger.error("Error fetching historical Kline data: %s", e)
            return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def place_order(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        """
        Place a market order on Bybit with retry logic.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").
            side (str): "Buy" or "Sell".
            qty (float): Quantity to trade in base currency (e.g., BTC).

        Returns:
            Dict: Success response with order ID, or error message.
        """
        try:
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=str(qty)
            )
            logger.debug("Order response: %s", response)
            if response.get('retCode') == 0:
                logger.info("Order placed: %s - %s - qty=%s", symbol, side, qty)
                return {"status": "success", "order_id": response['result']['order_id']}
            else:
                logger.error("Failed to place order: %s (retCode: %s)",
                             response.get('retMsg'), response.get('retCode'))
                return {"status": "error", "message": response.get('retMsg')}
        except Exception as e:
            logger.error("Error placing order: %s", e)
            return {"status": "error", "message": str(e)}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a symbol with retry logic.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").
            leverage (int): Desired leverage (e.g., 10).

        Returns:
            Dict: Success or error response.
        """
        try:
            response = self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buy_leverage=str(leverage),
                sell_leverage=str(leverage)
            )
            logger.debug("Leverage response: %s", response)
            if response.get('retCode') == 0:
                logger.info("Leverage set to %d for %s", leverage, symbol)
                return {"status": "success"}
            elif response.get('retCode') == 110043:  # Leverage already set
                logger.info("Leverage already set to %d for %s", leverage, symbol)
                return {"status": "success", "message": "leverage not modified"}
            else:
                logger.error("Failed to set leverage: %s (retCode: %s)",
                             response.get('retMsg'), response.get('retCode'))
                return {"status": "error", "message": response.get('retMsg')}
        except Exception as e:
            logger.error("Error setting leverage: %s", e)
            return {"status": "error", "message": str(e)}

    def get_balance(self, account_type: str = "UNIFIED", coin: str = "USDT") -> Dict[str, Any]:
        """
        Fetch wallet balance for a specific account type and coin.

        Args:
            account_type (str): Account type (e.g., "UNIFIED").
            coin (str): Coin to fetch balance for (e.g., "USDT").

        Returns:
            Dict: Balance info, or None on error.
        """
        try:
            response = self.client.get_wallet_balance(accountType=account_type)
            logger.debug("Raw balance response: %s", response)
            if response.get('retCode') == 0:
                logger.info("Fetched balance successfully")
                return response
            else:
                logger.error("Failed to fetch balance: %s (retCode: %s)",
                             response.get('retMsg'), response.get('retCode'))
                return None
        except Exception as e:
            logger.error("Exception fetching balance: %s", e)
            return None

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current position for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").

        Returns:
            Dict: Position details or empty position on no position/error.
        """
        try:
            response = self.client.get_positions(category="linear", symbol=symbol)
            logger.debug("Position response: %s", response)
            if response.get('retCode') == 0 and response['result']['list']:
                position = response['result']['list'][0]
                logger.info("Position for %s: size=%s, side=%s", symbol, position['size'], position['side'])
                return {
                    "status": "success",
                    "size": float(position['size']),
                    "side": position['side']  # "Buy" or "Sell"
                }
            logger.info("No active position for %s", symbol)
            return {"status": "success", "size": 0.0, "side": None}
        except Exception as e:
            logger.error("Error fetching position: %s", e)
            return {"status": "error", "message": str(e)}

    def check_open_orders(self, symbol: str) -> bool:
        """
        Check if there are open orders for the symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").

        Returns:
            bool: True if open orders exist, False otherwise.
        """
        try:
            response = self.client.get_open_orders(category="linear", symbol=symbol)
            if response.get('retCode') == 0:
                open_orders = response['result']['list']
                logger.info("Open orders for %s: %d", symbol, len(open_orders))
                return len(open_orders) > 0
            else:
                logger.error("Failed to fetch open orders: %s", response.get('retMsg'))
                return False
        except Exception as e:
            logger.error("Error checking open orders: %s", e)
            return False
