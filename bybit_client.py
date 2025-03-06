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

Requirements:
- pybit library (install with `pip install pybit`)
- Valid Bybit testnet API keys (from https://testnet.bybit.com)
"""

from pybit.unified_trading import HTTP

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
            api_secret=api_secret,
            # Optional: Enable request logging
            # enable_time_sync=True,
            # request_timeout=10  
        )

    def get_balance(self, coin: str = 'USDT') -> float:
        """
        Get total equity balance for a specific coin
        
        Args:
            coin (str): Currency to check (default: USDT)
            
        Returns:
            float: Total equity in specified coin
            
        Raises:
            Exception: On API error or missing data
        """
        try:
            response = self.client.get_wallet_balance(
                accountType="UNIFIED",  # Unified account mode
                coin=coin
            )
            # Extract balance from nested response structure
            balance_list = response['result']['list']
            if not balance_list:
                raise Exception("No balance data found")
                
            return float(balance_list[0]['coin'][0]['equity'])
            
        except Exception as e:
            raise Exception(f"Balance check failed: {str(e)}")

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Limit",
        price: float = None,
        time_in_force: str = "GTC"
    ) -> dict:
        """
        Place a new spot market order
        
        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            side (str): 'Buy' or 'Sell'
            qty (float): Order quantity
            order_type (str): 'Limit' or 'Market'
            price (float): Required for limit orders
            time_in_force (str): Order lifetime (GTC, IOC, FOK)
            
        Returns:
            dict: Order response from Bybit
            
        Raises:
            Exception: On placement failure
        """
        try:
            return self.client.place_order(
                category="spot",
                symbol=symbol,
                side=side,
                orderType=order_type,
                qty=str(qty),  # Bybit requires string quantities
                price=str(price) if price else None,
                timeInForce=time_in_force
            )
        except Exception as e:
            raise Exception(f"Order failed: {str(e)}")

    def get_market_price(self, symbol: str) -> float:
        """
        Get latest market price for a symbol
        
        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            float: Last traded price
            
        Raises:
            Exception: On data retrieval failure
        """
        try:
            response = self.client.get_tickers(
                category="spot",
                symbol=symbol
            )
            return float(response['result']['list'][0]['lastPrice'])
        except Exception as e:
            raise Exception(f"Price check failed: {str(e)}")

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
                category="spot",
                symbol=symbol,
                interval=str(interval),
                limit=limit
            )
            return [
                [float(item[1]), float(item[2]), float(item[3]), 
                float(item[4]), float(item[5]), float(item[6])
                ] for item in response['result']['list']
            ]
        except Exception as e:
            raise Exception(f"Historical data failed: {str(e)}")

    def close_all_positions(self, symbol: str) -> dict:
        """
        Close all open positions for a symbol (spot and derivatives)
        
        Args:
            symbol (str): Trading pair to close
            
        Returns:
            dict: API response
            
        Raises:
            Exception: On closure failure
        """
        try:
            # Close spot positions
            self.client.cancel_all_orders(category="spot", symbol=symbol)
            
            # Close derivatives positions
            return self.client.close_position(
                category="linear",
                symbol=symbol,
                positionIdx=0,  # Single-way position
                settleCoin="USDT"
            )
        except Exception as e:
            raise Exception(f"Position closure failed: {str(e)}")

# Example Usage
if __name__ == "__main__":
    # Initialize with testnet keys
    client = BybitClient(
        api_key="05EqRWk80CvjiSto64",
        api_secret="6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH",
        testnet=True
    )

    # Get current BTC price
    try:
        btc_price = client.get_market_price("BTCUSDT")
        print(f"Current BTC Price: ${btc_price:,.2f}")
    except Exception as e:
        print(f"Error: {str(e)}")
