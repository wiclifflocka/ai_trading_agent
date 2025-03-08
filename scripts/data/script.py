import requests
import logging
from typing import List, Dict, Any

class BybitAPI:
    def __init__(self, testnet: bool = False):
        """
        Initialize the BybitAPI with the appropriate base URL.

        Args:
            testnet (bool): If True, use the testnet API; otherwise, use mainnet. Defaults to False.
        """
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch recent trading records for a symbol directly from the Bybit API.

        Args:
            symbol (str): The trading pair symbol (e.g., "BTCUSDT").
            limit (int): The number of recent trades to fetch (default: 10).

        Returns:
            List[Dict[str, Any]]: A list of recent trades, where each trade is a dictionary
                                  containing trade details. Returns an empty list if an error occurs.
        """
        endpoint = "/v5/market/recent-trade"
        url = f"{self.base_url}{endpoint}"
        params = {
            "category": "linear",  # For perpetual contracts like BTCUSDT
            "symbol": symbol,
            "limit": limit
        }

        try:
            # Make the GET request to the Bybit API
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

            # Parse the JSON response
            data = response.json()

            # Check if the API returned an error
            if data.get("retCode") != 0:
                self.logger.error(f"API error fetching recent trades for {symbol}: {data.get('retMsg')}")
                return []

            # Extract the list of trades
            trades = data.get("result", {}).get("list", [])
            self.logger.info(f"Fetched {len(trades)} recent trades for {symbol}")
            return trades

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error while fetching recent trades for {symbol}: {e}")
            return []
        except ValueError as e:
            self.logger.error(f"Error parsing JSON response for {symbol}: {e}")
            return []

# Example usage
if __name__ == "__main__":
    api = BybitAPI(testnet=True)  # Use testnet
    trades = api.get_recent_trades(symbol="BTCUSDT", limit=5)
    for trade in trades:
        print(trade)
