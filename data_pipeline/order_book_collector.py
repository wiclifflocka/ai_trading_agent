import json
import time
import pandas as pd
from data_pipeline.bybit_api import BybitAPI

class OrderBookCollector:
    def __init__(self, api: BybitAPI):
        """
        Initialize the OrderBookCollector with a BybitAPI instance.

        Args:
            api (BybitAPI): Instance of BybitAPI for fetching order book data.
        """
        self.api = api

    def fetch_order_book(self, symbol: str = "BTCUSDT", save_to_csv: bool = True) -> dict | None:
        """
        Fetch order book data and store it in a structured format.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            save_to_csv (bool): Option to save data to CSV files. Defaults to True.

        Returns:
            dict | None: Order book data if successful, None if failed.
        """
        try:
            # Fetch order book data from the API
            data = self.api.get_order_book(symbol)

            # Check if the API response is valid
            if not data or 'bids' not in data or 'asks' not in data:
                print(f"Error: Invalid or empty order book data for {symbol}.")
                return None

            bids = sorted(data['bids'], key=lambda x: float(x[0]), reverse=True)  # Highest bid first
            asks = sorted(data['asks'], key=lambda x: float(x[0]))  # Lowest ask first

            # Prepare data for display and saving
            df_bids = pd.DataFrame(bids, columns=["Price", "Size"])
            df_asks = pd.DataFrame(asks, columns=["Price", "Size"])

            # Display the top 5 bids and asks
            print(f"\n--- Order Book (Top 5 Levels) for {symbol} ---")
            print("Bids:")
            print(df_bids.head())
            print("\nAsks:")
            print(df_asks.head())

            # Optionally save the data to CSV
            if save_to_csv:
                timestamp = int(time.time())
                df_bids.to_csv(f"data/bids_{symbol}_{timestamp}.csv", index=False)
                df_asks.to_csv(f"data/asks_{symbol}_{timestamp}.csv", index=False)
                print(f"Data saved as bids_{symbol}_{timestamp}.csv and asks_{symbol}_{timestamp}.csv")

            # Return the raw data for further use
            return {'bids': bids, 'asks': asks}

        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    api = BybitAPI()
    collector = OrderBookCollector(api)
    while True:
        collector.fetch_order_book(symbol="BTCUSDT")
        time.sleep(2)  # Fetch every 2 seconds
