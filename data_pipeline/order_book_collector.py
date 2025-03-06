import json
import time
import pandas as pd
from bybit_api import BybitAPI
from clients_bybit import BybitClient
from data_pipeline.bybit_api import BybitAPI

# Initialize API connection
api = BybitAPI()

def fetch_order_book(symbol="BTCUSDT", save_to_csv=True):
    """
    Fetch order book data and store it in a structured format.
    :param symbol: Trading pair (e.g., "BTCUSDT")
    :param save_to_csv: Option to save data for analysis
    """
    try:
        # Fetch order book data from the API
        data = api.get_order_book(symbol)

        # Check if the API response is valid
        if not data or 'bids' not in data or 'asks' not in data:
            print(f"Error: Invalid or empty order book data for {symbol}.")
            return

        bids = sorted(data['bids'], key=lambda x: x[0], reverse=True)  # Highest bid first
        asks = sorted(data['asks'], key=lambda x: x[0])  # Lowest ask first

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

    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")

# Example usage
if __name__ == "__main__":
    while True:
        fetch_order_book(symbol="BTCUSDT")  # You can change the symbol if needed
        time.sleep(2)  # Fetch every 2 seconds

