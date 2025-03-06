import json
import time
import pandas as pd
from bybit_api import BybitAPI

# Initialize API connection
api = BybitAPI()

def fetch_order_book(symbol="BTCUSDT", save_to_csv=True):
    """
    Fetch order book data and store it in a structured format.
    :param symbol: Trading pair (e.g., "BTCUSDT")
    :param save_to_csv: Option to save data for analysis
    """
    data = api.get_order_book(symbol)
    
    if data:
        bids = sorted(data['bids'], key=lambda x: x[0], reverse=True)  # Highest bid first
        asks = sorted(data['asks'], key=lambda x: x[0])  # Lowest ask first
        
        df_bids = pd.DataFrame(bids, columns=["Price", "Size"])
        df_asks = pd.DataFrame(asks, columns=["Price", "Size"])

        print("\n--- Order Book (Top 5 Levels) ---")
        print("Bids:")
        print(df_bids.head())
        print("\nAsks:")
        print(df_asks.head())

        if save_to_csv:
            timestamp = int(time.time())
            df_bids.to_csv(f"data/bids_{symbol}_{timestamp}.csv", index=False)
            df_asks.to_csv(f"data/asks_{symbol}_{timestamp}.csv", index=False)

# Example usage
if __name__ == "__main__":
    while True:
        fetch_order_book()
        time.sleep(2)  # Fetch every 2 seconds

