# strategies/trading_strategy.py
import time

class TradingStrategy:
    def __init__(self, client=None, symbol=None):
        """
        Initialize the TradingStrategy with a client and optional symbol.

        Args:
            client: An instance of BybitClient for API interactions.
            symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        """
        self.client = client
        self.symbol = symbol or "BTCUSDT"
        print("TradingStrategy initialized.")

    def execute_trade(self):
        """
        Base method to be overridden by subclasses.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement execute_trade")

def compute_imbalance(orderbook, N=10):
    """
    Compute the imbalance between the top N bids and asks in the order book.

    Args:
        orderbook (dict): The order book data from BybitClient.get_order_book, with 'bids' and 'asks'.
        N (int): Number of top levels to consider for bids and asks.

    Returns:
        float: Imbalance ratio between -1 and 1.
    """
    # Extract bids and asks from the order book
    bids = [(float(price), float(size)) for price, size in orderbook['bids']]
    asks = [(float(price), float(size)) for price, size in orderbook['asks']]

    # Sort bids by price descending (highest first) and asks ascending (lowest first)
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])

    # Take top N bids and asks
    top_bids = bids[:N]
    top_asks = asks[:N]

    # Calculate total volume for bids and asks
    total_buy_volume = sum(size for _, size in top_bids)
    total_sell_volume = sum(size for _, size in top_asks)

    # Compute imbalance
    total_volume = total_buy_volume + total_sell_volume
    if total_volume == 0:
        return 0  # Avoid division by zero
    imbalance = (total_buy_volume - total_sell_volume) / total_volume
    return imbalance

def trading_strategy(client, symbol="BTCUSDT", N=10, threshold=0.2, interval=10):
    """
    Simple trading strategy based on order book imbalance.

    Args:
        client: An instance of BybitClient for API interactions.
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        N (int): Number of top levels to analyze.
        threshold (float): Imbalance threshold for buy/sell decisions.
        interval (int): Time (seconds) between checks.
    """
    while True:
        try:
            # Fetch the order book using the provided client
            orderbook = client.get_order_book(symbol)
            print("Order book fetched:", orderbook)  # Debugging output

            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                print("Error: Invalid or empty orderbook data.")
                continue  # Skip this iteration

            # Calculate the imbalance
            imbalance = compute_imbalance(orderbook, N)
            print(f"Imbalance: {imbalance:.4f}")

            # Decide to buy, sell, or hold
            if imbalance > threshold:
                print("Buy signal")
                # Uncomment to execute a real buy order (use with caution!)
                # client.place_order(symbol=symbol, qty=0.001, side="Buy", order_type="Market")
            elif imbalance < -threshold:
                print("Sell signal")
                # Uncomment to execute a real sell order (use with caution!)
                # client.place_order(symbol=symbol, qty=0.001, side="Sell", order_type="Market")
            else:
                print("Hold")

        except Exception as e:
            print(f"Error during trading loop: {e}")

        # Wait before the next check
        time.sleep(interval)

if __name__ == "__main__":
    from bybit_client import BybitClient
    api_key = "YOUR_API_KEY"  # Replace with your actual key
    api_secret = "YOUR_API_SECRET"  # Replace with your actual secret
    client = BybitClient(api_key, api_secret, testnet=True)
    trading_strategy(client, symbol="BTCUSDT", N=10, threshold=0.2, interval=10)
