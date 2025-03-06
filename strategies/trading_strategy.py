from pybit.unified_trading import HTTP
import time

# Initialize the client for testnet
client = HTTP(testnet=True)

class TradingStrategy:
    def __init__(self, client):  # ✅ The constructor should accept 'client'
        self.client = client
        print("TradingStrategy initialized.")

    def execute_trade(self):
        print("Executing trade...")

def compute_imbalance(orderbook, N=10):
    """
    Compute the imbalance between the top N bids and asks in the order book.

    Args:
        orderbook (dict): The order book data from Bybit API.
        N (int): Number of top levels to consider for bids and asks.

    Returns:
        float: Imbalance ratio between -1 and 1.
    """
    # Separate bids (buy orders) and asks (sell orders)
    bids = [order for order in orderbook['result'] if order['side'] == 'Buy']
    asks = [order for order in orderbook['result'] if order['side'] == 'Sell']    

    # Sort bids by price descending (highest first) and asks ascending (lowest first)
    bids.sort(key=lambda x: float(x['price']), reverse=True)
    asks.sort(key=lambda x: float(x['price']))

    # Take top N bids and asks
    top_bids = bids[:N]
    top_asks = asks[:N]

    # Calculate total volume for bids and asks
    total_buy_volume = sum(float(order['size']) for order in top_bids)
    total_sell_volume = sum(float(order['size']) for order in top_asks)

    # Compute imbalance
    if total_buy_volume + total_sell_volume == 0:
        return 0  # Avoid division by zero
    imbalance = (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)
    return imbalance

def trading_strategy(symbol="BTCUSD", N=10, threshold=0.2, interval=10):
    """
    Simple trading strategy based on order book imbalance.

    Args:
        symbol (str): Trading pair symbol (e.g., "BTCUSD").
        N (int): Number of top levels to analyze.
        threshold (float): Imbalance threshold for buy/sell decisions.
        interval (int): Time (seconds) between checks.
    """
    while True:
        try:
            # Fetch the order book
            orderbook = client.Market.Market_orderbook(symbol=symbol).result()
            print("Raw response:", orderbook)  # Debugging output

            if not orderbook or 'result' not in orderbook or not orderbook['result']:
                print("Error: Invalid or empty orderbook data.")
                continue  # Skip this iteration

            # Calculate the imbalance
            imbalance = compute_imbalance(orderbook, N)
            print(f"Imbalance: {imbalance:.4f}")

            # Decide to buy, sell, or hold
            if imbalance > threshold:
                print("Buy signal")
                # Uncomment to execute a real buy order (use with caution!)
                # client.Order.Order_new(symbol=symbol, side="Buy", qty=1, order_type="Market").result()
            elif imbalance < -threshold:
                print("Sell signal")
                # Uncomment to execute a real sell order (use with caution!)
                # client.Order.Order_new(symbol=symbol, side="Sell", qty=1, order_type="Market").result()
            else:
                print("Hold")

        except Exception as e:
            print(f"Error during trading loop: {e}")
        
        # Wait before the next check
        time.sleep(interval)

if __name__ == "__main__":
    # Run the strategy
    trading_strategy(symbol="BTCUSD", N=10, threshold=0.2, interval=10)

