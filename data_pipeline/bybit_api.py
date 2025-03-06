import time
from pybit.unified_trading import HTTP
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://api-testnet.bybit.com"  # Use testnet for demo

# Initialize the API client
class BybitAPI:
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )

    def get_btc_price(self):
        try:
            response = self.client.latest_information_for_symbol(symbol="BTCUSDT")
            price = float(response["result"][0]["last_price"])
            logging.info(f"BTC Price: {price}")
            return price
        except Exception as e:
            logging.error(f"Error fetching BTC price: {e}")
            return None

    def get_recent_trades(self, symbol):
        try:
            response = self.client.recent_trading_records(symbol=symbol)
            return response["result"]
        except Exception as e:
            logging.error(f"Error fetching recent trades for {symbol}: {e}")
            return []

    def get_open_positions(self, symbol):
        try:
            positions = self.client.my_position(symbol=symbol)
            return positions["result"]
        except Exception as e:
            logging.error(f"Error fetching positions for {symbol}: {e}")
            return []

    def close_position(self, symbol):
        positions = self.get_open_positions(symbol)
        
        # Find position for given symbol
        position = next((p for p in positions if p["symbol"] == symbol), None)
        
        if position and float(position["size"]) > 0:
            try:
                close_order = self.client.place_active_order(
                    category="linear",
                    symbol=symbol,
                    side="Sell",  # Close the position by selling
                    order_type="Market",
                    qty=str(position["size"]),  # Close the whole position
                    reduce_only=True  # Ensure it reduces the position only
                )
                logging.info(f"Close Position Response: {close_order}")
            except Exception as e:
                logging.error(f"Error closing position for {symbol}: {e}")
        else:
            logging.info(f"No open position to close for {symbol}")
        
    def place_order(self, symbol, side, qty):
        try:
            response = self.client.place_active_order(
                category="linear",
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=str(qty),
            )
            logging.info(f"Order placed successfully: {response}")
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}")
            
def main():
    bybit_api = BybitAPI(API_KEY, API_SECRET)

    # Get BTC price
    price = bybit_api.get_btc_price()
    if price:
        logging.info(f"BTC price: {price}")
    else:
        logging.error("Failed to fetch BTC price.")
    
    # Get recent trades for BTCUSDT
    trades = bybit_api.get_recent_trades("BTCUSDT")
    logging.info(f"Recent Trades: {trades}")
    
    # Example: Place a buy order if no position
    if not trades:
        bybit_api.place_order("BTCUSDT", "Buy", 0.01)  # Adjust qty based on your needs

    # Example: Close position for BTCUSDT
    bybit_api.close_position("BTCUSDT")
    
    # Wait for a while before running the next cycle
    time.sleep(10)

if __name__ == "__main__":
    main()

