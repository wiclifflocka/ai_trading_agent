import json
import websocket
from data_pipeline.bybit_api import BybitAPI

class HFTTrading:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT"):
        """
        Implements high-frequency trading using real-time WebSockets.

        Args:
            api (BybitAPI): Instance of BybitAPI for placing orders.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
        """
        self.api = api
        self.symbol = symbol
        self.ws_url = f"wss://stream.bybit.com/v5/public/orderbook.{symbol}"
        self.ws = None

    def on_message(self, ws, message):
        """
        Processes real-time order book updates for ultra-fast trades.

        Args:
            ws: WebSocket instance.
            message: Incoming WebSocket message.
        """
        try:
            data = json.loads(message)
            if "bids" not in data or "asks" not in data:
                print(f"Invalid WebSocket data for {self.symbol}")
                return

            bids = sorted(data["bids"], key=lambda x: -float(x[0]))[:5]
            asks = sorted(data["asks"], key=lambda x: float(x[0]))[:5]

            best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid * 100

            if spread > 0.02:  # If spread is wide enough, execute a trade
                self.api.place_limit_order(self.symbol, "buy", best_bid, 0.01)
                self.api.place_limit_order(self.symbol, "sell", best_ask, 0.01)
                print(f"🔥 HFT Trade: BUY @ {best_bid}, SELL @ {best_ask}")
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")

    def execute_trade(self, symbol: str, side: str):
        """
        Executes a trade based on AI prediction or external call.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").
            side (str): Trade direction ("BUY" or "SELL").
        """
        price = self.api.get_current_price(symbol)
        if price:
            quantity = 0.01  # Fixed quantity for simplicity
            self.api.place_limit_order(symbol, side.lower(), price, quantity)
            print(f"Executed {side} trade for {symbol} at {price}")

    def run(self):
        """
        Establishes WebSocket connection for ultra-low-latency trading.
        """
        self.ws = websocket.WebSocketApp(self.ws_url, on_message=self.on_message)
        self.ws.run_forever()

# Example usage
if __name__ == "__main__":
    api = BybitAPI()
    hft_trading = HFTTrading(api)
    hft_trading.run()
