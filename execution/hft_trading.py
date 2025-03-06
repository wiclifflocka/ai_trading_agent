import json
import websocket
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

class HFTTrader:
    def __init__(self, symbol="BTCUSDT"):
        """
        Implements high-frequency trading using real-time WebSockets.
        :param symbol: Trading pair
        """
        self.symbol = symbol
        self.ws_url = f"wss://stream.bybit.com/v5/public/orderbook.{symbol}"

    def on_message(self, ws, message):
        """
        Processes real-time order book updates for ultra-fast trades.
        """
        data = json.loads(message)
        bids = sorted(data["bids"], key=lambda x: -float(x[0]))[:5]
        asks = sorted(data["asks"], key=lambda x: float(x[0]))[:5]

        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        spread = (best_ask - best_bid) / best_bid * 100

        if spread > 0.02:  # If spread is wide enough, execute a trade
            api.place_limit_order(self.symbol, "buy", best_bid, 0.01)
            api.place_limit_order(self.symbol, "sell", best_ask, 0.01)
            print(f"🔥 HFT Trade: BUY @ {best_bid}, SELL @ {best_ask}")

    def run(self):
        """
        Establishes WebSocket connection for ultra-low-latency trading.
        """
        ws = websocket.WebSocketApp(self.ws_url, on_message=self.on_message)
        ws.run_forever()

if __name__ == "__main__":
    hft_trader = HFTTrader()
    hft_trader.run()

