# execution/hft_trading.py
import json
import websocket
import threading

class HFTTrading:
    def __init__(self, client, symbol: str = "BTCUSDT", position_info=None, risk_components=None):
        """
        Implements high-frequency trading using real-time WebSockets.

        Args:
            client: BybitClient instance for placing orders.
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
            position_info: Shared position info from TradingSystem.
            risk_components: Shared risk management components from TradingSystem.
        """
        self.client = client
        self.symbol = symbol
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"  # Base URL for spot order book
        self.ws = None
        self.running = False
        self.thread = None
        self.position_info = position_info or {}
        self.risk_components = risk_components or {}

    def start(self):
        """Start the WebSocket thread for HFT."""
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        print(f"HFTTrading started for {self.symbol}")

    def on_message(self, ws, message):
        """
        Processes real-time order book updates for ultra-fast trades.

        Args:
            ws: WebSocket instance.
            message: Incoming WebSocket message.
        """
        try:
            data = json.loads(message)
            if "topic" not in data or "data" not in data or data["topic"] != f"orderbook.50.{self.symbol}":
                return  # Ignore non-orderbook messages

            orderbook = data["data"]
            if "b" not in orderbook or "a" not in orderbook:
                print(f"Invalid WebSocket data for {self.symbol}")
                return

            bids = sorted(orderbook["b"], key=lambda x: -float(x[0]))[:5]
            asks = sorted(orderbook["a"], key=lambda x: float(x[0]))[:5]

            best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid * 100

            if spread > 0.02 and not self.position_info.get('size', 0):  # Trade only if no position
                qty = 0.01  # Fixed qty, adjust with risk management if needed
                self.client.place_order(self.symbol, "Buy", qty, "Limit", price=best_bid)
                self.client.place_order(self.symbol, "Sell", qty, "Limit", price=best_ask)
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
        price = self.client.get_market_price(symbol)
        if price and not self.position_info.get('size', 0):
            quantity = 0.01  # Fixed quantity for simplicity
            self.client.place_order(symbol, side, quantity, "Limit", price=price)
            print(f"Executed {side} trade for {symbol} at {price}")

    def run(self):
        """
        Establishes WebSocket connection for ultra-low-latency trading.
        """
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_open=self.on_open
        )
        self.ws.run_forever()

    def on_open(self, ws):
        """Subscribe to order book updates on WebSocket connection open."""
        subscription = {
            "op": "subscribe",
            "args": [f"orderbook.50.{self.symbol}"]
        }
        ws.send(json.dumps(subscription))
        print(f"Subscribed to {self.symbol} order book")

    def stop(self):
        """Stop the WebSocket thread."""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print(f"HFTTrading stopped for {self.symbol}")

if __name__ == "__main__":
    from bybit_client import BybitClient
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    hft_trading = HFTTrading(client)
    hft_trading.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        hft_trading.stop()
