# analysis/market_insights/market_insights.py
import numpy as np

class MarketInsights:
    def __init__(self, client, symbols, timeframe='1m'):
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        print(f"MarketInsights initialized for symbols: {self.symbols} with timeframe: {self.timeframe}")

    def analyze_market(self):
        print("Analyzing market data...")

    def get_imbalance_ratio(self):
        """
        Calculates bid-ask imbalance.
        """
        book = self.client.get_order_book(self.symbols[0])
        if not book:
            return None

        top_bids = sum([float(x[1]) for x in book["bids"][:5]])
        top_asks = sum([float(x[1]) for x in book["asks"][:5]])

        imbalance_ratio = (top_bids - top_asks) / (top_bids + top_asks)
        return imbalance_ratio

    def detect_iceberg_orders(self):
        """
        Identifies iceberg orders (hidden liquidity).
        """
        trades = self.client.get_recent_trades(self.symbols[0])
        large_trades = [trade for trade in trades if float(trade["size"]) > 10]  # Arbitrary threshold

        if len(large_trades) > 5:
            print("🚨 Potential iceberg order detected!")

    def analyze_aggression(self):
        """
        Measures aggressive buying/selling.
        """
        trades = self.client.get_recent_trades(self.symbols[0])
        buy_vol = sum(float(t["size"]) for t in trades if t["side"] == "Buy")  # Capitalized "Buy"
        sell_vol = sum(float(t["size"]) for t in trades if t["side"] == "Sell")  # Capitalized "Sell"

        aggression_ratio = buy_vol / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0
        return aggression_ratio

    def run(self):
        """
        Runs market analysis.
        """
        imbalance = self.get_imbalance_ratio()
        aggression = self.analyze_aggression()
        self.detect_iceberg_orders()

        print(f"📊 **Market Insights ({self.timeframe}):**")
        print(f"🔹 Order Book Imbalance: {imbalance:.4f}")
        print(f"🔹 Order Flow Aggression: {aggression:.4f}")

if __name__ == "__main__":
    from bybit_client import BybitClient
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    insights = MarketInsights(client, ["BTCUSDT"])
    insights.run()
