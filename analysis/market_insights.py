import numpy as np
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

class MarketInsights:
    def __init__(self, client, symbols):
        self.client = client
        self.symbols = symbols
        print(f"MarketInsights initialized for symbols: {self.symbols}")

    def analyze_market(self):
        print("Analyzing market data...")


    def get_imbalance_ratio(self):
        """
        Calculates bid-ask imbalance.
        """
        book = api.get_order_book(self.symbol)
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
        trades = api.get_recent_trades(self.symbol)
        large_trades = [trade for trade in trades if float(trade["size"]) > 10]  # Arbitrary threshold

        if len(large_trades) > 5:
            print("🚨 Potential iceberg order detected!")

    def analyze_aggression(self):
        """
        Measures aggressive buying/selling.
        """
        trades = api.get_recent_trades(self.symbol)
        buy_vol = sum(float(t["size"]) for t in trades if t["side"] == "buy")
        sell_vol = sum(float(t["size"]) for t in trades if t["side"] == "sell")

        aggression_ratio = buy_vol / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0
        return aggression_ratio

    def run(self):
        """
        Runs market analysis.
        """
        imbalance = self.get_imbalance_ratio()
        aggression = self.analyze_aggression()
        self.detect_iceberg_orders()

        print(f"📊 **Market Insights:**")
        print(f"🔹 Order Book Imbalance: {imbalance:.4f}")
        print(f"🔹 Order Flow Aggression: {aggression:.4f}")

if __name__ == "__main__":
    insights = MarketInsights()
    insights.run()

