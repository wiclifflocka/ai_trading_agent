# analysis/market_insights/market_insights.py
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MarketInsights:
    def __init__(self, client, symbols, timeframe='1m'):
        self.client = client
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.timeframe = timeframe
        logger.info(f"MarketInsights initialized for symbols: {self.symbols} with timeframe: {self.timeframe}")

    def get_imbalance_ratio(self):
        try:
            book = self.client.get_order_book(self.symbols[0])
            if not book or 'bids' not in book or 'asks' not in book:
                logger.warning("No valid order book data")
                return 0.0
            top_bids = sum(float(x[1]) for x in book["bids"][:5])
            top_asks = sum(float(x[1]) for x in book["asks"][:5])
            total = top_bids + top_asks
            return (top_bids - top_asks) / total if total > 0 else 0.0
        except Exception as e:
            logger.error(f"Imbalance ratio calculation failed: {str(e)}")
            return 0.0

    def detect_iceberg_orders(self, trades=None):
        try:
            if trades is None:
                trades = self.client.get_recent_trades(self.symbols[0], limit=100)
            if not trades:
                logger.warning("No trades available for iceberg detection")
                return []
            large_trades = [trade for trade in trades if float(trade.get("size", 0)) > 10]
            if len(large_trades) > 5:
                logger.info("Potential iceberg order detected!")
                return [float(trade["price"]) for trade in large_trades[:5]]
            return []
        except Exception as e:
            logger.error(f"Iceberg detection failed: {str(e)}")
            return []

    def analyze_aggression(self, trades=None):
        try:
            if trades is None:
                trades = self.client.get_recent_trades(self.symbols[0], limit=100)
            if not trades:
                logger.warning("No trades available for aggression analysis")
                return 0.0
            buy_vol = sum(float(t.get("size", 0)) for t in trades if t.get("side") == "Buy")
            sell_vol = sum(float(t.get("size", 0)) for t in trades if t.get("side") == "Sell")
            total_vol = buy_vol + sell_vol
            return buy_vol / total_vol if total_vol > 0 else 0.0
        except Exception as e:
            logger.error(f"Aggression analysis failed: {str(e)}")
            return 0.0

    def run(self, trades=None):
        try:
            imbalance = self.get_imbalance_ratio()
            aggression = self.analyze_aggression(trades)
            icebergs = self.detect_iceberg_orders(trades)
            logger.info(f"Market Insights ({self.timeframe}):")
            logger.info(f"Order Book Imbalance: {imbalance:.4f}")
            logger.info(f"Order Flow Aggression: {aggression:.4f}")
            if icebergs:
                logger.info(f"Detected Iceberg Orders at: {icebergs}")
            return {
                "imbalance": imbalance,
                "aggression": aggression,
                "icebergs": icebergs
            }
        except Exception as e:
            logger.error(f"Market insights run failed: {str(e)}")
            return {"imbalance": 0.0, "aggression": 0.0, "icebergs": []}

if __name__ == "__main__":
    from bybit_client import BybitClient
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    insights = MarketInsights(client, ["BTCUSDT"])
    insights.run()
