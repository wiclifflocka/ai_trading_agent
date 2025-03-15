# analysis/ofi_analysis.py
"""
Order Flow Imbalance (OFI) Analysis Module

Calculates buying/selling pressure using order book data or trade data if provided.
"""

import logging
from typing import Optional, List, Dict
from bybit_client import BybitClient

# Set up logging consistent with other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OFIAnalysis:
    def __init__(self, client: BybitClient, symbol: str = "BTCUSDT"):
        """
        Initialize OFI analyzer.

        Args:
            client (BybitClient): Authenticated BybitClient instance.
            symbol (str): Trading pair (default: "BTCUSDT").
        """
        self.client = client
        self.symbol = symbol
        logger.info(f"OFIAnalysis initialized for {symbol}")

    def compute_order_flow_imbalance(self, levels: int = 5, trades: Optional[List[Dict]] = None) -> Optional[float]:
        """
        Calculate normalized OFI (-1 to 1 range) using trades if provided, otherwise order book data.

        Args:
            levels (int): Number of order book levels to consider (used if trades is None).
            trades (List[Dict], optional): List of recent trades. If provided, computes OFI from trades.

        Returns:
            float: OFI ratio (positive = buy pressure, negative = sell pressure) or 0.0 if failed.
        """
        try:
            if trades is not None:
                # Calculate OFI based on trade data
                if not trades or len(trades) < 1:
                    logger.warning(f"No trades provided for OFI computation for {self.symbol}")
                    return 0.0

                buy_volume = sum(float(trade.get('size', 0)) for trade in trades if trade.get('side', '').lower() == 'buy')
                sell_volume = sum(float(trade.get('size', 0)) for trade in trades if trade.get('side', '').lower() == 'sell')
                total_volume = buy_volume + sell_volume

                if total_volume == 0:
                    logger.debug("Total trade volume is zero, returning OFI of 0.0")
                    return 0.0

                ofi = (buy_volume - sell_volume) / total_volume
                logger.debug(f"Computed OFI from trades: {ofi:.4f} (Buy Vol: {buy_volume:.2f}, Sell Vol: {sell_volume:.2f})")
                return ofi

            # Fallback to order book data if no trades provided
            order_book = self.client.get_order_book(self.symbol)
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                logger.error("Invalid order book response")
                return 0.0

            bids = order_book['bids']
            asks = order_book['asks']

            if not bids or not asks:
                logger.warning("Empty order book data")
                return 0.0

            # Process top levels
            bid_vol = sum(float(b[1]) for b in sorted(
                bids,
                key=lambda x: float(x[0]),
                reverse=True
            )[:levels])

            ask_vol = sum(float(a[1]) for a in sorted(
                asks,
                key=lambda x: float(x[0])
            )[:levels])

            # Calculate normalized OFI
            total_vol = bid_vol + ask_vol
            if total_vol == 0:
                logger.debug("Total order book volume is zero, returning OFI of 0.0")
                return 0.0

            ofi = (bid_vol - ask_vol) / total_vol
            logger.debug(f"Computed OFI from order book: {ofi:.4f} (Bid Vol: {bid_vol:.2f}, Ask Vol: {ask_vol:.2f})")
            return ofi

        except Exception as e:
            logger.error(f"OFI calculation failed: {str(e)}", exc_info=True)
            return 0.0

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    client = BybitClient(api_key, api_secret, testnet=True)
    ofi_analyzer = OFIAnalysis(client, "BTCUSDT")
    
    # Test with order book
    ofi_ob = ofi_analyzer.compute_order_flow_imbalance(levels=5)
    print(f"Order Flow Imbalance (Order Book): {ofi_ob}")
    
    # Test with trades
    trades = client.get_recent_trades("BTCUSDT", limit=100)
    ofi_trades = ofi_analyzer.compute_order_flow_imbalance(trades=trades)
    print(f"Order Flow Imbalance (Trades): {ofi_trades}")
