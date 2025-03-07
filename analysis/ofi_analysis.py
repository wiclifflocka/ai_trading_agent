# analysis/ofi_analysis.py
"""
Order Flow Imbalance (OFI) Analysis Module

Calculates buying/selling pressure using order book data
"""

import logging
from typing import Optional
from data_pipeline.bybit_api import BybitAPI  # Changed to BybitAPI

logger = logging.getLogger(__name__)

class OFIAnalysis:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT"):
        """
        Initialize OFI analyzer

        Args:
            api: Authenticated BybitAPI instance
            symbol: Trading pair (default: BTCUSDT)
        """
        self.api = api
        self.symbol = symbol

    def compute_order_flow_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Calculate normalized OFI (-1 to 1 range)

        Args:
            levels: Number of order book levels to consider

        Returns:
            OFI ratio (positive = buy pressure, negative = sell pressure) or None if failed
        """
        try:
            # Get order book data using BybitAPI
            order_book = self.api.get_order_book(self.symbol)
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                logger.error("Invalid order book response")
                return None

            bids = order_book['bids']
            asks = order_book['asks']

            if not bids or not asks:
                logger.warning("Empty order book data")
                return None

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
                return 0.0

            return (bid_vol - ask_vol) / total_vol

        except KeyError as e:
            logger.error(f"Missing key in order book: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"OFI calculation failed: {str(e)}")
            return None
