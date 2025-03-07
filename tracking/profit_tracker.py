# tracking/profit_tracker.py
"""
ProfitTracker Module

Tracks and reports profit & loss (P&L) for trading activities.
"""

import time
import pandas as pd
from data_pipeline.bybit_api import BybitAPI

class ProfitTracker:
    def __init__(self, api: BybitAPI, symbol: str = "BTCUSDT"):
        """
        Tracks and reports profit & loss (P&L).

        Args:
            api (BybitAPI): An instance of BybitAPI for fetching trading data.
            symbol (str): Trading pair (default: "BTCUSDT").
        """
        self.api = api
        self.symbol = symbol
        self.trade_log = []

    def get_current_position(self):
        """
        Fetches the current position and calculates unrealized P&L.

        Returns:
            float or None: Unrealized P&L if position exists, None otherwise.
        """
        try:
            positions = self.api.get_open_positions(self.symbol)
            if not positions:
                return None

            # Assuming the first position is the relevant one (adjust if multiple positions possible)
            position = positions[0]
            if "entry_price" not in position:
                return None

            entry_price = float(position["entry_price"])
            # Use get_btc_price for BTCUSDT; adjust for other symbols if needed
            mark_price = float(self.api.get_btc_price()) if self.symbol == "BTCUSDT" else None
            if mark_price is None:
                return None

            size = float(position["size"])
            side = position["side"]  # "Buy" or "Sell"

            pnl = (mark_price - entry_price) * size if side == "Buy" else (entry_price - mark_price) * size
            return pnl
        except Exception as e:
            print(f"Error fetching current position: {e}")
            return None

    def record_trade(self, entry_price: float, exit_price: float, size: float, side: str):
        """
        Logs a completed trade.

        Args:
            entry_price (float): Price at which the trade was entered.
            exit_price (float): Price at which the trade was exited.
            size (float): Trade size (quantity).
            side (str): "Buy" or "Sell".
        """
        pnl = (exit_price - entry_price) * size if side == "Buy" else (entry_price - exit_price) * size
        self.trade_log.append({
            "Entry": entry_price,
            "Exit": exit_price,
            "Size": size,
            "Side": side,
            "PnL": pnl
        })

    def generate_report(self):
        """
        Summarizes trading performance.
        """
        if not self.trade_log:
            print("📉 No trades executed yet.")
            return

        df = pd.DataFrame(self.trade_log)
        total_pnl = df["PnL"].sum()
        win_rate = (df["PnL"] > 0).mean() * 100
        avg_pnl = df["PnL"].mean()

        print("\n📊 **Trading Performance Report** 📊")
        print(f"🔹 Total Trades: {len(df)}")
        print(f"🔹 Win Rate: {win_rate:.2f}%")
        print(f"🔹 Average PnL per Trade: {avg_pnl:.4f}")
        print(f"💰 **Total Profit/Loss: {total_pnl:.4f} USDT**")

    def run(self):
        """
        Continuously monitors and reports unrealized P&L.
        """
        while True:
            pnl = self.get_current_position()
            if pnl is not None:
                print(f"📈 Current Unrealized P&L: {pnl:.4f} USDT")
            time.sleep(5)

if __name__ == "__main__":
    # Example usage with credentials
    API_KEY = "your_api_key_here"  # Replace with actual key
    API_SECRET = "your_api_secret_here"  # Replace with actual secret
    api = BybitAPI(API_KEY, API_SECRET, testnet=True)
    tracker = ProfitTracker(api)
    tracker.run()
