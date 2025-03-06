import time
import pandas as pd
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

class ProfitTracker:
    def __init__(self, symbol="BTCUSDT"):
        """
        Tracks and reports profit & loss (P&L).
        :param symbol: Trading pair
        """
        self.symbol = symbol
        self.trade_log = []

    def get_current_position(self):
        """
        Fetches the current position and calculates unrealized P&L.
        """
        position = api.get_open_position(self.symbol)
        if not position or "entry_price" not in position:
            return None

        entry_price = float(position["entry_price"])
        mark_price = float(api.get_market_price(self.symbol))
        size = float(position["size"])
        side = position["side"]  # "Buy" or "Sell"

        pnl = (mark_price - entry_price) * size if side == "Buy" else (entry_price - mark_price) * size
        return pnl

    def record_trade(self, entry_price, exit_price, size, side):
        """
        Logs a completed trade.
        """
        pnl = (exit_price - entry_price) * size if side == "Buy" else (entry_price - exit_price) * size
        self.trade_log.append({"Entry": entry_price, "Exit": exit_price, "Size": size, "Side": side, "PnL": pnl})

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
        while True:
            pnl = self.get_current_position()
            if pnl is not None:
                print(f"📈 Current Unrealized P&L: {pnl:.4f} USDT")
            time.sleep(5)

if __name__ == "__main__":
    tracker = ProfitTracker()
    tracker.run()

