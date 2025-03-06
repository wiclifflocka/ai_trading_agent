import pandas as pd

class StrategyReport:
    def __init__(self):
        """
        Tracks the performance of different trading strategies.
        """
        self.strategy_data = []

    def log_trade(self, strategy, pnl):
        """
        Records the outcome of each trade.
        :param strategy: Name of the strategy used.
        :param pnl: Profit or loss from trade.
        """
        self.strategy_data.append({"Strategy": strategy, "PnL": pnl})

    def generate_report(self):
        """
        Summarizes performance for each strategy.
        """
        if not self.strategy_data:
            print("📊 No strategy data available.")
            return

        df = pd.DataFrame(self.strategy_data)
        strategy_summary = df.groupby("Strategy").agg(
            Total_Trades=("PnL", "count"),
            Win_Rate=("PnL", lambda x: (x > 0).mean() * 100),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean")
        ).sort_values(by="Total_PnL", ascending=False)

        print("\n📊 **Strategy Performance Report** 📊")
        print(strategy_summary)

if __name__ == "__main__":
    report = StrategyReport()
    report.generate_report()

