import pandas as pd

class StrategyReport:
    def __init__(self, client):
        """
        Tracks the performance of different trading strategies.

        Args:
            client: BybitClient instance for fetching account data.
        """
        self.client = client
        self.strategy_data = []

    def log_trade(self, strategy: str, pnl: float):
        """
        Records the outcome of each trade.

        Args:
            strategy (str): Name of the strategy used (e.g., "HFT", "MarketMaker").
            pnl (float): Profit or loss from the trade.
        """
        self.strategy_data.append({"Strategy": strategy, "PnL": pnl})

    def generate_strategy_report(self):  # Renamed for consistency with main.py
        """
        Summarizes performance for each strategy, including account balance.
        """
        if not self.strategy_data:
            print("📊 No strategy data available.")
            return

        # Fetch account balance (assuming client.get_balance exists)
        try:
            balance = self.client.get_balance()
        except Exception as e:
            print(f"Error fetching balance: {e}")
            balance = "N/A"

        # Generate report from logged trades
        df = pd.DataFrame(self.strategy_data)
        strategy_summary = df.groupby("Strategy").agg(
            Total_Trades=("PnL", "count"),
            Win_Rate=("PnL", lambda x: (x > 0).mean() * 100),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean")
        ).sort_values(by="Total_PnL", ascending=False)

        print("\n📊 **Strategy Performance Report** 📊")
        print(f"Current Account Balance: {balance}")
        print(strategy_summary)

# Example usage
if __name__ == "__main__":
    # Mock client for testing (replace with actual BybitClient in real use)
    class MockClient:
        def get_balance(self):
            return 10000.0

    client = MockClient()
    report = StrategyReport(client)
    report.log_trade("HFT", 50.0)
    report.log_trade("HFT", -20.0)
    report.log_trade("MarketMaker", 30.0)
    report.generate_strategy_report()
