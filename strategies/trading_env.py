# strategies/trading_env.py
class TradingEnvironment:
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Simulates a trading environment for strategy testing.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT"). Defaults to "BTCUSDT".
        """
        self.symbol = symbol
        self.balance = 10000  # Starting balance for simulation
        self.position = 0     # Current position size

    def reset(self):
        """
        Resets the trading environment to initial conditions.
        """
        self.balance = 10000
        self.position = 0

    def step(self, action: str, price: float, size: float) -> tuple:
        """
        Takes a trading action and updates the environment.

        Args:
            action (str): "BUY" or "SELL".
            price (float): Price at which the action is taken.
            size (float): Size of the trade.

        Returns:
            tuple: (new_balance, reward, done)
        """
        if action == "BUY":
            cost = price * size
            if cost <= self.balance:
                self.balance -= cost
                self.position += size
                reward = 0  # Placeholder; adjust based on strategy
            else:
                reward = -1  # Penalty for insufficient funds
        elif action == "SELL":
            if self.position >= size:
                revenue = price * size
                self.balance += revenue
                self.position -= size
                reward = revenue - (price * size)  # Simplified reward
            else:
                reward = -1  # Penalty for insufficient position
        else:
            reward = 0

        done = self.balance <= 0  # End if bankrupt
        return self.balance, reward, done

# Example usage
if __name__ == "__main__":
    env = TradingEnvironment()
    print(f"Initial balance: {env.balance}")
    env.step("BUY", 50000, 0.01)
    print(f"New balance: {env.balance}, Position: {env.position}")
