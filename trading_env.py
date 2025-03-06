import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data):  # data = historical order book snapshots
        self.data = data
        self.step = 0
        self.cash = 10000
        self.position = 0
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, action):
        state = extract_features(self.data[self.step])
        price = state[0]  # Mid-price
        reward = 0
        if action == 0 and self.cash >= price:  # Buy
            self.position += 1
            self.cash -= price * (1 + 0.00075)  # Include fee
        elif action == 1 and self.position > 0:  # Sell
            self.position -= 1
            self.cash += price * (1 - 0.00075)
            reward = price - self.data[self.step - 1]["result"]["bids"][0][0]  # Profit
        self.step += 1
        done = self.step >= len(self.data) - 1
        next_state = extract_features(self.data[self.step]) if not done else None
        return next_state, reward, done, {}

    def reset(self):
        self.step = 0
        self.cash = 10000
        self.position = 0
        return extract_features(self.data[0])
