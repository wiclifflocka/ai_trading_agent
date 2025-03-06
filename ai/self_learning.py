import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from data_pipeline.bybit_api import BybitAPI

api = BybitAPI()

class AITrader:
    def __init__(self):
        """
        Initializes AI trading agent with an LSTM-based reinforcement learning model.
        """
        self.model = self.build_model()

    def build_model(self):
        """
        Creates an LSTM-based deep learning model for market prediction.
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(10, 5)),  
            LSTM(32),
            Dense(1, activation="linear")
        ])
        model.compile(loss="mse", optimizer="adam")
        return model

    def train(self, data):
        """
        Trains AI model using historical market data.
        """
        X, y = self.prepare_data(data)
        self.model.fit(X, y, epochs=10, batch_size=32)

    def predict_action(self, state):
        """
        Predicts the best trading action using AI model.
        """
        return self.model.predict(state)

    def prepare_data(self, data):
        """
        Prepares market data for training.
        """
        X, y = [], []
        for i in range(len(data) - 10):
            X.append(data[i:i+10])
            y.append(data[i+10, 2])  # Predicting future mid-price
        return np.array(X), np.array(y)

if __name__ == "__main__":
    ai_trader = AITrader()
    historical_data = api.get_historical_data("BTCUSDT")
    ai_trader.train(historical_data)

