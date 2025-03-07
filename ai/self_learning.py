import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from data_pipeline.bybit_api import BybitAPI

class SelfLearning:
    def __init__(self, api: BybitAPI):
        print("SelfLearning initialized with api:", api)
        self.api = api
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
        Assumes data is a 2D NumPy array with shape (n_samples, 5) 
        (e.g., open, high, low, close, volume).
        """
        X, y = [], []
        for i in range(len(data) - 10):
            X.append(data[i:i+10, :5])  # Take 10 time steps with 5 features
            y.append(data[i+10, 3])  # Predict future close price (column 3)
        return np.array(X), np.array(y)

if __name__ == "__main__":
    # Replace with your actual Bybit API credentials
    API_KEY = "your_api_key_here"
    API_SECRET = "your_api_secret_here"
    
    # Initialize Bybit API with credentials
    api = BybitAPI(API_KEY, API_SECRET, testnet=True)
    
    # Initialize SelfLearning with the API instance
    self_learning = SelfLearning(api)
    
    # Fetch historical data
    historical_data = api.get_historical_data("BTCUSDT")
    
    # Train the model
    self_learning.train(historical_data)
