import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from data_pipeline.bybit_api import BybitAPI
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelfLearning:
    def __init__(self, api: BybitAPI, model_path: str = "trading_model.keras", sequence_length: int = 10):
        """
        Initialize the self-learning model.

        Args:
            api (BybitAPI): Bybit API instance.
            model_path (str): Path to save/load the model.
            sequence_length (int): Number of time steps for LSTM input.
        """
        self.api = api
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.feature_means = None
        self.feature_stds = None

        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info("Loaded existing model from %s", model_path)
        else:
            self.model = self.build_model()
            logger.info("Built new model.")

    def build_model(self) -> Sequential:
        """Build the LSTM model for price prediction."""
        model = Sequential([
            Input(shape=(self.sequence_length, 5)),  # 5 features: OHLCV
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Predict next close price
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using mean and standard deviation."""
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = np.mean(data, axis=0)
            self.feature_stds = np.std(data, axis=0) + 1e-8  # Avoid division by zero
        return (data - self.feature_means) / self.feature_stds

    def prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training."""
        if len(data) < self.sequence_length + 1:
            return np.array([]), np.array([])
        normalized_data = self.normalize_data(data)
        X, y = [], []
        for i in range(len(normalized_data) - self.sequence_length):
            X.append(normalized_data[i:i + self.sequence_length])
            y.append(normalized_data[i + self.sequence_length][3])  # Predict close price
        return np.array(X), np.array(y)

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the model on historical data."""
        X, y = self.prepare_data(data)
        if X.size == 0 or y.size == 0:
            logger.warning("Insufficient data for training.")
            return
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.model.save(self.model_path)
        logger.info("Model trained and saved to %s", self.model_path)

    def predict_action(self, state: np.ndarray, buy_threshold: float = 0.001, sell_threshold: float = 0.001) -> str:
        """
        Predict trading action based on price movement.

        Args:
            state (np.ndarray): Recent OHLCV data.
            buy_threshold (float): % increase to trigger buy.
            sell_threshold (float): % decrease to trigger sell.

        Returns:
            str: "BUY", "SELL", or "HOLD".
        """
        if len(state) < self.sequence_length:
            return "HOLD"
        normalized_state = self.normalize_data(state)
        X = np.array([normalized_state[-self.sequence_length:]])
        predicted_normalized_price = self.model.predict(X, verbose=0)[0][0]
        predicted_price = predicted_normalized_price * self.feature_stds[3] + self.feature_means[3]
        current_price = state[-1][3]
        price_change = (predicted_price - current_price) / current_price
        if price_change > buy_threshold:
            return "BUY"
        elif price_change < -sell_threshold:
            return "SELL"
        return "HOLD"
