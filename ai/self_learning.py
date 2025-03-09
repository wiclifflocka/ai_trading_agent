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
        Initialize the SelfLearning class.

        Args:
            api (BybitAPI): Instance of BybitAPI for fetching data.
            model_path (str): Path to save/load the model.
            sequence_length (int): Number of timesteps for input sequences.
        """
        self.api = api
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.feature_means = None
        self.feature_stds = None

        # Load existing model if available, otherwise build a new one
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info("Loaded existing model from %s", model_path)
        else:
            self.model = self.build_model()
            logger.info("Built new model.")

    def build_model(self) -> Sequential:
        """
        Build and return an LSTM model for predicting the next close price.

        Returns:
            Sequential: Compiled Keras model.
        """
        model = Sequential([
            Input(shape=(self.sequence_length, 5)),  # OHLCV: 5 features
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
        """
        Normalize the input data using mean and standard deviation.

        Args:
            data (np.ndarray): Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Normalized data.
        """
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = np.mean(data, axis=0)
            self.feature_stds = np.std(data, axis=0) + 1e-8  # Avoid division by zero
            logger.debug(f"Feature means: {self.feature_means}, Feature stds: {self.feature_stds}")
        normalized_data = (data - self.feature_means) / self.feature_stds
        logger.debug(f"Sample normalized data: {normalized_data[-1]}")  # Log last row
        return normalized_data

    def prepare_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences of data for training.

        Args:
            data (np.ndarray): Array of shape (n_timesteps, n_features).

        Returns:
            tuple: (X, y) where X is input sequences and y is target close prices.
        """
        if len(data) < self.sequence_length + 1:
            logger.warning("Not enough data to prepare sequences.")
            return np.array([]), np.array([])

        normalized_data = self.normalize_data(data)
        X, y = [], []
        for i in range(len(normalized_data) - self.sequence_length):
            X.append(normalized_data[i:i + self.sequence_length])
            y.append(normalized_data[i + self.sequence_length][3])  # Close price
        return np.array(X), np.array(y)

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """
        Train the model on historical data.

        Args:
            data (np.ndarray): Historical data with shape (n_timesteps, 5).
            epochs (int): Number of training epochs (increased to 50).
            batch_size (int): Batch size for training.
        """
        X, y = self.prepare_data(data)
        if X.size == 0 or y.size == 0:
            logger.warning("No valid sequences for training.")
            return

        logger.debug(f"Training with X shape: {X.shape}, y shape: {y.shape}")
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        logger.debug(f"Training loss: {history.history['loss'][-1]}")
        self.model.save(self.model_path, save_format='keras_v3')  # Use native Keras format
        logger.info("Model trained and saved to %s", self.model_path)

    def predict_action(self, state: np.ndarray, buy_threshold: float = 0.001, sell_threshold: float = 0.001) -> str:
        """
        Predict the next trading action based on the current state.

        Args:
            state (np.ndarray): Recent data with shape (n_timesteps, 5).
            buy_threshold (float): Percentage increase to trigger BUY.
            sell_threshold (float): Percentage decrease to trigger SELL.

        Returns:
            str: "BUY", "SELL", or "HOLD".
        """
        if len(state) < self.sequence_length:
            logger.warning("Insufficient state length for prediction.")
            return "HOLD"

        normalized_state = self.normalize_data(state)
        X = np.array([normalized_state[-self.sequence_length:]])
        predicted_normalized_price = self.model.predict(X, verbose=0)[0][0]

        # Denormalize the prediction
        predicted_price = predicted_normalized_price * self.feature_stds[3] + self.feature_means[3]
        current_price = state[-1][3]  # Close price

        price_change = (predicted_price - current_price) / current_price
        logger.debug(f"Current price: {current_price}, Predicted price: {predicted_price}, Price change: {price_change}")

        if price_change > buy_threshold:
            logger.info(f"Price change {price_change} > {buy_threshold}, triggering BUY")
            return "BUY"
        elif price_change < -sell_threshold:
            logger.info(f"Price change {price_change} < {-sell_threshold}, triggering SELL")
            return "SELL"
        else:
            logger.info(f"Price change {price_change} within thresholds, holding")
            return "HOLD"

    def update_model(self, new_data: np.ndarray):
        """
        Incrementally update the model with new data.

        Args:
            new_data (np.ndarray): New data to incorporate.
        """
        self.train(new_data, epochs=1)  # Fine-tune with one epoch
        logger.info("Model updated with new data.")
