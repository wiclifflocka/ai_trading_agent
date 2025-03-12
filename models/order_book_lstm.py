# models/order_book_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
import os

logger = logging.getLogger(__name__)

class OrderBookLSTMModel:
    def __init__(self, model_path: str, data_path: str, seq_length: int = 10):
        self.model_path = model_path
        self.data_path = data_path
        self.seq_length = seq_length
        self.model = None
        self.scaler = None

    def load_model(self) -> bool:
        try:
            self.model = load_model(self.model_path)
            logger.info("Loaded pre-trained order book model")
            return True
        except:
            return False

    def train(self, epochs: int = 10, batch_size: int = 32):
        """Train new model with historical data"""
        try:
            if not os.path.exists(self.data_path):
                logger.warning(f"Data file {self.data_path} not found. Skipping training.")
                return False
            data = pd.read_csv(self.data_path)
            processed_data = self._preprocess_data(data)
            
            if len(processed_data) <= self.seq_length:
                logger.warning(f"Insufficient data ({len(processed_data)} rows) for sequence length {self.seq_length}. Skipping training.")
                return False

            X, y = self._create_sequences(processed_data)
            if X.ndim != 3:
                raise ValueError(f"Expected 3D input for LSTM, got shape {X.shape}")

            self.model = self._build_model(X.shape[1:])  # Shape: (timesteps, features)
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            self.model.save(self.model_path)
            logger.info(f"Model trained and saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def predict(self, live_data: np.ndarray) -> float:
        """Predict next price from live order book data"""
        if not self.model:
            logger.warning("No trained model available for prediction")
            return None
        try:
            processed = self._preprocess_live_data(live_data)
            if processed.ndim == 2:  # Add batch dimension if missing
                processed = processed[np.newaxis, ...]
            prediction = self.model.predict(processed, verbose=0)[0][0]
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Normalize and format historical data"""
        return data[["Price", "Size"]].values  # Shape: (n_rows, 2)

    def _preprocess_live_data(self, live_data: np.ndarray) -> np.ndarray:
        """Format live order book data for prediction"""
        return live_data[:, :2]  # Shape: (timesteps, 2)

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create training sequences"""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])  # Shape: (seq_length, 2)
            y.append(data[i + self.seq_length, 0])  # Next price
        return np.array(X), np.array(y)  # X: (samples, seq_length, 2), y: (samples,)

    def _build_model(self, input_shape: tuple) -> Sequential:
        """Construct LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_data(self, new_data: pd.DataFrame):
        """Update model with new order book data"""
        try:
            new_data.to_csv(self.data_path, mode='a', header=not os.path.exists(self.data_path), index=False)
            logger.info("Updated training data with new observations")
        except Exception as e:
            logger.error(f"Data update failed: {str(e)}")
