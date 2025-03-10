import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_pipeline.bybit_api import BybitAPI
import logging
import os
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelfLearning:
    def __init__(self, api: BybitAPI, model_path: str = "trading_model_advanced.keras", sequence_length: int = 20):
        """
        Initialize the self-learning trading agent.

        Args:
            api (BybitAPI): Bybit API instance for market data and trading.
            model_path (str): Path to save/load the model.
            sequence_length (int): Number of time steps for LSTM input.
        """
        self.api = api
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.feature_means = None
        self.feature_stds = None
        self.model = self._load_or_build_model()
        
        # Trade state tracking
        self.current_position = None  # Tuple: (side: str, entry_price: float, qty: float)
        self.profit_target = 0.02    # 2% profit target
        self.stop_loss = 0.01        # 1% stop loss
        self.trailing_stop = 0.005   # 0.5% trailing stop
        self.max_risk_percent = 1.0  # Base risk percentage
        
        # Performance tracking
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0

    def _load_or_build_model(self) -> Sequential:
        """Load an existing model or build a new one."""
        if os.path.exists(self.model_path):
            model = load_model(self.model_path)
            logger.info("Loaded model from %s", self.model_path)
        else:
            model = self._build_advanced_model()
            logger.info("Built new model.")
        return model

    def _build_advanced_model(self) -> Sequential:
        """Build a simplified LSTM model to predict the next close price."""
        model = Sequential([
            Input(shape=(self.sequence_length, 5)),  # OHLCV features
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)  # Predict next close price
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        return model

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data dynamically using rolling statistics."""
        self.feature_means = np.mean(data[-self.sequence_length:], axis=0)
        self.feature_stds = np.std(data[-self.sequence_length:], axis=0) + 1e-8  # Avoid division by zero
        return (data - self.feature_means) / self.feature_stds

    def denormalize_price(self, normalized_value: float, feature_idx: int = 3) -> float:
        """Denormalize a value (default: close price) back to its original scale."""
        if self.feature_means is None or self.feature_stds is None:
            raise ValueError("Normalization parameters not set.")
        return normalized_value * self.feature_stds[feature_idx] + self.feature_means[feature_idx]

    def prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training to predict the next close price."""
        if len(data) < self.sequence_length + 1:
            return np.array([]), np.array([])
        normalized_data = self.normalize_data(data)
        X, y = [], []
        for i in range(len(normalized_data) - self.sequence_length):
            seq = normalized_data[i:i + self.sequence_length]
            next_close = normalized_data[i + self.sequence_length][3]  # Next close price
            X.append(seq)
            y.append(next_close)
        return np.array(X), np.array(y)

    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the model on historical data with validation."""
        X, y = self.prepare_data(data)
        if X.size == 0 or y.size == 0:
            logger.warning("Insufficient data for training.")
            return
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        self.model.save(self.model_path)
        logger.info("Model trained and saved to %s. Train loss: %.6f, Val loss: %.6f",
                    self.model_path, history.history['loss'][-1], history.history['val_loss'][-1])

    def update_trade_state(self, side: str, entry_price: float, qty: float):
        """Update the current position state after a trade is executed."""
        self.current_position = (side, entry_price, qty)
        self.total_trades += 1
        logger.info("Trade executed: %s, Entry price: %.2f, Qty: %.3f", side, entry_price, qty)

    def clear_trade_state(self, exit_price: float):
        """Clear trade state and update performance metrics."""
        if self.current_position is None:
            return
        side, entry_price, _ = self.current_position
        profit = (exit_price - entry_price) / entry_price if side == "Buy" else (entry_price - exit_price) / entry_price
        if profit > 0:
            self.win_count += 1
            logger.info("Trade closed with profit: %.2f%%", profit * 100)
        else:
            self.loss_count += 1
            logger.info("Trade closed with loss: %.2f%%", profit * 100)
        self.current_position = None

    def calculate_dynamic_thresholds(self, volatility: float) -> Tuple[float, float]:
        """Adjust buy/sell thresholds based on market volatility."""
        base_threshold = 0.001  # Reduced to 0.1% for more sensitivity
        adjusted = base_threshold * (1 + volatility * 5)  # Reduced scaling factor from 10 to 5
        return adjusted, adjusted

    def get_new_trade_signal(self, state: np.ndarray, current_price: float, volatility: float) -> Optional[str]:
        """
        Generate a signal for new trades based on predicted price change.

        Args:
            state (np.ndarray): Recent market data (OHLCV).
            current_price (float): Current market price.
            volatility (float): Current market volatility.

        Returns:
            Optional[str]: "Buy", "Sell", or None if no trade.
        """
        if len(state) < self.sequence_length:
            logger.warning("State length (%d) < sequence_length (%d), no signal.", len(state), self.sequence_length)
            return None

        # Normalize and predict next close price
        normalized_state = self.normalize_data(state)
        X = np.array([normalized_state[-self.sequence_length:]])
        predicted_normalized_close = self.model.predict(X, verbose=0)[0][0]
        predicted_close = self.denormalize_price(predicted_normalized_close)
        price_change = (predicted_close - current_price) / current_price
        logger.info("Prediction - Predicted close: %.2f, Current: %.2f, Change: %.4f%%",
                    predicted_close, current_price, price_change * 100)

        # Dynamic thresholds
        buy_threshold, sell_threshold = self.calculate_dynamic_thresholds(volatility)
        logger.info("Thresholds - Buy: %.4f%%, Sell: %.4f%%", buy_threshold * 100, sell_threshold * 100)

        if price_change > buy_threshold:
            return "Buy"
        elif price_change < -sell_threshold:
            return "Sell"
        return None

    def manage_position(self, current_price: float) -> str:
        """
        Manage an existing position based on profit targets and stop-loss.

        Args:
            current_price (float): Current market price.

        Returns:
            str: "CLOSE" to exit position, "HOLD" to keep it open.
        """
        if not self.current_position:
            return "HOLD"
        side, entry_price, _ = self.current_position
        if side == "Buy":
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        logger.info("Current profit: %.2f%%", current_profit * 100)

        if current_profit >= self.profit_target:
            logger.info("Profit target hit: %.2f%% >= %.2f%%", current_profit * 100, self.profit_target * 100)
            return "CLOSE"
        elif current_profit <= -self.stop_loss:
            logger.info("Stop loss triggered: %.2f%% <= %.2f%%", current_profit * 100, -self.stop_loss * 100)
            return "CLOSE"
        elif current_profit >= (self.profit_target - self.trailing_stop):
            logger.info("Trailing stop region: %.2f%%, holding", current_profit * 100)
            return "HOLD"  # Simplified; could enhance with peak tracking
        else:
            logger.info("Holding position: profit %.2f%% within thresholds", current_profit * 100)
            return "HOLD"

    def predict_action(self, state: np.ndarray, current_price: float, volatility: float) -> str:
        """
        Predict the trading action based on current state and position.

        Args:
            state (np.ndarray): Recent market data (OHLCV).
            current_price (float): Current market price.
            volatility (float): Current market volatility.

        Returns:
            str: "BUY", "SELL", or "HOLD".
        """
        if self.current_position is None:
            signal = self.get_new_trade_signal(state, current_price, volatility)
            if signal:
                return signal
            return "HOLD"
        else:
            action = self.manage_position(current_price)
            if action == "CLOSE":
                return "SELL" if self.current_position[0] == "Buy" else "BUY"
            return "HOLD"

    def get_performance_metrics(self) -> dict:
        """Return performance metrics for the trading agent."""
        win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
        return {
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": win_rate
        }
