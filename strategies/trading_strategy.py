# strategies/trading_strategy.py
import time
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

from bybit_client import BybitClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)

class Config:
    """Configuration parameters for the trading strategy"""
    MODEL_SAVE_DIR = Path("saved_models")
    RETRAIN_INTERVAL_HOURS = 24
    MAX_API_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    MIN_TRAINING_SAMPLES = 500
    MODEL_SAVE_FORMAT = "model_{timestamp}.h5"
    SCALER_SAVE_FORMAT = "scaler_{timestamp}.pkl"

class AdvancedTradingStrategy:
    def __init__(
        self,
        client: BybitClient,
        symbol: str = "BTCUSDT",
        N: int = 10,
        initial_threshold: float = 0.2,
        interval: int = 10,
        lookback_period: int = 20,
        volatility_window: int = 10,
        risk_per_trade: float = 0.01,
        stop_loss_factor: float = 0.02,
        take_profit_factor: float = 0.04,
        lstm_sequence_length: int = 60
    ):
        """Initialize the trading strategy with production-ready features"""
        self.client = client
        self.symbol = symbol
        self.N = N
        self.threshold = initial_threshold
        self.interval = interval
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.risk_per_trade = risk_per_trade
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.lstm_sequence_length = lstm_sequence_length

        self.position = None
        self.entry_price = None
        self.position_size = None
        self.lstm_model = None
        self.scaler = None
        self.last_trained = None

        # Initialize directories
        Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)

        # Load existing model if available
        self._load_latest_model()
        logging.info("Trading strategy initialized")

    def _safe_api_call(self, api_method, *args, **kwargs):
        """Wrapper for safe API calls with retries"""
        for attempt in range(Config.MAX_API_RETRIES):
            try:
                return api_method(*args, **kwargs)
            except Exception as e:
                logging.error(f"API call failed (attempt {attempt+1}): {str(e)}")
                if attempt < Config.MAX_API_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
        raise Exception(f"API method {api_method.__name__} failed after {Config.MAX_API_RETRIES} attempts")

    def compute_imbalance(self, orderbook: Dict) -> float:
        """Compute order book imbalance with error handling"""
        try:
            bids = [(float(p), float(s)) for p, s in orderbook['bids']]
            asks = [(float(p), float(s)) for p, s in orderbook['asks']]
            top_bids = sorted(bids, key=lambda x: x[0], reverse=True)[:self.N]
            top_asks = sorted(asks, key=lambda x: x[0])[:self.N]
            total_buy = sum(s for _, s in top_bids)
            total_sell = sum(s for _, s in top_asks)
            total = total_buy + total_sell
            return (total_buy - total_sell) / total if total > 0 else 0
        except (KeyError, ValueError, IndexError) as e:
            logging.error(f"Order book processing error: {str(e)}")
            return 0

    def get_current_price(self) -> Optional[float]:
        """Get current price with error handling"""
        try:
            return self._safe_api_call(self.client.get_current_price, self.symbol)
        except Exception as e:
            logging.error(f"Failed to get current price: {str(e)}")
            return None

    def get_historical_data(self, limit: int) -> Optional[np.ndarray]:
        """Get historical data with error handling"""
        try:
            data = self._safe_api_call(self.client.get_historical_data, self.symbol, limit=limit)
            return np.array([float(c[4]) for c in data]) if data else None
        except Exception as e:
            logging.error(f"Failed to get historical data: {str(e)}")
            return None

    def calculate_volatility(self, prices: np.ndarray) -> Optional[float]:
        """Calculate volatility with input validation"""
        if len(prices) < 2:
            return None
        try:
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns[-self.volatility_window:])
        except Exception as e:
            logging.error(f"Volatility calculation error: {str(e)}")
            return None

    def _save_model(self, model: Sequential, scaler: MinMaxScaler):
        """Save model and scaler with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Config.MODEL_SAVE_DIR / Config.MODEL_SAVE_FORMAT.format(timestamp=timestamp)
        scaler_path = Config.MODEL_SAVE_DIR / Config.SCALER_SAVE_FORMAT.format(timestamp=timestamp)

        try:
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            logging.info(f"Saved model: {model_path}, scaler: {scaler_path}")
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")

    def _load_latest_model(self):
        """Load the most recent model and scaler"""
        try:
            model_files = sorted(Config.MODEL_SAVE_DIR.glob("model_*.h5"), reverse=True)
            scaler_files = sorted(Config.MODEL_SAVE_DIR.glob("scaler_*.pkl"), reverse=True)

            if model_files and scaler_files:
                latest_model = model_files[0]
                latest_scaler = scaler_files[0]
                self.lstm_model = load_model(latest_model)
                self.scaler = joblib.load(latest_scaler)
                self.last_trained = datetime.fromtimestamp(latest_model.stat().st_mtime)
                logging.info(f"Loaded existing model: {latest_model}")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            self.lstm_model = None
            self.scaler = None

    def _retrain_model_if_needed(self):
        """Check if model needs retraining"""
        if not self.last_trained or \
           (datetime.now() - self.last_trained) > timedelta(hours=Config.RETRAIN_INTERVAL_HOURS):
            logging.info("Initiating model retraining...")
            self.train_lstm_model()

    def train_lstm_model(self):
        """Full training procedure with error handling"""
        try:
            hist_data = self.get_historical_data(
                self.lookback_period + self.volatility_window + self.lstm_sequence_length
            )

            if hist_data is None or len(hist_data) < Config.MIN_TRAINING_SAMPLES:
                logging.error("Insufficient data for training")
                return

            model, scaler = self._train_lstm_model_impl(hist_data)
            self.lstm_model = model
            self.scaler = scaler
            self.last_trained = datetime.now()
            self._save_model(model, scaler)
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            if not self.lstm_model:
                logging.warning("No working model available - using fallback strategy")

    def retrain_model(self):
        """Alias for train_lstm_model to match main.py expectations"""
        self.train_lstm_model()

    def _train_lstm_model_impl(self, data: np.ndarray) -> Tuple[Sequential, MinMaxScaler]:
        """Actual LSTM training implementation"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        X, y = [], []
        for i in range(self.lstm_sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.lstm_sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        return model, scaler

    def predict_price_movement(self, data: np.ndarray) -> Optional[float]:
        """Safe prediction with fallback"""
        if self.lstm_model is None or self.scaler is None:
            logging.warning("No LSTM model available for prediction")
            return None

        try:
            if len(data) < self.lstm_sequence_length:
                logging.error("Insufficient data for prediction")
                return None

            last_sequence = data[-self.lstm_sequence_length:]
            scaled_seq = self.scaler.transform(last_sequence.reshape(-1, 1))
            scaled_seq = scaled_seq.reshape(1, self.lstm_sequence_length, 1)
            predicted = self.lstm_model.predict(scaled_seq, verbose=0)
            return self.scaler.inverse_transform(predicted)[0,0] - data[-1]
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return None

    def get_signal(self) -> float:
        """Generate trading signal based on market data for main.py integration"""
        try:
            orderbook = self._safe_api_call(self.client.get_order_book, self.symbol)
            hist_data = self.get_historical_data(
                self.lookback_period + self.volatility_window + self.lstm_sequence_length
            )

            if not orderbook or hist_data is None:
                logging.error("Failed to fetch market data for signal")
                return 0.0

            imbalance = self.compute_imbalance(orderbook)
            volatility = self.calculate_volatility(hist_data)
            predicted_change = self.predict_price_movement(hist_data)

            if volatility is None or predicted_change is None:
                sma = np.mean(hist_data[-self.lookback_period:])
                current_price = self.get_current_price()
                if current_price is None:
                    return 0.0
                return 1.0 if imbalance > self.threshold and current_price > sma else -1.0 if imbalance < -self.threshold and current_price < sma else 0.0

            self.adjust_threshold(volatility)
            if imbalance > self.threshold and predicted_change > 0:
                return 1.0
            elif imbalance < -self.threshold and predicted_change < 0:
                return -1.0
            return 0.0
        except Exception as e:
            logging.error(f"Signal generation failed: {str(e)}")
            return 0.0

    def adjust_threshold(self, volatility: float):
        """Adjust threshold based on volatility"""
        if volatility is not None:
            self.threshold = max(self.initial_threshold, volatility * 1.5)
            logging.debug(f"Adjusted threshold to {self.threshold:.4f} based on volatility {volatility:.4f}")

    def calculate_position_size(self, balance: float, price: float) -> float:
        """Calculate position size based on risk"""
        try:
            return (balance * self.risk_per_trade) / (price * self.stop_loss_factor)
        except (TypeError, ZeroDivisionError) as e:
            logging.error(f"Position size calculation failed: {str(e)}")
            return 0.0

    def run(self):
        """Main trading loop with production-grade reliability"""
        if not self.lstm_model:
            self.train_lstm_model()

        while True:
            try:
                self._retrain_model_if_needed()
                orderbook = self._safe_api_call(self.client.get_order_book, self.symbol)
                hist_data = self.get_historical_data(
                    self.lookback_period + self.volatility_window + self.lstm_sequence_length
                )

                if not orderbook or hist_data is None:
                    logging.error("Critical data missing - skipping iteration")
                    time.sleep(self.interval)
                    continue

                imbalance = self.compute_imbalance(orderbook)
                volatility = self.calculate_volatility(hist_data)
                current_price = self.get_current_price()
                balance = self._safe_api_call(self.client.get_balance)
                predicted_change = self.predict_price_movement(hist_data)

                if None in [volatility, current_price, balance]:
                    logging.error("Invalid indicator values - skipping iteration")
                    continue

                self.adjust_threshold(volatility)

                if self.position:
                    self.manage_position(current_price)
                    time.sleep(self.interval)
                    continue

                position_size = self.calculate_position_size(balance, current_price)

                if predicted_change is None:
                    sma = np.mean(hist_data[-self.lookback_period:])
                    if imbalance > self.threshold and current_price > sma:
                        self._execute_trade("Buy", position_size, current_price)
                    elif imbalance < -self.threshold and current_price < sma:
                        self._execute_trade("Sell", position_size, current_price)
                else:
                    if imbalance > self.threshold and predicted_change > 0:
                        self._execute_trade("Buy", position_size, current_price)
                    elif imbalance < -self.threshold and predicted_change < 0:
                        self._execute_trade("Sell", position_size, current_price)

            except Exception as e:
                logging.error(f"Critical error in main loop: {str(e)}")
                time.sleep(60)

            time.sleep(self.interval)

    def _execute_trade(self, side: str, size: float, price: float):
        """Execute trade with position management"""
        try:
            self.place_order(side, size)
            self.position = 'long' if side == 'Buy' else 'short'
            self.entry_price = price
            self.position_size = size
            logging.info(f"{side} position opened at {price}")
        except Exception as e:
            logging.error(f"Trade execution failed: {str(e)}")
            self.position = None

    def place_order(self, side: str, qty: float):
        """Safe order placement"""
        try:
            self._safe_api_call(self.client.place_order,
                              self.symbol, qty=qty, side=side, order_type="Market")
            logging.info(f"Order executed: {side} {qty} {self.symbol}")
        except Exception as e:
            logging.error(f"Order placement failed: {str(e)}")
            raise

    def manage_position(self, current_price: float):
        """Position management with error handling"""
        try:
            if self.position == 'long':
                if current_price <= self.entry_price * (1 - self.stop_loss_factor):
                    self.place_order("Sell", self.position_size)
                    self.position = None
                elif current_price >= self.entry_price * (1 + self.take_profit_factor):
                    self.place_order("Sell", self.position_size)
                    self.position = None
            elif self.position == 'short':
                if current_price >= self.entry_price * (1 + self.stop_loss_factor):
                    self.place_order("Buy", self.position_size)
                    self.position = None
                elif current_price <= self.entry_price * (1 - self.take_profit_factor):
                    self.place_order("Buy", self.position_size)
                    self.position = None
        except Exception as e:
            logging.error(f"Position management error: {str(e)}")
            self.position = None

if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"

    try:
        client = BybitClient(api_key, api_secret, testnet=True)
        strategy = AdvancedTradingStrategy(
            client,
            symbol="BTCUSDT",
            N=10,
            initial_threshold=0.2,
            lstm_sequence_length=60
        )
        strategy.run()
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user")
    except Exception as e:
        logging.error(f"Fatal initialization error: {str(e)}")
