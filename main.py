import logging
import time
import os
import sys
from dotenv import load_dotenv
from typing import Tuple, Optional
import numpy as np
from data_pipeline.bybit_api import BybitAPI
from ai.self_learning import SelfLearning

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger = logging.getLogger(__name__)
logger.info("Attempting to load .env from: %s", env_path)
load_dotenv(dotenv_path=env_path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("Logging initialized.")

# Suppress debug messages from noisy libraries
logging.getLogger('h5py').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pybit').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Configuration Variables
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
API_KEY = os.getenv('BYBIT_API_KEY', '').strip()
API_SECRET = os.getenv('BYBIT_API_SECRET', '').strip()
TESTNET = True
MODEL_PATH = "trading_model_advanced.keras"
DATA_DIR = 'data'
BASE_RISK_PERCENTAGE = 1
MAX_LEVERAGE_CAP = 100
KLINE_INTERVAL = "1"
MIN_ORDER_QTY = 0.001
QTY_PRECISION = 3
MAX_POSITION_MULTIPLIER = 5  # Max position size = 5x MIN_ORDER_QTY

# Debug logging for environment variables
logger.info("Loaded SYMBOL: %s", SYMBOL)
logger.info("Loaded API_KEY: %s", API_KEY if API_KEY else "Not set")
logger.info("Loaded API_SECRET: %s", API_SECRET if API_SECRET else "Not set")

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info("Created directory: %s", DATA_DIR)

# Initialize API and Self-Learning Model
try:
    logger.info("Initializing BybitAPI...")
    api = BybitAPI(API_KEY, API_SECRET, testnet=TESTNET)
    logger.info("Initializing SelfLearning...")
    self_learning = SelfLearning(api, model_path=MODEL_PATH)
except Exception as e:
    logger.error("Initialization failed: %s", e)
    sys.exit(1)

def validate_environment():
    """Validate required environment variables."""
    logger.info("Validating environment...")
    if not API_KEY or not API_SECRET:
        logger.error("Missing Bybit API credentials in environment variables. Please set BYBIT_API_KEY and BYBIT_API_SECRET in the .env file.")
        sys.exit(1)
    if not os.path.exists(DATA_DIR):
        logger.error("Data directory %s not found", DATA_DIR)
        sys.exit(1)
    logger.info("Environment validated successfully.")

def fetch_initial_balance() -> float:
    """Fetch and validate initial balance."""
    logger.info("Fetching initial balance...")
    balance_response = api.get_balance()
    try:
        if balance_response.get('ret_code', 0) != 0:
            logger.error("Bybit API error: %s", balance_response.get('ret_msg'))
            sys.exit(1)
        initial_balance = float(balance_response['result']['list'][0]['totalAvailableBalance'])
        logger.info("Initial balance: %.2f USDT", initial_balance)
        return initial_balance
    except (KeyError, ValueError, TypeError) as e:
        logger.error("Balance parsing error: %s. Response: %s", e, balance_response)
        sys.exit(1)

# Validate environment and fetch balance
validate_environment()
initial_balance = fetch_initial_balance()

# Cache for historical data
last_fetch_time = 0
cached_data = None

def fetch_historical_data_with_cache() -> np.ndarray:
    """Fetch historical data with caching."""
    global last_fetch_time, cached_data
    logger.info("Checking historical data cache...")
    if time.time() - last_fetch_time > 60:
        try:
            logger.info("Fetching new Kline data...")
            data = api.get_historical_klines(symbol=SYMBOL, interval=KLINE_INTERVAL, limit=500)
            if not data or 'result' not in data or 'list' not in data['result']:
                logger.warning("No historical Kline data fetched.")
                return np.array([])
            kline_data = data['result']['list']
            formatted_data = []
            for k in kline_data:
                if len(k) < 6:
                    continue
                try:
                    formatted_data.append([float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
                except (ValueError, TypeError):
                    continue
            if not formatted_data:
                logger.warning("No valid Kline data formatted.")
                return np.array([])
            cached_data = np.array(formatted_data)[::-1]
            last_fetch_time = time.time()
            logger.info("Updated historical data cache with %d entries.", len(cached_data))
        except Exception as e:
            logger.error("Error fetching historical data: %s", e)
            return np.array([])
    else:
        logger.info("Using cached data with %d entries.", len(cached_data) if cached_data is not None else 0)
    return cached_data if cached_data is not None else np.array([])

def get_current_balance() -> float:
    """Get current available USDT balance."""
    logger.info("Fetching current balance...")
    try:
        balance_info = api.get_balance()
        if not balance_info or 'result' not in balance_info or 'list' not in balance_info['result']:
            logger.warning("Invalid balance response.")
            return 0.0
        total_available_balance = float(balance_info['result']['list'][0]['totalAvailableBalance'])
        logger.info("Current available USDT balance: %.2f", total_available_balance)
        return total_available_balance
    except Exception as e:
        logger.error("Error fetching current balance: %s", e)
        return 0.0

def get_position() -> Tuple[float, Optional[str]]:
    """Get current position size and side."""
    logger.info("Fetching position for %s...", SYMBOL)
    try:
        position_info = api.get_position(SYMBOL)
        if position_info['status'] == 'success':
            side = position_info['side'] if position_info['side'] else None
            logger.info("Position for %s: size=%.3f, side=%s", SYMBOL, position_info['size'], side)
            return position_info['size'], side
    except Exception as e:
        logger.error("Error fetching position: %s", e)
    return 0.0, None

def calculate_risk_percentage(volatility: float) -> float:
    """Adjust risk percentage based on market volatility."""
    if volatility > 0.05:
        logger.info("High volatility detected (%.4f), reducing risk to 0.5%%", volatility)
        return 0.5
    logger.info("Using base risk percentage: %d%%", BASE_RISK_PERCENTAGE)
    return BASE_RISK_PERCENTAGE

def calculate_leverage(balance: float, price: float, risk_percentage: float) -> int:
    """Calculate leverage based on balance, price, and risk."""
    if balance <= 0 or price <= 0:
        logger.warning("Invalid balance (%.2f) or price (%.2f), using leverage 1x", balance, price)
        return 1
    risk_amount = balance * (risk_percentage / 100)
    position_value = price * MIN_ORDER_QTY
    required_leverage = position_value / risk_amount
    leverage = max(1, min(int(required_leverage), MAX_LEVERAGE_CAP))
    logger.info("Calculated leverage: %dx", leverage)
    return leverage

def calculate_quantity(price: float, balance: float, leverage: int, risk_percentage: float, predicted_change: float) -> float:
    """Calculate trade quantity based on risk, leverage, and predicted change."""
    risk_amount = balance * (risk_percentage / 100)
    if risk_amount > balance:
        logger.warning("Risk amount %.2f exceeds balance %.2f.", risk_amount, balance)
        return 0.0
    base_qty = (risk_amount * leverage) / price
    change_factor = min(abs(predicted_change) / 0.005, MAX_POSITION_MULTIPLIER)  # 0.5% change = 1x, cap at 5x
    qty = base_qty * change_factor
    qty = max(qty, MIN_ORDER_QTY)
    logger.info("Calculated quantity: %.3f (base=%.3f, change_factor=%.2f)", qty, base_qty, change_factor)
    return round(qty, QTY_PRECISION)

def execute_ai_trading():
    """Execute a single trading cycle based on AI prediction."""
    logger.info("Starting trading cycle execution...")
    try:
        historical_data = fetch_historical_data_with_cache()
        if historical_data.size == 0:
            logger.warning("No historical data available, skipping cycle.")
            return

        recent_closes = historical_data[-20:, 3]
        volatility = np.std(recent_closes) / np.mean(recent_closes)
        risk_percentage = calculate_risk_percentage(volatility)

        current_price = historical_data[-1, 3]
        balance = get_current_balance()
        if balance <= 0:
            logger.warning("Insufficient balance (%.2f USDT), skipping cycle.", balance)
            return

        state = historical_data[-self_learning.sequence_length:]
        logger.info("Predicting action with current price: %.2f, volatility: %.4f", current_price, volatility)
        action = self_learning.predict_action(state, current_price, volatility)
        logger.info("Predicted action: %s", action)
        logger.info("Raw action value: '%s'", action)  # Debug exact action string

        predicted_close = self_learning.model.predict(state[np.newaxis, :, :])[0, 0]
        predicted_change = (predicted_close - current_price) / current_price * 100

        if action.upper() == "HOLD":
            logger.info("Holding position.")
            return

        position_size, position_side = get_position()
        logger.info("Current position - size: %.3f, side: %s", position_size, position_side)

        if api.check_open_orders(SYMBOL):
            logger.info("Open orders exist, skipping trade.")
            return

        leverage = calculate_leverage(balance, current_price, risk_percentage)
        leverage_response = api.set_leverage(SYMBOL, leverage)
        if leverage_response['status'] != 'success':
            logger.error("Failed to set leverage: %s", leverage_response.get('message'))
            return

        qty = calculate_quantity(current_price, balance, leverage, risk_percentage, predicted_change)
        max_position_size = MIN_ORDER_QTY * MAX_POSITION_MULTIPLIER
        logger.info("Max position size: %.3f, Calculated qty: %.3f", max_position_size, qty)

        margin_required = (qty * current_price) / leverage
        if margin_required > balance:
            logger.warning("Insufficient margin: required=%.2f USDT, available=%.2f USDT", margin_required, balance)
            qty = (balance * leverage) / current_price
            qty = max(qty, MIN_ORDER_QTY)
            logger.info("Adjusted quantity due to margin: %.3f", qty)

        logger.info("Evaluating trade action: %s", action)
        if action.upper() == "BUY":
            logger.info("Processing BUY action...")
            if position_side == "Sell" and position_size > 0:
                logger.info("Closing short position before buying: qty=%.3f", position_size)
                close_response = api.place_order(SYMBOL, "Buy", position_size)
                if close_response['status'] == 'success':
                    self_learning.clear_trade_state(current_price)
                    logger.info("Short position closed: orderId=%s", close_response['order_id'])
                else:
                    logger.error("Failed to close short: %s", close_response.get('message', 'Unknown error'))
                    return
            if position_side == "Buy" and position_size < max_position_size:
                available_qty = max_position_size - position_size
                trade_qty = min(qty, available_qty)
                logger.info("BUY condition check - position_size: %.3f, max_position_size: %.3f, available_qty: %.3f, trade_qty: %.3f, min_order_qty: %.3f",
                            position_size, max_position_size, available_qty, trade_qty, MIN_ORDER_QTY)
                if trade_qty >= MIN_ORDER_QTY:
                    logger.info("Increasing BUY position: qty=%.3f, price=%.2f, leverage=%dx", trade_qty, current_price, leverage)
                    try:
                        order_response = api.place_order(SYMBOL, "Buy", trade_qty)
                        if order_response['status'] == 'success':
                            self_learning.update_trade_state("Buy", current_price, trade_qty)
                            logger.info("BUY order placed: orderId=%s", order_response['order_id'])
                        else:
                            logger.error("Failed to place BUY order: %s", order_response.get('message', 'Unknown error'))
                    except Exception as e:
                        logger.error("Exception during BUY order placement: %s", e)
                else:
                    logger.info("Trade qty %.3f below minimum %.3f, skipping", trade_qty, MIN_ORDER_QTY)
            elif position_side != "Buy":
                logger.info("Opening BUY position: qty=%.3f, price=%.2f, leverage=%dx", qty, current_price, leverage)
                try:
                    order_response = api.place_order(SYMBOL, "Buy", qty)
                    if order_response['status'] == 'success':
                        self_learning.update_trade_state("Buy", current_price, qty)
                        logger.info("BUY order placed: orderId=%s", order_response['order_id'])
                    else:
                        logger.error("Failed to place BUY order: %s", order_response.get('message', 'Unknown error'))
                except Exception as e:
                    logger.error("Exception during BUY order placement: %s", e)

        elif action.upper() == "SELL":
            logger.info("Processing SELL action...")
            if position_side == "Buy" and position_size > 0:
                logger.info("Closing long position: qty=%.3f", position_size)
                try:
                    close_response = api.place_order(SYMBOL, "Sell", position_size)
                    if close_response['status'] == 'success':
                        self_learning.clear_trade_state(current_price)
                        logger.info("Long position closed: orderId=%s", close_response['order_id'])
                    else:
                        logger.error("Failed to close long: %s", close_response.get('message', 'Unknown error'))
                except Exception as e:
                    logger.error("Exception during SELL order placement: %s", e)
            elif position_side != "Sell" and qty >= MIN_ORDER_QTY:
                logger.info("Opening SELL (short) position: qty=%.3f, price=%.2f, leverage=%dx", qty, current_price, leverage)
                try:
                    order_response = api.place_order(SYMBOL, "Sell", qty)
                    if order_response['status'] == 'success':
                        self_learning.update_trade_state("Sell", current_price, qty)
                        logger.info("SELL order placed: orderId=%s", order_response['order_id'])
                    else:
                        logger.error("Failed to place SELL order: %s", order_response.get('message', 'Unknown error'))
                except Exception as e:
                    logger.error("Exception during SELL order placement: %s", e)

    except Exception as e:
        logger.error("Error in AI trading cycle: %s", e)

def trading_loop():
    """Main trading loop with periodic model retraining."""
    logger.info("Entering trading loop...")
    try:
        logger.info("Fetching initial historical data for training...")
        historical_data = fetch_historical_data_with_cache()
        if historical_data.size > 0:
            logger.info("Starting model training with %d data points...", len(historical_data))
            self_learning.train(historical_data, epochs=10)
        else:
            logger.warning("No data for initial training, proceeding to loop.")

        cycle_count = 0
        while True:
            logger.info("Starting trading cycle %d...", cycle_count + 1)
            execute_ai_trading()
            cycle_count += 1
            if cycle_count % 60 == 0:
                logger.info("Periodic data refresh and retraining...")
                historical_data = fetch_historical_data_with_cache()
                if historical_data.size > 0:
                    self_learning.train(historical_data, epochs=10)
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Trading loop stopped by user.")
    except Exception as e:
        logger.error("Unexpected error in trading loop: %s", e)

if __name__ == "__main__":
    logger.info("Starting AI Trading Agent for Futures...")
    trading_loop()
