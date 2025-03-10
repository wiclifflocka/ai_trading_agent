import logging
import time
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow warnings

import numpy as np
from data_pipeline.bybit_api import BybitAPI
from ai.self_learning import SelfLearning

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Variables
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')  # Trading pair
API_KEY = os.getenv('BYBIT_API_KEY')  # Load from environment
API_SECRET = os.getenv('BYBIT_API_SECRET')  # Load from environment
TESTNET = True  # Use testnet
MODEL_PATH = "trading_model.keras"
DATA_DIR = 'data'
BASE_RISK_PERCENTAGE = 1  # Base risk percentage (%)
MAX_LEVERAGE_CAP = 100  # Bybit max leverage for BTCUSDT
KLINE_INTERVAL = "1"  # 1-minute Kline interval
MIN_ORDER_QTY = 0.001  # Minimum BTC order size
QTY_PRECISION = 3  # Decimal places for quantity
BUY_THRESHOLD = 0.001  # 0.1% price increase to buy
SELL_THRESHOLD = 0.001  # 0.1% price decrease to sell

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info("Created directory: %s", DATA_DIR)

# Initialize API and Self-Learning Model
api = BybitAPI(API_KEY, API_SECRET, testnet=TESTNET)
self_learning = SelfLearning(api, model_path=MODEL_PATH)

def validate_environment():
    """Validate required environment variables."""
    if not API_KEY or not API_SECRET:
        logger.error("Missing Bybit API credentials in environment variables")
        exit(1)
    if not os.path.exists(DATA_DIR):
        logger.error("Data directory %s not found", DATA_DIR)
        exit(1)

# Fetch and log initial balance
def fetch_initial_balance():
    """Fetch and validate initial balance."""
    balance_response = api.get_balance()
    try:
        if balance_response.get('ret_code', 0) != 0:
            logger.error("Bybit API error: %s", balance_response.get('ret_msg'))
            exit(1)
            
        initial_balance = float(balance_response['result']['list'][0]['totalAvailableBalance'])
        logger.info("Initial balance: %.2f USDT", initial_balance)
        return initial_balance
    except (KeyError, ValueError, TypeError) as e:
        logger.error("Balance parsing error: %s. Response: %s", e, balance_response)
        exit(1)

# Validate environment and fetch balance first
validate_environment()
initial_balance = fetch_initial_balance()

# Cache for historical data
last_fetch_time = 0
cached_data = None

def fetch_historical_data_with_cache() -> np.ndarray:
    """Fetch historical data with caching to reduce API calls."""
    global last_fetch_time, cached_data
    if time.time() - last_fetch_time > 60:  # Refresh every 60 seconds
        try:
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
                return np.array([])
            cached_data = np.array(formatted_data)[::-1]  # Reverse to chronological order
            last_fetch_time = time.time()
            logger.info("Updated historical data cache.")
        except Exception as e:
            logger.error("Error fetching historical data: %s", e)
            return np.array([])
    return cached_data

def get_current_balance() -> float:
    """Get current available USDT balance."""
    try:
        balance_info = api.get_balance()
        if not balance_info or 'result' not in balance_info or 'list' not in balance_info['result']:
            return 0.0
        total_available_balance = float(balance_info['result']['list'][0]['totalAvailableBalance'])
        logger.info("Current available USDT balance: %.2f", total_available_balance)
        return total_available_balance
    except Exception as e:
        logger.error("Error fetching current balance: %s", e)
        return 0.0

def get_position() -> tuple[float, str]:
    """Get current position size and side."""
    try:
        position_info = api.get_position(SYMBOL)
        if position_info['status'] == 'success':
            return position_info['size'], position_info['side']
        return 0.0, None
    except Exception as e:
        logger.error("Error fetching position: %s", e)
        return 0.0, None

def calculate_risk_percentage(volatility: float) -> float:
    """
    Adjust risk percentage based on market volatility.

    Args:
        volatility (float): Market volatility (std dev / mean).

    Returns:
        float: Adjusted risk percentage.
    """
    if volatility > 0.05:  # High volatility threshold
        logger.info("High volatility detected (%.4f), reducing risk to 0.5%%", volatility)
        return 0.5
    return BASE_RISK_PERCENTAGE

def calculate_leverage(balance: float, price: float, risk_percentage: float) -> int:
    """Calculate leverage based on balance, price, and risk."""
    if balance <= 0 or price <= 0:
        return 1
    risk_amount = balance * (risk_percentage / 100)
    position_value = price * MIN_ORDER_QTY
    required_leverage = position_value / risk_amount
    leverage = max(1, min(int(required_leverage), MAX_LEVERAGE_CAP))
    logger.info("Calculated leverage: %dx", leverage)
    return leverage

def calculate_quantity(price: float, balance: float, leverage: int, risk_percentage: float) -> float:
    """Calculate trade quantity based on risk and leverage."""
    risk_amount = balance * (risk_percentage / 100)
    if risk_amount > balance:
        return 0.0
    qty = (risk_amount * leverage) / price
    qty = max(qty, MIN_ORDER_QTY)
    return round(qty, QTY_PRECISION)

def execute_ai_trading():
    """Execute a single trading cycle based on AI prediction."""
    try:
        historical_data = fetch_historical_data_with_cache()
        if historical_data.size == 0:
            logger.warning("No historical data available, skipping cycle.")
            return

        # Calculate volatility from recent close prices
        recent_closes = historical_data[-20:, 3]
        volatility = np.std(recent_closes) / np.mean(recent_closes)
        risk_percentage = calculate_risk_percentage(volatility)

        state = historical_data[-self_learning.sequence_length:]
        prediction = self_learning.predict_action(state, BUY_THRESHOLD, SELL_THRESHOLD)

        if prediction not in ["BUY", "SELL"]:
            logger.info("Prediction: %s - Holding position.", prediction)
            return

        current_price = historical_data[-1, 3]
        balance = get_current_balance()
        if balance <= 0:
            logger.warning("Insufficient balance (%.2f USDT), skipping trade.", balance)
            return
        position_size, position_side = get_position()

        # Prevent over-trading by checking open orders
        if api.check_open_orders(SYMBOL):
            logger.info("Open orders exist, skipping trade.")
            return

        leverage = calculate_leverage(balance, current_price, risk_percentage)
        leverage_response = api.set_leverage(SYMBOL, leverage)
        if leverage_response['status'] != 'success':
            logger.error("Failed to set leverage: %s", leverage_response.get('message'))
            return

        if prediction == "BUY" and position_side != "Buy":
            qty = calculate_quantity(current_price, balance, leverage, risk_percentage)
            if qty > 0:
                logger.info("Executing BUY: qty=%.3f, price=%.2f, leverage=%dx", qty, current_price, leverage)
                api.place_order(SYMBOL, "Buy", qty)
            else:
                logger.info("Skipping BUY: insufficient balance or invalid qty.")
        elif prediction == "SELL" and position_side == "Buy":
            qty = position_size
            if qty > 0:
                logger.info("Executing SELL: qty=%.3f, price=%.2f, leverage=%dx", qty, current_price, leverage)
                api.place_order(SYMBOL, "Sell", qty)
            else:
                logger.info("Skipping SELL: no position or invalid qty.")
        else:
            logger.info("No action: %s requested, position side=%s", prediction, position_side)

    except Exception as e:
        logger.error("Error in AI trading cycle: %s", e)

def trading_loop():
    """Main trading loop with periodic model retraining."""
    try:
        historical_data = fetch_historical_data_with_cache()
        if historical_data.size > 0:
            self_learning.train(historical_data)

        cycle_count = 0
        while True:
            logger.info("Starting trading cycle %d...", cycle_count + 1)
            execute_ai_trading()
            cycle_count += 1
            if cycle_count % 60 == 0:  # Retrain every 60 cycles (~10 minutes)
                historical_data = fetch_historical_data_with_cache()
                if historical_data.size > 0:
                    self_learning.train(historical_data)
            time.sleep(10)  # Wait 10 seconds between cycles
    except KeyboardInterrupt:
        logger.info("Trading loop stopped by user.")
    except Exception as e:
        logger.error("Unexpected error in trading loop: %s", e)

if __name__ == "__main__":
    logger.info("Starting AI Trading Agent for Futures...")
    trading_loop()
    
