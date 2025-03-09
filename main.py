import logging
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from data_pipeline.bybit_api import BybitAPI
from ai.self_learning import SelfLearning

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Variables
SYMBOL = "BTCUSDT"  # Trading pair for linear perpetual futures
API_KEY = "05EqRWk80CvjiSto64"  # Replace with your actual API key
API_SECRET = "6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH"  # Replace with your actual API secret
TESTNET = True  # Set to False for mainnet
MODEL_PATH = "trading_model.keras"  # Path to save/load the model
DATA_DIR = 'data'
RISK_PERCENTAGE = 1  # Risk 1% of balance per trade as margin
MAX_LEVERAGE_CAP = 100  # Maximum allowable leverage (Bybit's limit for BTCUSDT)
KLINE_INTERVAL = "1"  # Kline interval (e.g., '1', '5', '60', 'D')
MIN_ORDER_QTY = 0.001  # Minimum order quantity for BTCUSDT linear perpetual
QTY_PRECISION = 3  # Precision for quantity (e.g., 0.001 BTC)
BUY_THRESHOLD = 0.001  # 0.1% price increase to buy
SELL_THRESHOLD = 0.001  # 0.1% price decrease to sell

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info(f"Created directory: {DATA_DIR}")

# Initialize API and Self-Learning Model
api = BybitAPI(API_KEY, API_SECRET, testnet=TESTNET)
self_learning = SelfLearning(api, model_path=MODEL_PATH)

# Fetch initial balance
initial_balance_response = api.get_balance()
if initial_balance_response is None or 'result' not in initial_balance_response or 'list' not in initial_balance_response['result']:
    logger.error("Failed to fetch initial balance. Response: %s", initial_balance_response)
    exit(1)
try:
    initial_balance = float(initial_balance_response['result']['list'][0]['totalAvailableBalance'])
    logger.info(f"Initial balance: {initial_balance:.2f} USDT")
except (KeyError, ValueError, TypeError) as e:
    logger.error("Error parsing initial balance: %s. Response: %s", e, initial_balance_response)
    exit(1)

def fetch_historical_data() -> np.ndarray:
    """
    Fetch and format historical Kline data from Bybit for futures trading.

    Returns:
        np.ndarray: Array of shape (n_timesteps, 5) with [open, high, low, close, volume].
    """
    try:
        data = api.get_historical_klines(symbol=SYMBOL, interval=KLINE_INTERVAL, limit=500)  # Increased to 500
        if not data or 'result' not in data or 'list' not in data['result']:
            logger.warning("No historical Kline data fetched.")
            return np.array([])
        kline_data = data['result']['list']  # Access nested 'list' under 'result'
        formatted_data = []
        for k in kline_data:
            if len(k) < 6:
                logger.warning(f"Invalid Kline data entry: {k}")
                continue
            try:
                formatted_data.append([
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5])   # volume
                ])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting Kline data entry {k}: {e}")
                continue
        if not formatted_data:
            logger.warning("No valid Kline data entries after processing.")
            return np.array([])
        return np.array(formatted_data)[::-1]  # Reverse to chronological order
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return np.array([])

def get_current_balance() -> float:
    """
    Fetch the current available USDT balance for trading.

    Returns:
        float: Available USDT balance, or 0.0 on error.
    """
    try:
        balance_info = api.get_balance()
        if balance_info is None or 'result' not in balance_info or 'list' not in balance_info['result']:
            logger.error("Failed to fetch current balance. Response: %s", balance_info)
            return 0.0
        total_available_balance = float(balance_info['result']['list'][0]['totalAvailableBalance'])
        logger.info(f"Current available USDT balance: {total_available_balance:.2f}")
        return total_available_balance
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error parsing current balance: %s. Response: %s", e, balance_info)
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching current balance: %s", e)
        return 0.0

def get_position() -> tuple[float, str]:
    """
    Fetch the current position for the trading symbol.

    Returns:
        tuple: (size, side) where size is the position size and side is "Buy", "Sell", or None.
    """
    try:
        position_info = api.get_position(SYMBOL)
        if position_info['status'] == 'success':
            return position_info['size'], position_info['side']
        logger.error(f"Failed to fetch position: {position_info.get('message')}")
        return 0.0, None
    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        return 0.0, None

def calculate_leverage(balance: float, price: float, risk_percentage: float = RISK_PERCENTAGE) -> int:
    """
    Calculate dynamic leverage based on account balance and current price.

    Args:
        balance (float): Current available USDT balance.
        price (float): Current price of the asset (e.g., BTC price in USDT).
        risk_percentage (float): Percentage of balance to risk per trade.

    Returns:
        int: Calculated leverage, capped at MAX_LEVERAGE_CAP.
    """
    if balance <= 0 or price <= 0:
        logger.error("Invalid balance or price for leverage calculation.")
        return 1  # Default to minimum leverage

    risk_amount = balance * (risk_percentage / 100)
    position_value = price * MIN_ORDER_QTY
    required_leverage = position_value / risk_amount

    leverage = max(1, min(int(required_leverage), MAX_LEVERAGE_CAP))
    logger.info(f"Calculated leverage: {leverage}x for balance {balance:.2f} USDT and price {price:.2f}")
    return leverage

def calculate_quantity(price: float, balance: float, leverage: int) -> float:
    """
    Calculate the trade quantity for futures trading based on leverage and risk.

    Args:
        price (float): Current price of the asset (e.g., BTC price in USDT).
        balance (float): Current available USDT balance.
        leverage (int): Leverage to use for the trade.

    Returns:
        float: Quantity to trade, or 0 if insufficient balance.
    """
    risk_amount = balance * (RISK_PERCENTAGE / 100)  # Margin in USDT
    if risk_amount > balance:
        logger.warning(f"Risk amount ({risk_amount:.2f}) exceeds available balance ({balance:.2f}).")
        return 0.0
    qty = (risk_amount * leverage) / price  # Quantity in BTC
    qty = max(qty, MIN_ORDER_QTY)  # Ensure minimum order quantity
    return round(qty, QTY_PRECISION)

def execute_ai_trading():
    """Execute AI-based trading using self-learning predictions with dynamic leverage."""
    leverage_set = False
    current_leverage = None

    try:
        historical_data = fetch_historical_data()
        if historical_data.size == 0:
            logger.warning(f"No recent Kline data for {SYMBOL}")
            return

        # Train the model
        self_learning.train(historical_data)
        state = historical_data[-self_learning.sequence_length:]
        prediction = self_learning.predict_action(state, BUY_THRESHOLD, SELL_THRESHOLD)

        if prediction not in ["BUY", "SELL"]:
            logger.info("Holding position, no trade executed.")
            return

        # Fetch balance, price, and position
        current_price = historical_data[-1, 3]  # Latest close price
        balance = get_current_balance()
        if balance <= 0:
            logger.warning("Insufficient balance to trade.")
            return
        position_size, position_side = get_position()

        # Set leverage once or if changed
        leverage = calculate_leverage(balance, current_price)
        if not leverage_set or leverage != current_leverage:
            leverage_response = api.set_leverage(SYMBOL, leverage)
            if leverage_response['status'] == 'success':
                leverage_set = True
                current_leverage = leverage
            elif leverage_response.get('message') == "leverage not modified":
                logger.info("Leverage already set to %d for %s", leverage, SYMBOL)
                leverage_set = True
                current_leverage = leverage
            else:
                logger.error("Failed to set leverage: %s", leverage_response.get('message'))
                return

        # Execute trade based on position
        if prediction == "BUY" and position_side != "Buy":
            qty = calculate_quantity(current_price, balance, leverage)
            if qty > 0:
                logger.info(f"Executing BUY trade with quantity {qty:.3f} at price {current_price:.2f} with leverage {leverage}x.")
                api.place_order(SYMBOL, "Buy", qty)
            else:
                logger.info("Skipping BUY trade due to insufficient balance or invalid quantity.")
        elif prediction == "SELL" and position_side == "Buy":
            qty = position_size  # Sell entire position
            if qty > 0:
                logger.info(f"Executing SELL trade with quantity {qty:.3f} at price {current_price:.2f} with leverage {leverage}x.")
                api.place_order(SYMBOL, "Sell", qty)
            else:
                logger.info("Skipping SELL trade due to no position or invalid quantity.")
        else:
            logger.info(f"No action taken: {prediction} requested but position side is {position_side}")

    except Exception as e:
        logger.error(f"Error in AI trading: {e}")

def trading_loop():
    """Main trading loop that runs continuously."""
    try:
        while True:
            logger.info("Executing AI trading cycle...")
            execute_ai_trading()
            time.sleep(10)  # Sleep for 10 seconds between cycles
    except KeyboardInterrupt:
        logger.info("Trading loop stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error in trading loop: {e}")

if __name__ == "__main__":
    logger.info("Starting AI Trading Agent for Futures...")
    trading_loop()
