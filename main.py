# main.py

import time
import logging
from bybit_client import BybitClient
from risk_management.position_sizing import RiskManagement
from risk_management.max_drawdown import MaxDrawdown
from risk_management.stop_loss_take_profit import StopLossTakeProfit
from risk_management.max_loss import MaxLossPerTrade
from risk_management.leverage_control import LeverageControl
from risk_management.trailing_stop import TrailingStopLoss
from market_insights.market_analysis import MarketInsights
from strategies.trading_strategy import TradingStrategy  

from strategies.strategy_switcher import StrategySwitcher

# Setting up logging for tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Variables
API_KEY = 'your-api-key-here'
API_SECRET = 'your-api-secret-here'
BASE_CURRENCY = 'USD'
SYMBOL = 'BTCUSD'  # Default trading pair
RISK_PERCENTAGE = 1  # Risk per trade
MAX_DRAWDOWN_LIMIT = 0.2  # Max drawdown limit (20%)
LEVERAGE_LIMIT = 10  # Max leverage limit
TRADE_SIZE = 1000  # Base trade size in USD

# Initialize Bybit Client
client = BybitClient(API_KEY, API_SECRET)

# Initialize Risk Management System
risk_manager = RiskManagement(client, account_balance=10000)  # Example starting balance
max_drawdown = MaxDrawdown(client, initial_balance=10000)  # Example starting balance
stop_loss_take_profit = StopLossTakeProfit(client)
max_loss = MaxLossPerTrade(client, account_balance=10000)  # Example starting balance
leverage_control = LeverageControl(client)
trailing_stop_loss = TrailingStopLoss(client)

# Initialize Market Insights
symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]  # Add the trading pairs you want
market_insights = MarketInsights(client, symbols)

# Initialize Strategy Switcher
strategy_switcher = StrategySwitcher(client)

# Initialize Trading Strategy
trading_strategy = TradingStrategy(client)

# Function to check account status and risk management
def check_risk_management():
    # Check if max drawdown is exceeded
    current_balance = client.get_balance()
    if max_drawdown.check_drawdown(current_balance):
        logger.warning("Max Drawdown Exceeded, stopping trading!")
        return False
    
    # Ensure leverage control is within safe limits
    leverage_control.check_and_set_leverage(SYMBOL)
    
    # Check max loss for trade
    max_loss_value = max_loss.calculate_max_loss()
    logger.info(f"Maximum allowed loss per trade: {max_loss_value}")

    return True

# Function to execute a trade based on the strategy and risk management
def execute_trade():
    if not check_risk_management():
        return

    # Get current market analysis and insights
    market_analysis = market_insights.analyze_market(SYMBOL)
    
    # Determine if the strategy should be changed (e.g., based on volatility)
    strategy_switcher.switch_strategy_based_on_market_conditions(SYMBOL)

    # Calculate position size based on risk management
    position_size = risk_manager.calculate_position_size(RISK_PERCENTAGE)
    logger.info(f"Calculated position size: {position_size}")

    # Get entry price and set stop loss / take profit levels
    entry_price = market_analysis['entry_price']
    stop_loss_percentage = market_analysis['stop_loss_percentage']
    take_profit_percentage = market_analysis['take_profit_percentage']
    
    stop_loss_take_profit.place_stop_loss(SYMBOL, entry_price, stop_loss_percentage)
    stop_loss_take_profit.place_take_profit(SYMBOL, entry_price, take_profit_percentage)

    # Place trade
    logger.info(f"Placing trade with {position_size} units of {SYMBOL}")
    client.place_order(SYMBOL, position_size, "BUY")

    # Implement trailing stop if necessary
    trailing_stop_loss.place_trailing_stop(SYMBOL, entry_price, trail_percentage=1.5)

# Function to monitor and update the trading loop
def trading_loop():
    while True:
        try:
            logger.info("Checking market conditions and executing trade...")
            execute_trade()
            time.sleep(10)  # Sleep for 10 seconds before the next iteration
        except Exception as e:
            logger.error(f"Error during trading loop: {e}")
            time.sleep(10)  # Retry after a short delay

# Start the trading loop
if __name__ == "__main__":
    logger.info("Starting AI Trading Agent...")
    trading_loop()

