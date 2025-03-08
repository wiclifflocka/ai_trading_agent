# main.py
"""
Main Trading Agent Script

Orchestrates trading strategies, risk management, and reporting for the AI trading system
"""

import time
import logging
import os

# Suppress TensorFlow oneDNN messages and set log level
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Bybit API and Client
from bybit_client import BybitClient
from data_pipeline.bybit_api import BybitAPI

# AI & Self-Learning
from ai.self_learning import SelfLearning

# Risk Management Modules
from risk_management.position_sizing import RiskManagement
from risk_management.max_drawdown import MaxDrawdown
from risk_management.stop_loss_take_profit import StopLossTakeProfit
from risk_management.max_loss import MaxLossPerTrade
from risk_management.leverage_control import LeverageControl
from risk_management.trailing_stop import TrailingStopLoss

# Market Insights & Order Book Analysis
from market_insights.market_analysis import MarketInsights
from analysis.order_book_analysis import OrderBookAnalysis
from analysis.ofi_analysis import OFIAnalysis
from analysis.iceberg_detector import IcebergDetector
from analysis.stop_hunt_detector import StopHuntDetector

# Data Collection
from data_pipeline.order_book_collector import OrderBookCollector

# Execution Strategies
from execution.hft_trading import HFTTrading
from execution.market_maker import MarketMaker
from execution.scalping_strategy import ScalpingStrategy

# Trading Strategy
from strategies.trading_strategy import TradingStrategy
from strategies.strategy_switcher import StrategySwitcher

# Tracking & Reporting
from tracking.profit_tracker import ProfitTracker
from tracking.strategy_report import StrategyReport

# Setting up logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Variables
API_KEY = '05EqRWk80CvjiSto64'
API_SECRET = '6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH'
SYMBOL = 'BTCUSDT'
RISK_PERCENTAGE = 1  # Risk 1% of balance per trade
DATA_DIR = 'data'
MAX_LEVERAGE = 10  # Maximum allowed leverage

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info(f"Created directory: {DATA_DIR}")

# Initialize Bybit Client and API
client = BybitClient(API_KEY, API_SECRET, testnet=True)
api = BybitAPI(API_KEY, API_SECRET, testnet=True)

# Fetch initial balance
initial_balance = client.get_balance()
if initial_balance is None or not isinstance(initial_balance, (int, float)):
    logger.error("Failed to fetch initial balance. Exiting.")
    exit(1)
logger.info(f"Initial balance: {initial_balance} USD")

# Initialize Components with initial_balance
order_book_analysis = OrderBookAnalysis(api, symbol=SYMBOL)
risk_manager = RiskManagement(client, account_balance=initial_balance)
max_drawdown = MaxDrawdown(client, initial_balance=initial_balance)
stop_loss_take_profit = StopLossTakeProfit(client)
max_loss = MaxLossPerTrade(client, account_balance=initial_balance)
leverage_control = LeverageControl(client)
trailing_stop_loss = TrailingStopLoss(client)
symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
market_insights = MarketInsights(client, symbols)  # Instance variable
ofi_analysis = OFIAnalysis(api, symbol=SYMBOL)
iceberg_detector = IcebergDetector(api)
stop_hunt_detector = StopHuntDetector(api)
self_learning = SelfLearning(api)
order_book_collector = OrderBookCollector(api)
hft_trading = HFTTrading(api)
market_maker = MarketMaker(api)
scalping_strategy = ScalpingStrategy(api)
strategy_switcher = StrategySwitcher(client)
trading_strategy = TradingStrategy(client)
profit_tracker = ProfitTracker(api, SYMBOL)
strategy_report = StrategyReport(client=client)

def check_risk_management() -> bool:
    try:
        current_balance = client.get_balance()
        if current_balance is None or not isinstance(current_balance, (int, float)):
            logger.error(f"Failed to fetch current balance: {current_balance}. Stopping trading.")
            return False
        logger.info(f"Fetched current balance: {current_balance} USD")

        risk_manager.account_balance = current_balance
        max_loss.account_balance = current_balance

        if max_drawdown.check_drawdown(current_balance):
            logger.warning("Max Drawdown Exceeded, stopping trading!")
            return False

        leverage_control.check_and_set_leverage(SYMBOL)
        leverage = client.get_leverage(SYMBOL)
        if leverage is None:
            logger.error("Failed to fetch leverage, skipping trade execution.")
            return False
        if leverage > MAX_LEVERAGE:
            logger.warning(f"Leverage {leverage}x exceeds maximum allowed {MAX_LEVERAGE}x")
            leverage_control.set_leverage(SYMBOL, MAX_LEVERAGE)
            logger.info(f"Leverage adjusted to {MAX_LEVERAGE}x")

        max_loss_value = max_loss.calculate_max_loss()
        logger.info(f"Maximum allowed loss per trade: {max_loss_value}")
        return True
    except Exception as e:
        logger.error(f"Error in risk management: {e}")
        return False

def execute_trade():
    if not check_risk_management():
        return

    try:
        market_analysis = market_insights.analyze_market().get(SYMBOL, {})  # Use instance variable
        strategy_switcher.switch_strategy_based_on_market_conditions(SYMBOL)
        position_size = risk_manager.calculate_position_size(RISK_PERCENTAGE)
        logger.info(f"Calculated position size: {position_size}")

        entry_price = market_analysis.get('entry_price')
        stop_loss_percentage = market_analysis.get('stop_loss_percentage')
        take_profit_percentage = market_analysis.get('take_profit_percentage')

        if entry_price and stop_loss_percentage and take_profit_percentage:
            stop_loss_take_profit.place_stop_loss(SYMBOL, entry_price, stop_loss_percentage)
            stop_loss_take_profit.place_take_profit(SYMBOL, entry_price, take_profit_percentage)
            logger.info(f"Placing trade with {position_size} units of {SYMBOL}")
            api.place_order(SYMBOL, "Buy", position_size)
            trailing_stop_loss.place_trailing_stop(SYMBOL, entry_price, trail_percentage=1.5)
        else:
            logger.warning("Incomplete market analysis data, skipping trade.")

        strategy_switcher.execute()
    except Exception as e:
        logger.error(f"Error in execute_trade: {e}")
        
def analyze_order_book():
    try:
        order_book_data = order_book_collector.fetch_order_book(SYMBOL)
        if not order_book_data or not isinstance(order_book_data, dict):
            logger.warning("Invalid or no order book data received.")
            return

        ofi = order_book_analysis.calculate_order_flow_imbalance()
        if ofi is not None:
            logger.info(f"OFI: {ofi}")
        ofi_result = ofi_analysis.compute_order_flow_imbalance()
        if ofi_result is not None:
            logger.info(f"OFI from OFIAnalysis: {ofi_result}")
    except Exception as e:
        logger.error(f"Error analyzing order book: {e}")

def execute_ai_trading():
    try:
        historical_data = api.get_recent_trades(SYMBOL)
        if historical_data:
            self_learning.train(historical_data)
            state = None  # Placeholder; replace with actual state preparation if needed
            prediction = self_learning.predict_action(state)
            if prediction == "BUY":
                hft_trading.execute_trade(SYMBOL, "BUY")
            elif prediction == "SELL":
                market_maker.place_orders()
            else:
                logger.info("Holding position, no trade executed.")
        else:
            logger.warning(f"No recent trades data for {SYMBOL}")
    except Exception as e:
        logger.error(f"Error in AI trading: {e}")

def generate_report():
    try:
        profit_tracker.generate_report()
        strategy_report.generate_strategy_report()
    except Exception as e:
        logger.error(f"Error generating report: {e}")

def trading_loop():
    while True:
        try:
            logger.info("Checking market conditions and executing trade...")
            analyze_order_book()
            execute_trade()
            execute_ai_trading()
            generate_report()
            time.sleep(10)
        except Exception as e:
            logger.error(f"Error during trading loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    logger.info("Starting AI Trading Agent...")
    trading_loop()
