# main.py

import time
import logging
from bybit_client import BybitClient

# Risk Management Modules
from risk_management.position_sizing import RiskManagement
from risk_management.max_drawdown import MaxDrawdown
from risk_management.stop_loss_take_profit import StopLossTakeProfit
from risk_management.max_loss import MaxLossPerTrade
from risk_management.leverage_control import LeverageControl
from risk_management.risk_manager import RiskManager
from risk_management.trailing_stop import TrailingStopLoss

# Market Insights & Order Book Analysis
from market_insights.market_analysis import MarketInsights
from analysis.order_book_analysis import OrderBookAnalysis
from analysis.ofi_analysis import OFIAnalysis  # Adjusted import path
from analysis.iceberg_detector import IcebergDetector  # Adjusted import path
from analysis.stop_hunt_detector import StopHuntDetector  # Adjusted import path

# AI & Self-Learning
from ai.self_learning import SelfLearning

# Data Collection
from data_pipeline.order_book_collector import OrderBookCollector
from data_pipeline.bybit_api import BybitAPI

# Execution Strategies
from execution.hft_trading import HFTTrading
from execution.market_maker import MarketMaker
from execution.scalping_strategy import ScalpingStrategy

# Trading Strategy
from strategies.trading_strategy import TradingStrategy
from strategies.strategy_switcher import StrategySwitcher
from strategies.trading_env import TradingEnvironment

# Tracking & Reporting
from tracking.profit_tracker import ProfitTracker
from tracking.strategy_report import StrategyReport

# Setting up logging for tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Variables
API_KEY = '05EqRWk80CvjiSto64'
API_SECRET = '6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH'
SYMBOL = 'BTCUSDT'  # Standardized to BTCUSDT for consistency
RISK_PERCENTAGE = 1
MAX_DRAWDOWN_LIMIT = 0.2
LEVERAGE_LIMIT = 10
TRADE_SIZE = 1000

# Initialize Bybit Client
client = BybitClient(API_KEY, API_SECRET)

# Initialize API
api = BybitAPI()

# Initialize OrderBookAnalysis with api
order_book_analysis = OrderBookAnalysis(api, symbol=SYMBOL)

# Get OFI
ofi = order_book_analysis.calculate_order_flow_imbalance()
if ofi is not None:
    if ofi > 0:
        print("Buy signal")
    elif ofi < 0:
        print("Sell signal")
    else:
        print("Neutral")

# Initialize Risk Management System
risk_manager = RiskManagement(client, account_balance=10000)
max_drawdown = MaxDrawdown(client, initial_balance=10000)
stop_loss_take_profit = StopLossTakeProfit(client)
max_loss = MaxLossPerTrade(client, account_balance=10000)
leverage_control = LeverageControl(client)
trailing_stop_loss = TrailingStopLoss(client)

# Initialize Market Insights & Other Analysis
symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
market_insights = MarketInsights(client, symbols)
ofi_analysis = OFIAnalysis(client)
iceberg_detector = IcebergDetector(client)
stop_hunt_detector = StopHuntDetector(client)

# Initialize Self-Learning AI
self_learning = SelfLearning(client)

# Initialize Data Collection
order_book_collector = OrderBookCollector(client)
bybit_api = BybitAPI()  # Note: This might be redundant; consider using `api` instead

# Initialize Trading Strategies
hft_trading = HFTTrading(client)
market_maker = MarketMaker(client)
scalping_strategy = ScalpingStrategy(client)
strategy_switcher = StrategySwitcher(client)
trading_strategy = TradingStrategy(client)

# Initialize Tracking & Reporting
profit_tracker = ProfitTracker(client)
strategy_report = StrategyReport(client)

# Function to check risk and execute trades
def check_risk_management():
    current_balance = client.get_balance()

    if max_drawdown.check_drawdown(current_balance):
        logger.warning("Max Drawdown Exceeded, stopping trading!")
        return False

    leverage_control.check_and_set_leverage(SYMBOL)
    max_loss_value = max_loss.calculate_max_loss()
    logger.info(f"Maximum allowed loss per trade: {max_loss_value}")

    return True

# Function to execute a trade
def execute_trade():
    if not check_risk_management():
        return

    # Assuming analyze_market returns a dict with symbol keys
    market_analysis = market_insights.analyze_market().get(SYMBOL, {})
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
        client.place_order(SYMBOL, position_size, "BUY")

        trailing_stop_loss.place_trailing_stop(SYMBOL, entry_price, trail_percentage=1.5)
    else:
        logger.warning("Incomplete market analysis data, skipping trade.")

# Function to analyze order book and improve AI model
def analyze_order_book():
    order_book_data = order_book_collector.collect_order_book(SYMBOL)

    if not order_book_data:
        logger.warning("No order book data received.")
        return

    order_book_analysis.analyze_order_book(order_book_data)
    ofi_analysis.detect_order_flow_imbalance(order_book_data)
    iceberg_detector.detect_iceberg_orders(order_book_data)
    stop_hunt_detector.detect_stop_hunting_patterns(order_book_data)

# Function to execute AI-based trading strategies
def execute_ai_trading():
    self_learning.train_model()
    prediction = self_learning.predict_market_conditions(SYMBOL)

    if prediction == "BUY":
        hft_trading.execute_trade(SYMBOL, "BUY")
    elif prediction == "SELL":
        market_maker.execute_trade(SYMBOL, "SELL")
    else:
        logger.info("Holding position, no trade executed.")

# Function to generate trading reports
def generate_report():
    profit_tracker.track_profit_loss(SYMBOL)
    strategy_report.generate_strategy_report()

# Main Trading Loop
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

# Start the trading loop
if __name__ == "__main__":
    logger.info("Starting AI Trading Agent...")
    trading_loop()
