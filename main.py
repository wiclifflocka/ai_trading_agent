# main.py
import logging
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import threading
from datetime import datetime, timedelta
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import io
import sys
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Risk Management
from risk_management.leverage_control import LeverageControl
from risk_management.max_drawdown import MaxDrawdown
from risk_management.max_loss import MaxLossPerTrade
from risk_management.position_sizing import RiskManagement as PositionSizing
from risk_management.risk_manager import RiskManager
from risk_management.stop_loss_take_profit import StopLossTakeProfit
from risk_management.trailing_stop import TrailingStopLoss

# Strategies and Analysis
from strategies.trading_strategy import AdvancedTradingStrategy
from analysis.iceberg_detector import IcebergDetector
from analysis.market_insights.market_insights import MarketInsights
from analysis.ofi_analysis import OFIAnalysis
from analysis.order_book_analysis import OrderBookAnalysis
from analysis.order_timing import OrderTimingOptimizer
from analysis.stop_hunt_detector import StopHuntDetector
from models.order_book_lstm import OrderBookLSTMModel

# Execution Strategies
from execution.hft_trading import HFTTrading
from execution.market_maker import MarketMaker
from execution.scalping_strategy import ScalpingStrategy

# Tracking Modules
from tracking.profit_tracker import ProfitTracker
from tracking.strategy_report import StrategyReport

# Self-Learning Module
from ai.self_learning import SelfLearning

# Client
from bybit_client import BybitClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# Trading configurations from .env
SYMBOL_BTC = os.getenv('SYMBOL_BTC', 'BTCUSDT')
TRADE_SIZE_BTC = float(os.getenv('TRADE_SIZE_BTC', '0.01'))
MAX_POSITION_BTC = float(os.getenv('MAX_POSITION_BTC', '0.1'))

# Enable flags
ENABLE_PROFIT_TRACKER = os.getenv('ENABLE_PROFIT_TRACKER', 'True').lower() == 'true'
ENABLE_STRATEGY_REPORT = os.getenv('ENABLE_STRATEGY_REPORT', 'True').lower() == 'true'
ENABLE_MARKET_INSIGHTS = os.getenv('ENABLE_MARKET_INSIGHTS', 'True').lower() == 'true'

class TradingSystem:
    def __init__(self):
        """Initialize the complete trading system"""
        self.client = self._initialize_client()
        self.symbol = SYMBOL_BTC  # Use SYMBOL_BTC from .env
        self.initial_balance = self._get_initial_balance()
        self.trade_size = TRADE_SIZE_BTC  # Add trade size from .env
        self.max_position = MAX_POSITION_BTC  # Add max position from .env
        self.order_book_model = None
        self.execution_strategies = {}
        self.active_strategy = None
        self.strategy_thread = None
        self.running = True

        # State tracking (moved before execution strategies)
        self.position_info = {
            'size': 0.0,
            'side': None,
            'entry_price': 0.0,
            'timestamp': None
        }

        # Initialize components
        self.risk_components = self._initialize_risk_management()
        self.analysis_components = self._initialize_analysis_tools()
        self.trading_strategy = self._initialize_trading_strategy()
        self.tracking_components = self._initialize_tracking_components()
        self.learning_components = self._initialize_learning_components()
        self._initialize_order_book_model()
        self._initialize_execution_strategies()

        # Directory for reports
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def _initialize_client(self) -> BybitClient:
        """Initialize and verify API connection"""
        try:
            client = BybitClient(API_KEY, API_SECRET, testnet=TESTNET)
            logger.info("Successfully connected to Bybit API")
            return client
        except Exception as e:
            logger.error(f"Client initialization failed: {str(e)}")
            raise

    def _initialize_execution_strategies(self):
        """Initialize all execution strategies with shared state"""
        common_args = {
            'position_info': self.position_info,
            'risk_components': self.risk_components
        }

        self.execution_strategies = {
            'hft': HFTTrading(self.client, self.symbol, **common_args),
            'market_making': MarketMaker(self.client, self.symbol, **common_args),
            'scalping': ScalpingStrategy(self.client, self.symbol, **common_args),
            'default': self.trading_strategy
        }

    def _initialize_order_book_model(self):
        """Initialize order book prediction model"""
        try:
            # Dynamically select and combine the latest bids files
            data_dir = Path("data")
            bid_files = sorted(data_dir.glob(f"bids_{self.symbol}_*.csv"), key=os.path.getmtime, reverse=True)
            if not bid_files:
                data_path = f"data/bids_{self.symbol}_sample.csv"
                logger.warning(f"No bid files found. Falling back to {data_path}")
            else:
                # Combine up to 10 latest files or until sufficient data
                combined_data = pd.DataFrame()
                for file in bid_files[:10]:
                    df = pd.read_csv(file)
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
                    if len(combined_data) > 20:
                        break
                data_path = data_dir / f"combined_bids_{self.symbol}.csv"
                combined_data.to_csv(data_path, index=False)
                logger.info(f"Combined {len(combined_data)} rows from {len(bid_files[:10])} files into {data_path}")

            logger.info(f"Using data file: {data_path}")
            self.order_book_model = OrderBookLSTMModel(
                model_path="models/order_book_predictor.h5",
                data_path=str(data_path)
            )
            if not self.order_book_model.load_model():
                logger.info("Training new order book prediction model...")
                self.order_book_model.train()
            logger.info("Order book LSTM model initialized")
        except Exception as e:
            logger.error(f"Order book model initialization failed: {str(e)}")
            self.order_book_model = None

    def _get_initial_balance(self) -> float:
        """Fetch and validate initial balance"""
        balance = self.client.get_balance()
        if not isinstance(balance, float) or balance <= 0:
            raise ValueError(f"Invalid initial balance: {balance}")
        logger.info(f"Initial balance: ${balance:,.2f}")
        return balance

    def _initialize_risk_management(self) -> Dict:
        """Initialize all risk management components"""
        return {
            'leverage': LeverageControl(self.client),
            'drawdown': MaxDrawdown(self.client, self.initial_balance, max_drawdown=0.2),
            'max_loss': MaxLossPerTrade(self.client, self.initial_balance),
            'position_sizing': PositionSizing(self.client, self.initial_balance),
            'risk_manager': RiskManager(self.client, symbol=self.symbol, max_loss=0.02, volatility_threshold=0.5),
            'stop_loss': StopLossTakeProfit(self.client),
            'trailing_stop': TrailingStopLoss(self.client)
        }

    def _initialize_analysis_tools(self) -> Dict:
        """Initialize market analysis components"""
        return {
            'iceberg_detector': IcebergDetector(self.client, self.symbol),
            'market_insights': MarketInsights(self.client, [self.symbol]),
            'ofi_analyzer': OFIAnalysis(self.client, self.symbol),
            'order_book_analyzer': OrderBookAnalysis(self.client, self.symbol),
            'order_timing': OrderTimingOptimizer(self.client, self.symbol),
            'stop_hunt_detector': StopHuntDetector(self.client, self.symbol),
            'market_insights_1h': MarketInsights(self.client, [self.symbol], timeframe='1h'),
        }

    def _initialize_tracking_components(self) -> Dict:
        """Initialize tracking components for profit and strategy reporting"""
        return {
            'profit_tracker': ProfitTracker(self.client, self.symbol),
            'strategy_report': StrategyReport(self.client)
        }

    def _initialize_learning_components(self) -> Dict:
        """Initialize self-learning components"""
        return {
            'self_learning': SelfLearning(self.client, model_path="models/trading_model_advanced.keras", sequence_length=20)
        }

    def _initialize_trading_strategy(self) -> AdvancedTradingStrategy:
        """Initialize and configure advanced trading strategy"""
        return AdvancedTradingStrategy(
            client=self.client,
            symbol=self.symbol,
            N=10,
            initial_threshold=0.2,
            interval=10,
            lookback_period=20,
            volatility_window=10,
            risk_per_trade=0.01,
            stop_loss_factor=0.02,
            take_profit_factor=0.04,
            lstm_sequence_length=60
        )

    def _update_position_info(self):
        """Fetch and update current position information"""
        try:
            position = self.client.get_positions(self.symbol)
            self.position_info.update({
                'size': float(position[0].get('size', 0)) if position else 0.0,
                'side': position[0].get('side') if position else None,
                'entry_price': float(position[0].get('entryPrice', 0)) if position else 0.0,
                'timestamp': datetime.now()
            })
            self._sync_strategy_positions()
        except Exception as e:
            logger.error(f"Failed to update position info: {str(e)}")

    def _sync_strategy_positions(self):
        """Synchronize position across all strategies"""
        for strategy in self.execution_strategies.values():
            if hasattr(strategy, 'position_info'):
                strategy.position_info = self.position_info

    def _execute_safety_checks(self) -> bool:
        """Perform all risk management checks"""
        try:
            current_balance = self.client.get_balance()
            if not self.risk_components['drawdown'].check_drawdown(current_balance):
                logger.warning("Max drawdown limit breached!")
                return False

            if not self.risk_components['leverage'].check_and_set_leverage(self.symbol):
                logger.warning("Leverage adjustment failed!")
                return False

            if self.risk_components['risk_manager'].check_volatility():
                logger.info("High volatility detected - reducing position size")
                self.trading_strategy.risk_per_trade *= 0.5

            return True
        except Exception as e:
            logger.error(f"Safety check failed: {str(e)}")
            return False

    def _get_order_book_predictions(self) -> Optional[float]:
        """Get price prediction from order book LSTM model"""
        if not self.order_book_model:
            return None

        try:
            order_book_data = self.client.get_order_book(self.symbol)
            processed_data = self._process_order_book_data(order_book_data)
            prediction = self.order_book_model.predict(processed_data)
            logger.info(f"Order book model prediction: {prediction:.2f}")
            return prediction
        except Exception as e:
            logger.error(f"Order book prediction failed: {str(e)}")
            return None

    def _process_order_book_data(self, order_book: Dict) -> np.ndarray:
        """Process order book data for model input"""
        bids = np.array([[float(p), float(s)] for p, s in order_book['bids']])
        asks = np.array([[float(p), float(s)] for p, s in order_book['asks']])
        return np.concatenate([bids, asks])[:50]

    def _fetch_ohlcv_data(self, limit: int = 100) -> np.ndarray:
        """Fetch recent OHLCV data for self-learning model"""
        try:
            ohlcv = self.client.get_historical_data(self.symbol, interval='1', limit=limit)
            return np.array([[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in ohlcv])
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {str(e)}")
            return np.array([])

    def _analyze_market_conditions(self) -> Dict:
        """Run complete market analysis"""
        analysis_results = {}
        try:
            market_analysis = self.analysis_components['market_insights'].analyze_market() if ENABLE_MARKET_INSIGHTS else {}
            analysis_results['insights'] = market_analysis
            analysis_results.update({
                'ofi': self.analysis_components['ofi_analyzer'].compute_order_flow_imbalance(),
                'icebergs': self.analysis_components['iceberg_detector'].detect_iceberg_orders(),
                'ob_prediction': self._get_order_book_predictions(),
                'insights': self.analysis_components['market_insights'].run() if ENABLE_MARKET_INSIGHTS else {},
                'stop_hunt': self.analysis_components['stop_hunt_detector'].detect_stop_hunts(),
                'order_timing': self.analysis_components['order_timing'].detect_large_orders(),
                'volatility': self.risk_components['risk_manager'].current_volatility
            })
            logger.info(f"Market Analysis Results: {analysis_results}")
            return analysis_results
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")
            return {}

    def _select_execution_strategy(self, analysis: Dict) -> str:
        """Select optimal execution strategy based on market conditions"""
        if analysis.get('stop_hunt'):
            return 'hft'
        if analysis.get('ofi', 0) > 0.3:
            return 'market_making'
        if analysis.get('volatility', 0) > 0.05:
            return 'scalping'
        return 'default'

    def _switch_strategy(self, new_strategy: str):
        """Safely switch between execution strategies"""
        if self.active_strategy and self.active_strategy != new_strategy:
            logger.info(f"Switching from {self.active_strategy} to {new_strategy}")
            if hasattr(self.execution_strategies[self.active_strategy], 'stop'):
                self.execution_strategies[self.active_strategy].stop()

        self.active_strategy = new_strategy
        logger.info(f"Activating {self.active_strategy} strategy")
        if hasattr(self.execution_strategies[self.active_strategy], 'start'):
            self.execution_strategies[self.active_strategy].start()

    def _execute_trading_cycle(self):
        """Complete trading cycle execution"""
        if not self._execute_safety_checks():
            logger.debug("Trading cycle skipped due to safety check failure")
            return

        market_analysis = self._analyze_market_conditions()
        self._update_position_info()

        try:
            selected_strategy = self._select_execution_strategy(market_analysis)
            self._switch_strategy(selected_strategy)

            if selected_strategy == 'default':
                strategy_signal = self.trading_strategy.get_signal()
                ob_prediction = market_analysis.get('ob_prediction', 0)
                ohlcv_data = self._fetch_ohlcv_data(limit=self.learning_components['self_learning'].sequence_length + 1)
                current_price = self.client.get_market_price(self.symbol)
                volatility = market_analysis.get('volatility', 0)
                learning_signal = self.learning_components['self_learning'].predict_action(ohlcv_data, current_price, volatility)
                learning_signal_value = 1 if learning_signal == "BUY" else -1 if learning_signal == "SELL" else 0
                combined_signal = (strategy_signal * 0.5) + (ob_prediction * 0.3) + (learning_signal_value * 0.2)
                self._execute_signal_based_trade(combined_signal, selected_strategy)
            else:
                self._execute_signal_based_trade(0, selected_strategy)

        except Exception as e:
            logger.error(f"Trading cycle failed: {str(e)}")

    def _execute_signal_based_trade(self, signal: float, strategy_name: str):
        """Execute trades for default strategy and log performance"""
        current_price = self.client.get_market_price(self.symbol)
        position_size = min(
            self.risk_components['position_sizing'].calculate_position_size(self.trading_strategy.risk_per_trade),
            self.max_position  # Cap at max position from .env
        )

        try:
            if signal > 0.5 and not self.position_info['size']:
                self._open_position('long', self.trade_size, current_price)  # Use trade_size from .env
            elif signal < -0.5 and not self.position_info['size']:
                self._open_position('short', self.trade_size, current_price)
            elif abs(signal) < 0.2 and self.position_info['size']:
                self._close_position()
            self.tracking_components['strategy_report'].log_trade(strategy_name, self.tracking_components['profit_tracker'].get_current_position() or 0)
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")

    def _open_position(self, direction: str, size: float, price: float):
        """Open new position with risk management and log entry"""
        try:
            self.client.place_order(
                symbol=self.symbol,
                side=direction,
                qty=size,
                order_type="Market"
            )
            self.position_info.update({
                'size': size,
                'side': direction,
                'entry_price': price,
                'timestamp': datetime.now()
            })
            self._sync_strategy_positions()
            self.risk_components['stop_loss'].set_levels(
                self.symbol, price,
                self.trading_strategy.stop_loss_factor,
                self.trading_strategy.take_profit_factor
            )
            self.learning_components['self_learning'].update_trade_state(direction.capitalize(), price, size)
            logger.info(f"Opened {direction} position of {size} @ {price}")
        except Exception as e:
            logger.error(f"Position opening failed: {str(e)}")

    def _close_position(self):
        """Close existing position and log trade"""
        try:
            exit_price = self.client.get_market_price(self.symbol)
            self.client.place_order(
                symbol=self.symbol,
                side='Sell' if self.position_info['side'] == 'long' else 'Buy',
                qty=self.position_info['size'],
                order_type="Market"
            )
            self.tracking_components['profit_tracker'].record_trade(
                entry_price=self.position_info['entry_price'],
                exit_price=exit_price,
                size=self.position_info['size'],
                side=self.position_info['side']
            )
            self.learning_components['self_learning'].clear_trade_state(exit_price)
            logger.info(f"Closed {self.position_info['size']} position")
            self.position_info = {
                'size': 0.0,
                'side': None,
                'entry_price': 0.0,
                'timestamp': None
            }
            self._sync_strategy_positions()
        except Exception as e:
            logger.error(f"Position closing failed: {str(e)}")

    def _retrain_models(self):
        """Retrain prediction models periodically"""
        try:
            logger.info("Starting model retraining...")
            if self.order_book_model:
                order_book = self.client.get_order_book(self.symbol)
                bids = pd.DataFrame(order_book['bids'], columns=['Price', 'Size'])
                asks = pd.DataFrame(order_book['asks'], columns=['Price', 'Size'])
                new_data = pd.concat([bids, asks], ignore_index=True)
                self.order_book_model.update_data(new_data)
                self.order_book_model.train()
            self.trading_strategy.retrain_model()
            ohlcv_data = self._fetch_ohlcv_data(limit=1000)
            if ohlcv_data.size > 0:
                self.learning_components['self_learning'].train(ohlcv_data)
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Model retraining failed: {str(e)}")

    def _capture_output(self, func):
        """Capture the output of a function that prints to stdout"""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            func()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output

    def _save_report_to_file(self, report_content: str, report_type: str):
        """Save report content to a file with a timestamped name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_report_{timestamp}.txt"
        filepath = self.report_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Saved {report_type} report to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save {report_type} report: {str(e)}")
            return None

    def _send_email(self, profit_report: str, strategy_report: str, learning_report: str, profit_report_file: str, strategy_report_file: str, learning_report_file: str):
        """Send reports via email with files attached"""
        recipient = "collins4oloo@gmail.com"
        subject = f"Trading System Reports - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = recipient
        msg['Subject'] = subject

        body = "Please find the attached trading system reports.\n\nProfit Report:\n" + profit_report + "\n\nStrategy Report:\n" + strategy_report + "\n\nSelf-Learning Report:\n" + learning_report
        msg.attach(MIMEText(body, 'plain'))

        for filepath in [profit_report_file, strategy_report_file, learning_report_file]:
            if filepath and os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(filepath))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filepath)}"'
                    msg.attach(part)

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_USER, recipient, msg.as_string())
            logger.info(f"Sent reports via email to {recipient}")
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")

    def _generate_self_learning_report(self):
        """Generate a report for self-learning performance metrics"""
        metrics = self.learning_components['self_learning'].get_performance_metrics()
        report = f"📊 **Self-Learning Performance Report** 📊\n"
        report += f"🔹 Total Trades: {metrics['total_trades']}\n"
        report += f"🔹 Wins: {metrics['win_count']}\n"
        report += f"🔹 Losses: {metrics['loss_count']}\n"
        report += f"🔹 Win Rate: {metrics['win_rate']:.2%}\n"
        return report

    def _generate_reports(self):
        """Generate profit, strategy, and self-learning reports, save to files, and send via email"""
        try:
            logger.info("Generating trading reports...")
            profit_report_file = strategy_report_file = learning_report_file = None

            if ENABLE_PROFIT_TRACKER:
                profit_report_content = self._capture_output(self.tracking_components['profit_tracker'].generate_report)
                profit_report_file = self._save_report_to_file(profit_report_content, "profit")
            else:
                profit_report_content = "Profit tracking disabled"

            if ENABLE_STRATEGY_REPORT:
                strategy_report_content = self._capture_output(self.tracking_components['strategy_report'].generate_strategy_report)
                strategy_report_file = self._save_report_to_file(strategy_report_content, "strategy")
            else:
                strategy_report_content = "Strategy reporting disabled"

            learning_report_content = self._generate_self_learning_report()
            learning_report_file = self._save_report_to_file(learning_report_content, "self_learning")

            if EMAIL_USER and EMAIL_PASSWORD:
                self._send_email(profit_report_content, strategy_report_content, learning_report_content, profit_report_file, strategy_report_file, learning_report_file)
            else:
                logger.warning("Email credentials not set. Skipping email sending.")

            logger.info("Reports generated successfully")
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")

    def run(self):
        """Main trading loop with strategy management and reporting"""
        logger.info("Starting trading system...")
        try:
            while self.running:
                start_time = time.time()
                logger.debug("Starting trading cycle")
                self._execute_trading_cycle()

                if datetime.now().hour % 6 == 0:
                    logger.debug("Triggering periodic retraining and reporting")
                    self._retrain_models()
                    self._generate_reports()

                elapsed = time.time() - start_time
                sleep_time = max(60 - elapsed, 5)
                logger.debug(f"Cycle completed in {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Shutting down via KeyboardInterrupt...")
        except Exception as e:
            logger.critical(f"Unexpected error in run loop: {str(e)}", exc_info=True)
        finally:
            self.running = False
            if self.active_strategy and hasattr(self.execution_strategies[self.active_strategy], 'stop'):
                self.execution_strategies[self.active_strategy].stop()
            if self.strategy_thread and self.strategy_thread.is_alive():
                self.strategy_thread.join()
            logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    try:
        trading_system = TradingSystem()
        trading_system.run()
    except Exception as e:
        logger.critical(f"System startup failed: {str(e)}")
