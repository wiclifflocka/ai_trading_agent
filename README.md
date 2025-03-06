AI Trading Agent for Bybit
This project creates a sophisticated AI agent designed to trade profitably on Bybit, using advanced market analysis strategies. The agent reads the order book, applies scalping and HFT (High-Frequency Trading) strategies, tracks profits, and generates performance reports.


📂 Table of Contents

Features
Installation
Environment Setup
How to Use
Deployment
Configuration
Testing
License


🔧 Features

Real-time Order Book Streaming: The AI agent continuously monitors the Bybit order book.
Scalping & HFT Strategies: Trades based on advanced, high-frequency strategies to maximize profits.
Profit Tracking: Tracks both unrealized and realized P&L (Profit & Loss) in real-time.
Strategy Performance Reports: Provides detailed reports on strategy performance, including win rate, total P&L, and other metrics.
Market Insights: Provides analysis of market conditions such as order book imbalance, aggression, and potential iceberg orders.
Real-time Alerts: Sends Telegram alerts for significant events like trade executions, P&L updates, and more.
Self-Learning: Continuously improves strategy performance based on historical trades.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/yourusername/bybit-ai-trading-agent.git
cd bybit-ai-trading-agent


2. Install Python & Dependencies
Ensure that you have Python 3.x installed. If not, install it first.

Install required Python dependencies:
pip install -r requirements.txt

🌍 Environment Setup
1. Setup API Keys
Create a .env file to store your Bybit API keys and other configuration settings.
cp .env.example .env
nano .env

Inside the .env file, add your Bybit API key and secret:
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
ENABLE_PROFIT_TRACKER=True
ENABLE_STRATEGY_REPORT=True
ENABLE_MARKET_INSIGHTS=True

Make sure to replace the placeholders with your real Bybit API credentials.

🛠️ How to Use
1. Start the AI Trading Agent
Run the agent in a screen session to ensure it runs continuously even when disconnected from the server.
screen -S trading_bot
python main.py

To detach from the screen session, press CTRL + A, then D. To reattach:
screen -r trading_bot

2. Monitor Strategy & Profit
Use the following scripts to monitor the agent's performance:

Profit Tracker: Tracks real-time P&L.
python tracking/profit_tracker.py


Strategy Report: Displays performance of each strategy.
python tracking/strategy_report.py


Market Insights: Analyzes market conditions based on the order book.
python analysis/market_insights.py

🚀 Deployment
1. Set Up Server
For production, deploy the AI trading agent on a VPS (e.g., DigitalOcean, AWS, Linode).
Ensure that your server has Python 3 installed and the environment is set up.

2. Run with Cron Jobs
To continuously track profits and market insights, set up cron jobs to run the scripts automatically:
crontab -e

Add the following cron jobs to run every 5 minutes:
*/5 * * * * /usr/bin/python3 /path/to/tracking/profit_tracker.py
*/5 * * * * /usr/bin/python3 /path/to/tracking/strategy_report.py
*/5 * * * * /usr/bin/python3 /path/to/analysis/market_insights.py

🔧 Configuration
You can modify key parameters in the .env file to adjust the agent’s behavior:

BYBIT_API_KEY: Your Bybit API Key.
BYBIT_API_SECRET: Your Bybit API Secret.
ENABLE_PROFIT_TRACKER: Set to True to enable real-time profit tracking.
ENABLE_STRATEGY_REPORT: Set to True to generate strategy performance reports.
ENABLE_MARKET_INSIGHTS: Set to True to enable market condition analysis.

🧪 Testing
To ensure the code is working, you can run unit tests for core functionalities:

bash
pytest tests/
This will check the logic of the trading agent, profit tracker, and other modules.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
