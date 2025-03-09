AI Trading Agent for Bybit

This project is an AI-powered trading agent designed to automate cryptocurrency trading on the Bybit platform. It leverages real-time market data, risk management techniques, and AI-driven decision-making to execute trades efficiently. The agent is modular, allowing for easy customization and extension of its features.

Table of Contents

Project Overview (#project-overview)

Key Features (#key-features)

Current Setup (#current-setup)

Getting Started (#getting-started)
Prerequisites (#prerequisites)

Installation (#installation)

Configuration (#configuration)

Running the Agent (#running-the-agent)

Usage (#usage)

Troubleshooting (#troubleshooting)

Future Enhancements (#future-enhancements)

Contributing (#contributing)

License (#license)

Contact (#contact)


Project Overview

The AI Trading Agent automates cryptocurrency trading on Bybit by integrating modules for market analysis, risk management, trade execution, and performance tracking. It is designed to help traders make data-driven decisions while minimizing risk and maximizing profitability.

Key Features
Market Data Analysis:
Real-time order book analysis and calculation of Order Flow Imbalance (OFI).

Market insights for multiple symbols (e.g., BTCUSDT, ETHUSDT, XRPUSDT).

Advanced detection tools (under development) for iceberg orders and stop hunts.

Risk Management:
Dynamic position sizing based on account balance and risk percentage.

Automatic stop-loss, take-profit, and trailing stop-loss settings.

Leverage control and max loss per trade to limit exposure.

Max drawdown monitoring to halt trading if losses exceed a threshold.

Trade Execution:
Supports multiple strategies: High-Frequency Trading (HFT), Market Making, and Scalping.

AI-driven decision-making to predict actions like "BUY," "SELL," or "HOLD."

Self-Learning and Adaptation:
Trains an AI model using historical trade data to improve decision-making over time.

Performance Tracking and Reporting:
Real-time profit tracking and strategy performance reports.

Current Setup
The agent is configured to run on Bybit's testnet, allowing for simulated trading without risking real funds.

Some features, such as the AI learning component and advanced detection tools, are still under development.


Getting Started
Prerequisites
Python 3.x

Bybit Testnet Account: Sign up at Bybit Testnet and generate API keys.

Dependencies:
pybit: Bybit's official Python SDK.

Other libraries as required (e.g., numpy, pandas, keras for AI components).

Installation
Clone the repository:
bash

git clone https://github.com/yourusername/ai_trading_agent.git
cd ai_trading_agent

Install the required dependencies:
bash

pip install pybit tensorflow numpy



Configuration
Update the API_KEY and API_SECRET in main.py with your Bybit testnet API credentials:
python

API_KEY = 'your_testnet_api_key'
API_SECRET = 'your_testnet_api_secret'

Ensure the SYMBOL variable is set to the desired trading pair (e.g., 'BTCUSDT').

Running the Agent
Execute the main script:
bash

python main.py

The agent will start analyzing market conditions, executing trades, and generating reports in a loop.

Usage
The agent runs continuously, checking market conditions every 10 seconds.

It analyzes the order book, calculates OFI, and executes trades based on its strategies and risk management rules.

Logs are generated to track the agent's activities, including balance checks, trade executions, and any errors.

Troubleshooting
No Balance in Testnet Account:
If you see Fetched current balance: 0.0 USD, fund your Bybit testnet account with USDT via the testnet platform.

Error Fetching Recent Trades:
Ensure the get_recent_trades method in bybit_api.py uses the correct pybit method (get_public_trading_records).

Max Drawdown Exceeded:
This warning appears if the account balance is too low. Ensure your testnet account has sufficient funds.

Other API Errors:
Check the logs for specific error messages and verify your API keys and permissions.

Future Enhancements
Fully implement the AI learning component for improved decision-making.

Integrate advanced detection tools for iceberg orders and stop hunts.

Switch to Bybit's mainnet for live trading with real funds.

Customize risk parameters, trading strategies, and symbols for optimal performance.

Contributing
Contributions are welcome! To contribute:
Fork the repository.

Create a new branch for your feature or bugfix.

Follow the project's coding standards.

Submit a pull request for review.

License
This project is licensed under the MIT License (LICENSE).
Contact
For questions or support, please contact [collaustine27@gmail.com (mailto:collins4oloo@gmail.com)].


