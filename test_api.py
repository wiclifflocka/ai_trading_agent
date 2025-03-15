# test_api.py
import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
testnet = os.getenv("USE_TESTNET", "True").lower() == "true"

client = ccxt.bybit({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'testnet': testnet,
})

try:
    balance = client.fetch_balance()
    print("Balance:", balance['total'])
except Exception as e:
    print(f"Error: {e}")
