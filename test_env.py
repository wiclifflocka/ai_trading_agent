# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
use_testnet = os.getenv("USE_TESTNET")

print(f"API_KEY: {api_key} ({len(api_key) if api_key else 'None'} chars)")
print(f"API_SECRET: {api_secret} ({len(api_secret) if api_secret else 'None'} chars)")
print(f"USE_TESTNET: {use_testnet}")
