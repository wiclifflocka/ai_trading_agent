# bybit_client.py

import requests
import json
import time
import hashlib
import hmac
from urllib.parse import urlencode


class BybitClient:
    def __init__(self, api_key, api_secret, base_url="https://api.bybit.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    # Function to create the signature for the request
    def _create_signature(self, params):
        query_string = urlencode(sorted(params.items()))
        return hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    # Function to make an authenticated GET request
    def _send_get_request(self, endpoint, params=None):
        if params is None:
            params = {}

        params['api_key'] = self.api_key
        params['timestamp'] = str(int(time.time() * 1000))
        params['sign'] = self._create_signature(params)

        response = requests.get(self.base_url + endpoint, params=params)
        return response.json()

    # Function to make an authenticated POST request
    def _send_post_request(self, endpoint, params=None):
        if params is None:
            params = {}

        params['api_key'] = self.api_key
        params['timestamp'] = str(int(time.time() * 1000))
        params['sign'] = self._create_signature(params)

        response = requests.post(self.base_url + endpoint, data=params)
        return response.json()

    # Function to retrieve account balance
    def get_balance(self):
        endpoint = "/v2/private/wallet/balance"
        params = {}
        response = self._send_get_request(endpoint, params)
        if response.get('ret_code') == 0:
            balance_data = response.get('result', {}).get('total', {})
            return float(balance_data.get('equity', 0))
        else:
            raise Exception(f"Error retrieving balance: {response.get('ret_msg')}")

    # Function to place a limit order
    def place_order(self, symbol, qty, side, order_type="Limit", price=None, time_in_force="GoodTillCancel"):
        endpoint = "/v2/private/order/create"
        params = {
            "symbol": symbol,
            "order_type": order_type,
            "qty": qty,
            "side": side,
            "time_in_force": time_in_force
        }

        if price:
            params["price"] = price
        
        response = self._send_post_request(endpoint, params)
        return response

    # Function to get market price
    def get_market_price(self, symbol):
        endpoint = f"/v2/public/tickers"
        params = {"symbol": symbol}
        response = self._send_get_request(endpoint, params)
        if response.get('ret_code') == 0:
            return float(response.get('result', [{}])[0].get('last_price', 0))
        else:
            raise Exception(f"Error retrieving market price: {response.get('ret_msg')}")

    # Function to get order book data
    def get_order_book(self, symbol):
        endpoint = f"/v2/public/orderBook/L2"
        params = {"symbol": symbol}
        response = self._send_get_request(endpoint, params)
        if response.get('ret_code') == 0:
            return response.get('result', [])
        else:
            raise Exception(f"Error retrieving order book: {response.get('ret_msg')}")

    # Function to get market data (candlestick data)
    def get_market_data(self, symbol, interval="1", limit=200):
        endpoint = "/v2/public/kline/list"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = self._send_get_request(endpoint, params)
        if response.get('ret_code') == 0:
            return response.get('result', [])
        else:
            raise Exception(f"Error retrieving market data: {response.get('ret_msg')}")

    # Function to close a position
    def close_position(self, symbol):
        endpoint = "/v2/private/position/close"
        params = {
            "symbol": symbol
        }
        response = self._send_post_request(endpoint, params)
        return response

