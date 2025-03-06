import numpy as np
import pandas as pd
import tensorflow as tf
from data_pipeline.bybit_api import BybitAPI
from models.order_book_lstm import create_sequences

# Load trained model
model = tf.keras.models.load_model("models/order_book_predictor.h5")

# Initialize API
api = BybitAPI()

def fetch_live_order_book(symbol="BTCUSDT", seq_length=10):
    """
    Fetches live order book data and formats it for model prediction.
    :param symbol: Trading pair (e.g., "BTCUSDT")
    :param seq_length: Number of past order book snapshots to use for prediction
    :return: Prepared input for LSTM model
    """
    data = api.get_order_book(symbol)
    
    if not data:
        return None
    
    bids = sorted(data['bids'], key=lambda x: x[0], reverse=True)[:5]  # Top 5 bid levels
    asks = sorted(data['asks'], key=lambda x: x[0])[:5]  # Top 5 ask levels
    
    # Combine bids and asks into a single list
    order_book_data = bids + asks  # Merging top bids & asks
    df = pd.DataFrame(order_book_data, columns=["Price", "Size"])
    
    # Convert to numpy array
    data_array = df[["Price", "Size"]].values
    
    # Create input sequence for LSTM model
    X_input, _ = create_sequences(data_array, seq_length)
    
    return X_input[-1].reshape(1, seq_length, 2)  # Latest sequence for prediction

# Run real-time predictions
def predict_next_price(symbol="BTCUSDT"):
    """
    Predicts the next price movement using live order book data.
    """
    X_input = fetch_live_order_book(symbol)
    
    if X_input is not None:
        predicted_price = model.predict(X_input)
        print(f"Predicted Next Price for {symbol}: {predicted_price[0][0]:.2f}")
    else:
        print("Error fetching order book data.")

# Example usage
if __name__ == "__main__":
    while True:
        predict_next_price()
        import time
        time.sleep(2)  # Predict every 2 seconds

