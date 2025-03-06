import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load historical order book data
def load_data(file_path):
    """
    Loads historical order book data for training.
    :param file_path: CSV file path
    :return: Processed numpy arrays for training
    """
    df = pd.read_csv(file_path)
    data = df[["Price", "Size"]].values  # Selecting price and volume
    return np.array(data)

# Prepare training dataset
def create_sequences(data, seq_length=10):
    """
    Creates sequences for LSTM model training.
    :param data: Input data array
    :param seq_length: Number of past order book snapshots to use for prediction
    :return: X (features), Y (targets)
    """
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length, 0])  # Predict next price
    return np.array(X), np.array(Y)

# Load and preprocess data
data = load_data("data/bids_BTCUSDT_sample.csv")
X, Y = create_sequences(data)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)  # Predict next price
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, Y, epochs=10, batch_size=32)

# Save the trained model
model.save("models/order_book_predictor.h5")
print("Model trained and saved!")

