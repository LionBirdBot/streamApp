import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def fetch_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df[['Close']]


def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def create_model(look_back):
    model = tf.keras.modelsSequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_lstm_model(ticker):
    data = fetch_stock_data(ticker)
    X, y, scaler = prepare_data(data)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = create_model(X.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    return model, X_test, y_test, scaler


def predict_lstm(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def get_lstm_prediction(ticker):
    model, X_test, y_test, scaler = train_lstm_model(ticker)
    last_60_days = X_test[-1].reshape(1, 60, 1)
    prediction = predict_lstm(model, last_60_days, scaler)[0][0]

    actual_price = scaler.inverse_transform(y_test[-1].reshape(1, -1))[0][0]
    percent_change = ((prediction - actual_price) / actual_price) * 100

    if percent_change > 5:
        action = "Buy"
    elif percent_change < -5:
        action = "Sell"
    else:
        action = "Hold"

    accuracy = 100 - abs(percent_change)  # A simple accuracy measure
    accuracy = max(0, min(accuracy, 100))  # Ensure accuracy is between 0 and 100

    return {"prediction": action, "accuracy": round(accuracy, 2)}