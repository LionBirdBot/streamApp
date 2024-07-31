import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def fetch_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df[['Close']]


def prepare_data(data, look_back=60):
    data['Prediction'] = data['Close'].shift(-look_back)
    X = np.array(data.drop('Prediction', axis=1))[:-look_back]
    y = np.array(data['Prediction'])[:-look_back]
    return X, y


def train_random_forest_model(ticker):
    data = fetch_stock_data(ticker)
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, data['Close'].values[-60:]


def predict_random_forest(model, last_60_days):
    prediction = model.predict(last_60_days.reshape(1, -1))
    return prediction[0]


def get_random_forest_prediction(ticker):
    model, X_test, y_test, last_60_days = train_random_forest_model(ticker)
    prediction = predict_random_forest(model, last_60_days)

    actual_price = y_test[-1]
    percent_change = ((prediction - actual_price) / actual_price) * 100

    if percent_change > 5:
        action = "Buy"
    elif percent_change < -5:
        action = "Sell"
    else:
        action = "Hold"

    accuracy = model.score(X_test, y_test) * 100  # R-squared score

    return {"prediction": action, "accuracy": round(accuracy, 2)}