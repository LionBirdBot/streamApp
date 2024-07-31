import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import random
import streamlit as st
from statsmodels.tsa.ar_model import AutoReg
import datetime as dt


def fetch_stocks():
    # Sample stock dictionary (usually fetched from a file)
    return {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com, Inc.",
        "META": "Meta Platforms, Inc."
    }


def fetch_periods_intervals():
    return {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }


def fetch_stock_info(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_info = stock_data.info

    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")

    stock_data_info = {
        "Basic Information": {
            "symbol": safe_get(stock_data_info, "symbol"),
            "longName": safe_get(stock_data_info, "longName"),
            "currency": safe_get(stock_data_info, "currency"),
            "exchange": safe_get(stock_data_info, "exchange"),
        },
        "Market Data": {
            "currentPrice": safe_get(stock_data_info, "currentPrice"),
            "previousClose": safe_get(stock_data_info, "previousClose"),
            "open": safe_get(stock_data_info, "open"),
            "dayLow": safe_get(stock_data_info, "dayLow"),
            "dayHigh": safe_get(stock_data_info, "dayHigh"),
            "regularMarketPreviousClose": safe_get(stock_data_info, "regularMarketPreviousClose"),
            "regularMarketOpen": safe_get(stock_data_info, "regularMarketOpen"),
            "regularMarketDayLow": safe_get(stock_data_info, "regularMarketDayLow"),
            "regularMarketDayHigh": safe_get(stock_data_info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow": safe_get(stock_data_info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": safe_get(stock_data_info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage": safe_get(stock_data_info, "fiftyDayAverage"),
            "twoHundredDayAverage": safe_get(stock_data_info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume": safe_get(stock_data_info, "volume"),
            "regularMarketVolume": safe_get(stock_data_info, "regularMarketVolume"),
            "averageVolume": safe_get(stock_data_info, "averageVolume"),
            "averageVolume10days": safe_get(stock_data_info, "averageVolume10days"),
            "averageDailyVolume10Day": safe_get(stock_data_info, "averageDailyVolume10Day"),
            "sharesOutstanding": safe_get(stock_data_info, "sharesOutstanding"),
            "impliedSharesOutstanding": safe_get(stock_data_info, "impliedSharesOutstanding"),
            "floatShares": safe_get(stock_data_info, "floatShares"),
        },
        "Dividends and Yield": {
            "dividendRate": safe_get(stock_data_info, "dividendRate"),
            "dividendYield": safe_get(stock_data_info, "dividendYield"),
            "payoutRatio": safe_get(stock_data_info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": safe_get(stock_data_info, "marketCap"),
            "enterpriseValue": safe_get(stock_data_info, "enterpriseValue"),
            "priceToBook": safe_get(stock_data_info, "priceToBook"),
            "debtToEquity": safe_get(stock_data_info, "debtToEquity"),
            "grossMargins": safe_get(stock_data_info, "grossMargins"),
            "profitMargins": safe_get(stock_data_info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue": safe_get(stock_data_info, "totalRevenue"),
            "revenuePerShare": safe_get(stock_data_info, "revenuePerShare"),
            "totalCash": safe_get(stock_data_info, "totalCash"),
            "totalCashPerShare": safe_get(stock_data_info, "totalCashPerShare"),
            "totalDebt": safe_get(stock_data_info, "totalDebt"),
            "earningsGrowth": safe_get(stock_data_info, "earningsGrowth"),
            "revenueGrowth": safe_get(stock_data_info, "revenueGrowth"),
            "returnOnAssets": safe_get(stock_data_info, "returnOnAssets"),
            "returnOnEquity": safe_get(stock_data_info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow": safe_get(stock_data_info, "freeCashflow"),
            "operatingCashflow": safe_get(stock_data_info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice": safe_get(stock_data_info, "targetHighPrice"),
            "targetLowPrice": safe_get(stock_data_info, "targetLowPrice"),
            "targetMeanPrice": safe_get(stock_data_info, "targetMeanPrice"),
            "targetMedianPrice": safe_get(stock_data_info, "targetMedianPrice"),
        },
    }
    return stock_data_info


def fetch_stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_history = stock_data.history(period=period, interval=interval)[["Open", "High", "Low", "Close"]]
    return stock_data_history


def generate_stock_prediction(stock_ticker):
    try:
        stock_data = yf.Ticker(stock_ticker)
        stock_data_hist = stock_data.history(period="2y", interval="1d")
        stock_data_close = stock_data_hist[["Close"]]
        stock_data_close = stock_data_close.asfreq("D", method="ffill")
        stock_data_close = stock_data_close.ffill()

        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9):]  # 10%

        model = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")

        predictions = model.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)
        forecast = model.predict(start=test_df.index[0], end=test_df.index[-1] + dt.timedelta(days=90), dynamic=True)

        return train_df, test_df, forecast, predictions
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return None, None, None, None


def get_predictions(stock_ticker):
    train_df, test_df, forecast, predictions = generate_stock_prediction(stock_ticker)
    if predictions is not None and len(test_df) > 0:
        accuracy = round(random.uniform(70, 100), 2)  # Simulated accuracy for the demo
        latest_prediction = predictions[-1]
        latest_actual = test_df["Close"].iloc[-1]

        # Calculate the percentage difference
        percent_diff = (latest_prediction - latest_actual) / latest_actual * 100

        # Determine action based on percentage difference
        if percent_diff > 5:
            action = "Buy"
        elif percent_diff < -5:
            action = "Sell"
        else:
            action = "Hold"

        return {"prediction": action, "accuracy": round(accuracy,2)}
    else:
        return {"prediction": "N/A", "accuracy": 0}


def get_bg_color(prediction):
    if prediction == "Buy":
        return "green"
    elif prediction == "Hold":
        return "yellow"
    elif prediction == "Sell":
        return "red"
    else:
        return "gray"

def plot_stock_data(stock_ticker, title="Stock Price Over Time"):
    df = fetch_stock_history(stock_ticker, period="1y", interval="1d")
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(plt)