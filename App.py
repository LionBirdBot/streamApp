import streamlit as st
import time
import random
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from AutoregModel import fetch_stocks, fetch_periods_intervals, fetch_stock_info, fetch_stock_history, \
    generate_stock_prediction, get_predictions, plot_stock_data, get_bg_color
from LSTMModel import get_lstm_prediction
from RandomForestModel import get_random_forest_prediction

# Set page configuration
st.set_page_config(layout="wide")

# App title
st.title("Trading App")

# Sidebar configuration
st.sidebar.header("Navigation")
pages = ["Home", "About", "Statistics"]
model_options = ["AUTOREG", "LSTM", "RANDOM FOREST"]

# Sidebar page selection
st.sidebar.header("PAGES")
select_page = st.sidebar.selectbox("Select page", pages)

# Sidebar model selection
st.sidebar.header("MODELS")
select_model = st.sidebar.selectbox("Select model", model_options)

# Page content
def get_autoreg_predictions(stock_ticker):
    pass


if select_page == "Home":
    st.title("Home")
    st.markdown("""
    We provide quick predictions for your favorite stocks. These predictions are updated every minute.
    """)

    stock_dict = fetch_stocks()

    for stock in stock_dict.keys():
        st.markdown(f"<h3>{stock}</h3>", unsafe_allow_html=True)
        if select_model == "AUTOREG":
            prediction_data = get_autoreg_predictions(stock)
        elif select_model == "LSTM":
            prediction_data = get_lstm_prediction(stock)
        else:  # RANDOM FOREST
            prediction_data = get_random_forest_prediction(stock)

        color = get_bg_color(prediction_data["prediction"])
        accuracy = prediction_data["accuracy"]
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <span style="font-size: 16px;">{stock}</span>
            <div style="background-color: {color}; padding: 5px; border-radius: 5px; width: 80px; text-align: center; display: inline-block; margin-right: 10px;">
                <p style="color: white; margin: 0; font-size: 12px;">{prediction_data["prediction"]}</p>
            </div>
            <span style="font-size: 12px;">Accuracy: {accuracy}%</span>
        </div>
        """, unsafe_allow_html=True)

    # Refresh the page every minute
    time.sleep(60)
    st.experimental_rerun()

elif select_page == "About":
    st.title("About")
    if select_model == "AUTOREG":
        st.markdown("""
            ### About AUTOREG
            Autoregression (AR) is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, etc. In this application, we use an AutoReg model to predict stock prices and provide trading recommendations.
            """)

        # Integrate stock prediction functionality
        st.write("Welcome to the Stock Prediction App. Please select a stock to view its information and predictions.")

        # Fetch the stocks
        stock_dict = fetch_stocks()

        # Fetch the periods and intervals
        periods_intervals = fetch_periods_intervals()

        # Create sidebar options for stock selection
        stock_ticker = st.sidebar.selectbox("Select a stock", list(stock_dict.keys()))
        period = st.sidebar.selectbox("Select period", list(periods_intervals.keys()))
        interval = st.sidebar.selectbox("Select interval", periods_intervals[period])

        # Fetch stock info and display
        st.header(f"Stock Information for {stock_ticker}")
        stock_info = fetch_stock_info(stock_ticker)
        for section, info in stock_info.items():
            st.subheader(section)
            for key, value in info.items():
                st.write(f"**{key}**: {value}")

        # Fetch stock history and display
        st.header(f"Stock History for {stock_ticker}")
        stock_history = fetch_stock_history(stock_ticker, period, interval)
        st.write(stock_history.tail())

        # Generate stock prediction and display
        st.header(f"Stock Prediction for {stock_ticker}")
        train_df, test_df, forecast, predictions = generate_stock_prediction(stock_ticker)
        if train_df is not None:
            st.write("Train Data")
            st.line_chart(train_df)
            st.write("Test Data")
            st.line_chart(test_df)
            st.write("Forecast")
            st.line_chart(forecast)
            st.write("Predictions")
            st.line_chart(predictions)

        # Display stock chart
        st.header(f"{stock_ticker} Stock Chart")
        plot_stock_data(stock_ticker)

        # Display a tip based on the model's prediction
        st.header(f"Trading Tip for {stock_ticker}")
        prediction_data = get_predictions(stock_ticker)
        st.markdown(f"### Tip: {prediction_data['prediction']}")
        st.markdown(f"**Model Accuracy**: {prediction_data['accuracy']}%")

        # Explain the prediction
        st.markdown("""
            ### Understanding the Prediction
            - **Buy**: The model predicts the stock price will increase by more than 5% in the near future.
            - **Sell**: The model predicts the stock price will decrease by more than 5% in the near future.
            - **Hold**: The model predicts the stock price will remain within ±5% of its current value in the near future.

            Please note that these predictions are based on historical data and should not be the sole basis for making investment decisions. Always consult with a financial advisor before making investment choices.
            """)

    elif select_model == "LSTM":
        st.markdown("""
        ### About LSTM
        Long Short-Term Memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. It is particularly well-suited for time series prediction tasks like stock price forecasting.
        """)

        # Integrate stock prediction functionality
        st.write("Welcome to the Stock Prediction App. Please select a stock to view its information and predictions.")

        # Fetch the stocks
        stock_dict = fetch_stocks()

        # Create sidebar options for stock selection
        stock_ticker = st.sidebar.selectbox("Select a stock", list(stock_dict.keys()))

        # Fetch stock info and display
        st.header(f"Stock Information for {stock_ticker}")
        stock_info = fetch_stock_info(stock_ticker)
        for section, info in stock_info.items():
            st.subheader(section)
            for key, value in info.items():
                st.write(f"**{key}**: {value}")

        # Display stock chart
        st.header(f"{stock_ticker} Stock Chart")
        plot_stock_data(stock_ticker)

        # Display a tip based on the model's prediction
        st.header(f"Trading Tip for {stock_ticker}")
        prediction_data = get_lstm_prediction(stock_ticker)
        st.markdown(f"### Tip: {prediction_data['prediction']}")
        st.markdown(f"**Model Accuracy**: {prediction_data['accuracy']}%")

        # Explain the prediction
        st.markdown("""
        ### Understanding the Prediction
        - **Buy**: The model predicts the stock price will increase by more than 5% in the near future.
        - **Sell**: The model predicts the stock price will decrease by more than 5% in the near future.
        - **Hold**: The model predicts the stock price will remain within ±5% of its current value in the near future.

        Please note that these predictions are based on historical data and should not be the sole basis for making investment decisions. Always consult with a financial advisor before making investment choices.
        """)

    elif select_model == "RANDOM FOREST":
        st.markdown("""
        ### About Random Forest
        Random Forest is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
        """)

        # Integrate stock prediction functionality
        st.write("Welcome to the Stock Prediction App. Please select a stock to view its information and predictions.")

        # Fetch the stocks
        stock_dict = fetch_stocks()

        # Create sidebar options for stock selection
        stock_ticker = st.sidebar.selectbox("Select a stock", list(stock_dict.keys()))

        # Fetch stock info and display
        st.header(f"Stock Information for {stock_ticker}")
        stock_info = fetch_stock_info(stock_ticker)
        for section, info in stock_info.items():
            st.subheader(section)
            for key, value in info.items():
                st.write(f"**{key}**: {value}")

        # Display stock chart
        st.header(f"{stock_ticker} Stock Chart")
        plot_stock_data(stock_ticker)

        # Display a tip based on the model's prediction
        st.header(f"Trading Tip for {stock_ticker}")
        prediction_data = get_random_forest_prediction(stock_ticker)
        st.markdown(f"### Tip: {prediction_data['prediction']}")
        st.markdown(f"**Model Accuracy**: {prediction_data['accuracy']}%")

        # Explain the prediction
        st.markdown("""
        ### Understanding the Prediction
        - **Buy**: The model predicts the stock price will increase by more than 5% in the near future.
        - **Sell**: The model predicts the stock price will decrease by more than 5% in the near future.
        - **Hold**: The model predicts the stock price will remain within ±5% of its current value in the near future.

        Please note that these predictions are based on historical data and should not be the sole basis for making investment decisions. Always consult with a financial advisor before making investment choices.
        """)

    elif select_page == "Statistics":
        st.title("Model Comparison Statistics")

        # Fetch the stocks
        stock_dict = fetch_stocks()

        # Create sidebar options for stock selection
        stock_ticker = st.sidebar.selectbox("Select a stock for comparison", list(stock_dict.keys()))

        st.header(f"Model Comparison for {stock_ticker}")

        # Get predictions from all models
        autoreg_pred = get_autoreg_predictions(stock_ticker)
        lstm_pred = get_lstm_prediction(stock_ticker)
        rf_pred = get_random_forest_prediction(stock_ticker)

        # Create a DataFrame for easy comparison
        comparison_df = pd.DataFrame({
            'Model': ['AutoReg', 'LSTM', 'Random Forest'],
            'Prediction': [autoreg_pred['prediction'], lstm_pred['prediction'], rf_pred['prediction']],
            'Accuracy': [autoreg_pred['accuracy'], lstm_pred['accuracy'], rf_pred['accuracy']]
        })

        # Display the comparison table
        st.table(comparison_df)

        # Check for agreement between models
        predictions = [autoreg_pred['prediction'], lstm_pred['prediction'], rf_pred['prediction']]
        if len(set(predictions)) == 1:  # All models agree
            st.markdown(f"### 📊 All models agree: {predictions[0]}")
            st.markdown("This strong agreement suggests a higher confidence in the prediction.")
        elif predictions.count(predictions[0]) == 2 or predictions.count(predictions[1]) == 2:  # Two models agree
            agreed_prediction = max(set(predictions), key=predictions.count)
            agreeing_models = [model for model, pred in zip(['AutoReg', 'LSTM', 'Random Forest'], predictions) if
                               pred == agreed_prediction]
            st.markdown(f"### 📈 Two models agree: {agreed_prediction}")
            st.markdown(
                f"The {' and '.join(agreeing_models)} models suggest to {agreed_prediction}. This agreement increases confidence in the prediction.")
        else:  # No agreement
            st.markdown("### 🔀 Models disagree")
            st.markdown("There's no consensus among the models. Consider this uncertainty in your decision-making.")

        # Display the most accurate model
        most_accurate_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        st.markdown(f"### 🏆 Most accurate model: {most_accurate_model['Model']}")
        st.markdown(f"Accuracy: {most_accurate_model['Accuracy']}%")
        st.markdown(f"Prediction: {most_accurate_model['Prediction']}")

        # Plot accuracy comparison
        st.header("Model Accuracy Comparison")
        fig, ax = plt.subplots()
        ax.bar(comparison_df['Model'], comparison_df['Accuracy'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Model Accuracy Comparison for {stock_ticker}')
        st.pyplot(fig)

        # Additional information
        st.markdown("""
        ### Understanding the Comparison
        - Each model uses different algorithms and may consider different aspects of the stock's historical data.
        - Agreement between models can suggest a stronger signal, but it's not a guarantee of future performance.
        - The most accurate model is determined based on its historical performance, which may not always predict future accuracy.
        - Always consider multiple factors and consult with a financial advisor before making investment decisions.
        """)
    if select_model == "AUTOREG":
        st.markdown("""
        - **Parameter 1**: Value
        - **Parameter 2**: Value
        - **Differentiation Variable**: Explanation
        """)
    elif select_model == "LSTM":
        st.markdown("""
        - **Parameter 1**: Value
        - **Parameter 2**: Value
        - **Differentiation Variable**: Explanation
        """)
    elif select_model == "RANDOM FOREST":
        st.markdown("""
        - **Parameter 1**: Value
        - **Parameter 2**: Value
        - **Differentiation Variable**: Explanation
        """)

    # Example plot
    stock = yf.Ticker('AAPL')
    st.write(f"Showing data for {stock.ticker}")
    plot_stock_data(stock, title=f"{stock.ticker} Stock Price Over Time")
