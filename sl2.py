import streamlit as st
import pandas as pd
from yahoo_fin import stock_info as si
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime


# Load data using yahoo_fin
def load_data(ticker, start_date, end_date):
    data = si.get_data(ticker, start_date=start_date, end_date=end_date)
    data.reset_index(inplace=True)  # Reset index to move the date to a column
    return data


# Train model using Linear Regression
def train_model(data):
    data['date'] = pd.to_datetime(data['index']).map(datetime.datetime.toordinal)  # Update the column name to 'index'
    X = data[['date']]
    y = data['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


# Predict stock prices
def predict(model, X, days_ahead):
    last_date = X['date'].max()
    future_dates = np.array([last_date + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)
    future_prices = model.predict(future_dates)
    future_dates = [datetime.datetime.fromordinal(int(date)) for date in future_dates.flatten()]
    return future_dates, future_prices


# Streamlit app
st.title('Stock Price Predictor with Linear Regression')

ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))
days_ahead = st.slider('Days ahead to predict:', 1, 365)

if st.button('Predict'):
    data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.write("No data available for the selected ticker and date range.")
    else:
        model, X_train, y_train, X_test, y_test = train_model(data)
        future_dates, future_prices = predict(model, X_train, days_ahead)

        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close Price': future_prices
        })

        st.write(f'Predicted prices for the next {days_ahead} days:')
        st.write(predictions_df)

        st.line_chart(predictions_df.set_index('Date'))

        # Evaluate model performance
        st.write('Model Performance:')
        r2_score = model.score(X_test, y_test)
        st.write(f'R-squared: {r2_score}')
        st.write(f'Mean Absolute Error: {np.mean(np.abs(model.predict(X_test) - y_test))}')
        st.write(f'Mean Squared Error: {np.mean((model.predict(X_test) - y_test) ** 2)}')
