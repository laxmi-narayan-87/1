import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the ML model
try:
    model = joblib.load('model.pkl')  # Load the machine learning model
except:
    model = None  # Fallback in case the model can't be loaded, no output shown

# Title for the app
st.title("AgroValue: Price Prediction for Agricultural Commodities")

# CSV File Path
file_path = "monthly_data.csv"  

# Load data from CSV
try:
    df = pd.read_csv(file_path)
except:
    df = None  # If CSV loading fails, no output shown

if df is not None:
    # Data preprocessing for SARIMAX forecasting
    df.set_index('Commodities', inplace=True)
    df = df.T
    df.index = pd.date_range(start='2014-01', periods=len(df), freq='ME')
    df = df.ffill()  # Forward fill to handle missing data

    # Get commodity list from CSV data
    commodities = df.columns.tolist()

    # Select a commodity for forecasting
    selected_commodity = st.selectbox("Choose a Commodity for SARIMAX Forecast", commodities)

    # SARIMAX Forecasting
    if st.button("Submit"):
        try:
            data = df[selected_commodity]

            # Train SARIMAX model
            model_sarimax = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            sarimax_model = model_sarimax.fit(disp=False)

            # Forecast for 5 years (60 months)
            forecast = sarimax_model.get_forecast(steps=12 * 5)  # 5 years forecast
            forecasted_values = forecast.predicted_mean

            # Create forecast DataFrame
            forecast_years = pd.date_range(start='2025-01', periods=12 * 5, freq='M')
            forecast_df = pd.DataFrame({'Year': forecast_years, f'{selected_commodity}_Price_Forecast': forecasted_values})

            # Display forecast
            st.write(f"### {selected_commodity} Price Forecast (2025-2029)")
            st.write(forecast_df)

            # Plot actual vs forecasted values
            plt.figure(figsize=(10, 6))
            plt.plot(data, label=f'Actual {selected_commodity} Prices')
            plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
            plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
            plt.xlabel('Year')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            # Calculate and display RMSE for training
            train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
            st.write(f"Training RMSE: {train_rmse:.4f}")
        except:
            pass  # Suppress error if any occurs during SARIMAX

# Input fields for machine learning model prediction (adjust based on your model's inputs)
if model is not None:
    st.write("### Machine Learning Model Prediction")
    feature1 = st.number_input("Enter Feature 1 for ML Prediction", value=0.0)
    feature2 = st.number_input("Enter Feature 2 for ML Prediction", value=0.0)

    # Convert input features to numpy array
    features = np.array([[feature1, feature2]])  # Adjust shape to fit your model

    # Predict using ML model
    if st.button('Predict with ML Model'):
        try:
            prediction = model.predict(features)
            st.success(f"Predicted Price using ML Model: {prediction[0]}")
        except:
            pass  # Suppress error if any occurs during prediction

# Additional models: Random Forest, XGBoost, and LSTM
st.write("### Additional Models")

# Random Forest
st.write("#### Random Forest")
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
st.write(f"Random Forest Accuracy: {rf_model.score(X_test, y_test)}")

# XGBoost
st.write("#### XGBoost")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
st.write(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)}")

# LSTM
st.write("#### LSTM")
time_series_data = np.sin(np.arange(0, 100, 0.1)).reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(time_series_data)
time_step = 10
X, y = [], []
for i in range(len(data_scaled) - time_step - 1):
    X.append(data_scaled[i:(i + time_step), 0])
    y.append(data_scaled[i + time_step, 0])
X = np.array(X).reshape(-1, time_step, 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
train_predict = lstm_model.predict(X_train)
test_predict = lstm_model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
st.write(f"LSTM Train Prediction: {train_predict}")
st.write(f"LSTM Test Prediction: {test_predict}")
