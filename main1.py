import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Train ML models on the same data
if df is not None:
    # Prepare the data for ML models
    df_ml = df.T.reset_index(drop=True)  # Transpose data and reset index
    df_ml = df_ml.dropna()  # Drop any rows with NaN values

    # Split data into features and target (using first column as target for demonstration)
    X = df_ml.iloc[:, 1:]  # Features
    y = df_ml.iloc[:, 0]  # Target (price)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    st.write("### Additional Models")

    st.write("#### Random Forest")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    st.write(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")

    # XGBoost Model
    st.write("#### XGBoost")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    st.write(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

    # LSTM Model
    st.write("#### LSTM")

    # Data Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(X)  # Scale the features for LSTM

    time_step = 10
    X_lstm, y_lstm = [], []

    for i in range(len(data_scaled) - time_step - 1):
        X_lstm.append(data_scaled[i:(i + time_step), 0])
        y_lstm.append(data_scaled[i + time_step, 0])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    # Reshape X for LSTM input
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

    # Train Test Split for LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # LSTM Model Definition
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=1, verbose=1)

    # Make Predictions
    train_predict_lstm = lstm_model.predict(X_train_lstm)
    test_predict_lstm = lstm_model.predict(X_test_lstm)

    # Inverse scaling to get original values
    train_predict_lstm = scaler.inverse_transform(train_predict_lstm)
    test_predict_lstm = scaler.inverse_transform(test_predict_lstm)

    st.write(f"LSTM Train Prediction: {train_predict_lstm}")
    st.write(f"LSTM Test Prediction: {test_predict_lstm}")
