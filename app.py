import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset for machine learning models
@st.cache
def load_ml_data():
    df = pd.read_csv("https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/Agriculture_commodities_dataset.csv")
    return df

df = load_ml_data()
st.title("Agriculture Commodities Price Prediction and Forecasting")

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df_num = df.select_dtypes(include=['int64', 'float64'])

# Normalizing numeric data (similar to MinMaxScaler)
df_num_normalized = (df_num - df_num.min()) / (df_num.max() - df_num.min())

df_cat = df.select_dtypes(include=object)
df_cat_encoded = pd.get_dummies(df_cat)  # One-hot encoding for categorical variables

df_pred = pd.concat([df_cat_encoded, df_num_normalized], axis=1)
x = df_pred.drop(columns=['modal_price'])
y = df_pred['modal_price']

# Train Test Split
train_size = int(0.8 * len(df_pred))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# User can select which XGBoost model to run
model_choice = st.selectbox('Choose Model', ['XGBoost Decision Tree', 'XGBoost Random Forest'])

# XGBoost Model Setup
if model_choice == 'XGBoost Decision Tree':
    model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Show metrics
    st.write("### XGBoost Decision Tree Metrics")
    st.write(f"MSE: {np.mean((y_test - y_pred) ** 2)}")
    st.write(f"MAE: {np.mean(abs(y_test - y_pred))}")
    st.write(f"R2 Score: {1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)}")

elif model_choice == 'XGBoost Random Forest':
    model = xgb.XGBRFRegressor(objective='reg:squarederror', max_depth=5, n_estimators=100)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Show metrics
    st.write("### XGBoost Random Forest Metrics")
    st.write(f"MSE: {np.mean((y_test - y_pred) ** 2)}")
    st.write(f"MAE: {np.mean(abs(y_test - y_pred))}")
    st.write(f"R2 Score: {1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)}")

# Visualization of Min_price vs Max_price
st.write("### Visualization of Min_price vs Max_price")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='min_price', y='max_price', data=df, color='blue', alpha=0.7)
plt.title('Scatter Plot for Min_price vs Maximum_price')
plt.xlabel('Min_price')
plt.ylabel('Max_price')
st.pyplot(plt)

# Load dataset for SARIMAX forecasting
@st.cache
def load_sarimax_data():
    df_forecast = pd.read_csv("https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/monthly_data.csv")
    return df_forecast

df_forecast = load_sarimax_data()
df_forecast.set_index('Commodities', inplace=True)
df_forecast = df_forecast.T
df_forecast.index = pd.date_range(start='2014-01', periods=len(df_forecast), freq='ME')
df_forecast = df_forecast.ffill()

# SARIMAX Forecasting
st.write("## SARIMAX Forecasting")
commodities = df_forecast.columns.tolist()
selected_commodity = st.selectbox("Choose a Commodity for Forecasting", commodities)

if st.button("Submit"):
    data = df_forecast[selected_commodity]
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    sarimax_model = model.fit(disp=False)

    # Forecast for 5 years (12 months per year)
    forecast = sarimax_model.get_forecast(steps=12 * 5)
    forecasted_values = forecast.predicted_mean

    # Create a dataframe for forecasted values
    forecast_years = pd.date_range(start='2025-01', periods=12 * 5, freq='M')
    forecast_df = pd.DataFrame({'Year': forecast_years, f'{selected_commodity}_Price_Forecast': forecasted_values})

    # Display forecast dataframe
    st.write(f"### {selected_commodity} Price Forecast (2025-2029):")
    st.write(forecast_df)

    # Plotting the forecasted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'Actual {selected_commodity} Prices')
    plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
    plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Training RMSE
    train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
    st.write(f"Training RMSE: {train_rmse:.4f}")
