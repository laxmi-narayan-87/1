# Project: A Study On Agriculture Commodities Price Prediction and Forecasting

import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess

# Function to install packages
def install_package(package):
    try:
        __import__(package)
    except ModuleNotFoundError:
        st.warning(f"{package} is not installed. Installing now...")
        subprocess.check_call([f"pip", "install", package])

# Install required packages if not already installed
install_package('xgboost')
install_package('statsmodels')
install_package('matplotlib')
install_package('seaborn')
install_package('sklearn')

# Import the installed packages
try:
    import xgboost as xgb
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import seaborn as sns
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    st.error(f"Error importing libraries: {e}. Please check your requirements.")

# Load the dataset
try:
    df = pd.read_csv("https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/Agriculture_commodities_dataset.csv")
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# Data Preprocessing
try:
    df.isnull().sum()
    df.info()

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Feature engineering on numerical columns
    df_num = df.select_dtypes(include=['int64', 'float64'])
    df_num_normalized = (df_num - df_num.min()) / (df_num.max() - df_num.min())  # Manual normalization

    # Feature engineering on categorical columns using one-hot encoding
    df_cat = pd.get_dummies(df.select_dtypes(include=object))

    # Concatenate numerical and categorical columns
    df_pred = pd.concat([df_cat, df_num_normalized], axis=1)
    x = df_pred.drop(columns=['modal_price'])
    y = df_pred['modal_price']

    # Train Test Split
    train_size = int(0.8 * len(df_pred))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
except Exception as e:
    st.error(f"Error during data preprocessing: {e}")

# Multi Linear Regression (MLR)
try:
    MLR_model1 = sm.OLS(y_train, x_train).fit()
    st.write(MLR_model1.summary())

    # Prediction using MLR
    y_test_pred = MLR_model1.predict(x_test)

    # Evaluation Metrics for MLR
    st.write("MLR Metrics:")
    st.write("MSE:", mean_squared_error(y_test, y_test_pred))
    st.write("MAE:", mean_absolute_error(y_test, y_test_pred))
    st.write("R2:", r2_score(y_test, y_test_pred))
except Exception as e:
    st.error(f"Error during MLR training or evaluation: {e}")

# XGBoost Decision Tree
try:
    xgb_dtree = xgb.XGBRegressor(objective='reg:squarederror', max_depth=5)
    xgb_dtree.fit(x_train, y_train)
    y_pred1 = xgb_dtree.predict(x_test)

    # Evaluation Metrics for XGBoost Decision Tree
    st.write("\nXGBoost Decision Tree Metrics:")
    st.write("MSE:", mean_squared_error(y_test, y_pred1))
    st.write("MAE:", mean_absolute_error(y_test, y_pred1))
    st.write("R2:", r2_score(y_test, y_pred1))
except Exception as e:
    st.error(f"Error during XGBoost Decision Tree training or evaluation: {e}")

# XGBoost Random Forest
try:
    xgb_rf = xgb.XGBRFRegressor(objective='reg:squarederror', max_depth=5, n_estimators=100)
    xgb_rf.fit(x_train, y_train)
    y_pred2 = xgb_rf.predict(x_test)

    # Evaluation Metrics for XGBoost Random Forest
    st.write("\nXGBoost Random Forest Metrics:")
    st.write("MSE:", mean_squared_error(y_test, y_pred2))
    st.write("MAE:", mean_absolute_error(y_test, y_pred2))
    st.write("R2:", r2_score(y_test, y_pred2))
except Exception as e:
    st.error(f"Error during XGBoost Random Forest training or evaluation: {e}")

# Visualization
try:
    plt.figure(figsize=(10, 6))

    # Scatter plot for Min_price vs Max_price
    sns.scatterplot(x='min_price', y='max_price', data=df, color='blue', alpha=0.7)
    plt.title('Scatter Plot for Min_price vs Maximum_price')
    plt.xlabel('Min_price')
    plt.ylabel('Max_price')
    plt.show()
    st.pyplot(plt)

    # Bar chart for Month and Commodity
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Month', hue='Commodity', data=df, palette='Set2')
    plt.title('Bar Chart for Month and Commodity')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.show()
    st.pyplot(plt)

    # Correlation heatmap
    df_numeric = df.select_dtypes(include=[np.number])
    correlation_matrix = df_numeric.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Plot')
    plt.show()
    st.pyplot(plt)
except Exception as e:
    st.error(f"Error during visualization: {e}")

# ---------------------------------------------------------------

# SARIMAX Commodity Price Forecasting
try:
    file_path = "https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/monthly_data.csv"
    df_forecast = pd.read_csv(file_path)

    # Preprocessing for SARIMAX
    df_forecast.set_index('Commodities', inplace=True)
    df_forecast = df_forecast.T
    df_forecast.index = pd.date_range(start='2014-01', periods=len(df_forecast), freq='ME')
    df_forecast = df_forecast.ffill()

    # List of commodities for forecasting
    commodities = df_forecast.columns.tolist()

    # Choose a commodity for forecasting
    selected_commodity = commodities[0]  # You can change this to any commodity

    # Apply SARIMAX
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
    st.write(f"Forecast for {selected_commodity} (2025-2029):")
    st.write(forecast_df)

    # Plotting the forecasted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'Actual {selected_commodity} Prices')
    plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
    plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    # Training RMSE
    train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
    st.write(f"Training RMSE: {train_rmse:.4f}")
except Exception as e:
    st.error(f"Error during SARIMAX forecasting: {e}")
