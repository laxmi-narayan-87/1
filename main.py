# Project: A Study On Agriculture Commodities Price Prediction and Forecasting

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/Agriculture_commodities_dataset.csv")

# Data Preprocessing
df.isnull().sum()
df.info()

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Feature engineering on numerical columns
df_num = df.select_dtypes(include=['int64', 'float64'])
mn = MinMaxScaler()
df_num_mn = pd.DataFrame(mn.fit_transform(df_num), columns=df_num.columns)

# Feature engineering on categorical columns
df_cat = df.select_dtypes(include=object)
le = LabelEncoder()

for col in ['APMC', 'Commodity', 'Month', 'district_name', 'state_name']:
    df_cat[col] = le.fit_transform(df_cat[col])

# Concatenate numerical and categorical columns
df_pred = pd.concat([df_cat, df_num_mn], axis=1)
x = df_pred.iloc[:, :9]
y = df_pred[['modal_price']]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Multi Linear Regression (MLR)
MLR_model1 = sm.OLS(y_train, x_train).fit()
print(MLR_model1.summary())

# Prediction using MLR
y_test_pred = MLR_model1.predict(x_test)

# Evaluation Metrics for MLR
print("MLR Metrics:")
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("R2:", r2_score(y_test, y_test_pred))

# Decision Tree
dtree = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
dtree.fit(x_train, y_train)
y_pred1 = dtree.predict(x_test)

# Evaluation Metrics for Decision Tree
print("\nDecision Tree Metrics:")
print("MSE:", mean_squared_error(y_test, y_pred1))
print("MAE:", mean_absolute_error(y_test, y_pred1))
print("R2:", r2_score(y_test, y_pred1))

# Random Forest Regression
classifier = RandomForestRegressor(n_estimators=500, criterion='squared_error')
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_test)

# Evaluation Metrics for Random Forest
print("\nRandom Forest Metrics:")
print("MSE:", mean_squared_error(y_test, y_pred2))
print("MAE:", mean_absolute_error(y_test, y_pred2))
print("R2:", r2_score(y_test, y_pred2))

# Visualization
plt.figure(figsize=(10, 6))

# Scatter plot for Min_price vs Max_price
sns.scatterplot(x='min_price', y='max_price', data=df, color='blue', alpha=0.7)
plt.title('Scatter Plot for Min_price vs Maximum_price')
plt.xlabel('Min_price')
plt.ylabel('Max_price')
plt.show()

# Bar chart for Month and Commodity
plt.figure(figsize=(10, 6))
sns.countplot(x='Month', hue='Commodity', data=df, palette='Set2')
plt.title('Bar Chart for Month and Commodity')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
df_numeric = df.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Plot')
plt.show()

# ---------------------------------------------------------------

# SARIMAX Commodity Price Forecasting

# Load dataset for SARIMAX
file_path = "https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/monthly_data.csv"  # Update this with your local file path
df_forecast = pd.read_csv(file_path)

# Preprocessing for SARIMAX
df_forecast.set_index('Commodities', inplace=True)
df_forecast = df_forecast.T
df_forecast.index = pd.date_range(start='2014-01', periods=len(df_forecast), freq='M')
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
print(f"Forecast for {selected_commodity} (2025-2029):")
print(forecast_df)

# Plotting the forecasted and actual values
plt.figure(figsize=(10, 6))
plt.plot(data, label=f'Actual {selected_commodity} Prices')
plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.show()

# Training RMSE
train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
print(f"Training RMSE: {train_rmse:.4f}")
