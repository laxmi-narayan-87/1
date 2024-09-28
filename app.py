#import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
try:
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    st.write("scikit-learn imported successfully!")
except ModuleNotFoundError:
    st.write("scikit-learn not installed. Please check requirements.")

#from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset for ML models
@st.cache
def load_ml_data():
    df = pd.read_csv("https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/Agriculture_commodities_dataset.csv")
    return df

df = load_ml_data()
st.title("Agriculture Commodities Price Prediction and Forecasting")

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df_num = df.select_dtypes(include=['int64', 'float64'])
mn = MinMaxScaler()
df_num_mn = pd.DataFrame(mn.fit_transform(df_num), columns=df_num.columns)

df_cat = df.select_dtypes(include=object)
le = LabelEncoder()

for col in ['APMC', 'Commodity', 'Month', 'district_name', 'state_name']:
    df_cat[col] = le.fit_transform(df_cat[col])

df_pred = pd.concat([df_cat, df_num_mn], axis=1)
x = df_pred.iloc[:, :9]
y = df_pred[['modal_price']]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# User can select which model to run
model_choice = st.selectbox('Choose Machine Learning Model', ['Decision Tree', 'Random Forest'])

# Train and Predict using Decision Tree
if model_choice == 'Decision Tree':
    dtree = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
    dtree.fit(x_train, y_train)
    y_pred = dtree.predict(x_test)

    # Show metrics
    st.write("### Decision Tree Metrics")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    st.write(f"R2: {r2_score(y_test, y_pred)}")

# Train and Predict using Random Forest
elif model_choice == 'Random Forest':
    classifier = RandomForestRegressor(n_estimators=500, criterion='squared_error')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    # Show metrics
    st.write("### Random Forest Metrics")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    st.write(f"R2: {r2_score(y_test, y_pred)}")

# Visualization
st.write("### Visualization of Min_price vs Max_price")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='min_price', y='max_price', data=df, color='blue', alpha=0.7)
plt.title('Scatter Plot for Min_price vs Maximum_price')
plt.xlabel('Min_price')
plt.ylabel('Max_price')
st.pyplot(plt)

# Load dataset for SARIMAX
@st.cache
def load_sarimax_data():
    df_forecast = pd.read_csv("https://raw.githubusercontent.com/laxmi-narayan-87/AgroValue/refs/heads/main/monthly_data.csv")  # Replace with the correct file path
    return df_forecast

df_forecast = load_sarimax_data()
df_forecast.set_index('Commodities', inplace=True)
df_forecast = df_forecast.T
df_forecast.index = pd.date_range(start='2014-01', periods=len(df_forecast), freq='M')
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
