# Store Sales - Time Series Forecasting
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as ets
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the transactions dataset (assuming the dataset has a 'transactions' column)
#file_path = r"C:\PhD\4. TIM-8555\Week 8\transactions.csv"
#data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
data = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv', parse_dates = ['date'], index_col='date')

# Ensure the data contains only the 'transactions' column (if there are additional columns)
data = data[['transactions']]  # Keep only the 'transactions' column

# Resampling the data to monthly frequency and summing values
data = data.resample('MS').sum()  # 'MS' for start of the month, not 'M'

# Verify the time series is 1-dimensional
print(data.head())

# Plot the time series to visualize the data
data.plot()
plt.ylabel("Store Transactions")
plt.title("Store Transactions Over Time")
plt.show()

# Now proceed with the model training and forecasting as before
# Split the dataset into training and testing sets
train = data.iloc[:-12]  # Use all but the last 12 months for training
test = data.iloc[-12:]   # Use the last 12 months for testing

# Additive ETS Model Forecast
model_ets = ets(train['transactions'], error="add", trend="add", seasonal="add", damped_trend=True, seasonal_periods=12)
fit_ets = model_ets.fit()

# Get predictions from the ETS model for the test period
pred_ets = fit_ets.get_prediction(start=test.index[0], end=test.index[-1])
ets_df = pred_ets.summary_frame(alpha=0.05)
ets_df['test'] = test['transactions']

# Plot the actual data, fitted values, and forecasts for ETS model
data['transactions'].plot(label="Actual", color='black')
fit_ets.fittedvalues.plot(label="Fitted (ETS)", color='blue')
ets_df['test'].plot(label='Test (Actual)', color='red')
ets_df['mean'].plot(label='Forecast (ETS)', color='green')
plt.legend()
plt.title("ETS Model: Actual vs Forecasted Transactions")
plt.show()

# ARIMA Model Forecast
# Plot the ACF and PACF to help in ARIMA order selection
plot_acf(train)
plot_pacf(train)
plt.show()

# Fit an ARIMA model
model_arima = ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fit_arima = model_arima.fit()

# Get predictions from the ARIMA model for the test period
pred_arima = fit_arima.get_prediction(start=test.index[0], end=test.index[-1])
arima_df = pred_arima.summary_frame()
arima_df['test'] = test

# Plot the actual data, fitted values, and forecasts for ARIMA model
data.plot(label="Actual", color='black')
fit_arima.fittedvalues.plot(label="Fitted (ARIMA)", color='blue')
arima_df['test'].plot(label='Test (Actual)', color='red')
arima_df['mean'].plot(label='Forecast (ARIMA)', color='green')
plt.legend()
plt.title("ARIMA Model: Actual vs Forecasted Transactions")
plt.show()

# Metrics Calculation
def calculate_metrics(test, y_pred):
    error = test - y_pred
    MSE = np.mean(error**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(error))
    MAPE = np.mean(np.abs(error) / test)
    ME = np.mean(error)
    MPE = np.mean((test - y_pred) / test)
    return {"MSE": MSE, "RMSE": RMSE, "MAE": MAE, "MAPE": MAPE, "ME": ME, "MPE": MPE}

# Calculate the error metrics for ETS and ARIMA models
ets_metrics = calculate_metrics(test, ets_df['mean'])
arima_metrics = calculate_metrics(test, arima_df['mean'])

# Create a DataFrame to compare the results
metrics_df = pd.DataFrame([ets_metrics, arima_metrics], index=["ETS", "ARIMA"])
print(metrics_df)


