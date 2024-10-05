#####################  WEEK 8 - Course 8550 Final Assigment###############
#######  By: Tinsae Abdeta

#1. Store Sales Time Series Forecasting

# Store Sales - Time Series Forecasting
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as ets
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the transactions dataset (assuming the dataset has a 'transactions' column)
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


#2 House Price Prediction
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Load the datasets
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#train_data = pd.read_csv(r"C:\PhD\4. TIM-8555\Week 8\train.csv")
#test_data = pd.read_csv(r"C:\PhD\4. TIM-8555\Week 8\test.csv")

# Handle missing values separately for numeric and categorical columns

# Fill numeric columns with the median, excluding 'SalePrice' in the train data
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.drop('SalePrice')
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].median())
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].median())

# Fill categorical columns with the most frequent value (mode)
categorical_cols = train_data.select_dtypes(include=[object]).columns
train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])
test_data[categorical_cols] = test_data[categorical_cols].fillna(test_data[categorical_cols].mode().iloc[0])

# Convert categorical variables to dummy variables (one-hot encoding)
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# Align the test data with the train data (remove extra columns in the test set)
train_data, test_data = train_data.align(test_data, join='left', axis=1)
test_data.fillna(0, inplace=True)  # Fill any remaining missing values in test after aligning

# Drop 'SalePrice' and 'Id' from the training data for model training
X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']

# Drop 'Id' from the test data (since it's not needed for prediction)
X_test = test_data.drop(['Id'], axis=1)

# Ensure train and test have the same columns (ignore the target column 'SalePrice')
# Perform column alignment again just to be safe
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Ensure train and test have the same columns (important step to avoid the error)
assert X.columns.equals(X_test.columns), "Train and test feature columns do not match!"

# Split the training data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_valid_poly = poly.transform(X_valid)
X_test_poly = poly.transform(X_test)

# Standardize the data
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_valid_poly = scaler.transform(X_valid_poly)
X_test_poly = scaler.transform(X_test_poly)

# Model 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_poly, y_train)

# Predict and evaluate on validation set
y_pred_train = linear_model.predict(X_train_poly)
y_pred_valid = linear_model.predict(X_valid_poly)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_valid = mean_squared_error(y_valid, y_pred_valid)

print(f"Linear Regression - Train MSE: {mse_train}, Validation MSE: {mse_valid}")

# Model 2: Ridge Regression with Polynomial Features
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_poly, y_train)

# Predict and evaluate on validation set
y_pred_ridge_train = ridge_model.predict(X_train_poly)
y_pred_ridge_valid = ridge_model.predict(X_valid_poly)
mse_ridge_train = mean_squared_error(y_train, y_pred_ridge_train)
mse_ridge_valid = mean_squared_error(y_valid, y_pred_ridge_valid)

print(f"Ridge Regression - Train MSE: {mse_ridge_train}, Validation MSE: {mse_ridge_valid}")

# Dimension Reduction with PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_poly)
X_valid_pca = pca.transform(X_valid_poly)
X_test_pca = pca.transform(X_test_poly)

# Model 3: Lasso Regression with PCA
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_pca, y_train)

# Predict and evaluate on validation set
y_pred_lasso_train = lasso_model.predict(X_train_pca)
y_pred_lasso_valid = lasso_model.predict(X_valid_pca)
mse_lasso_train = mean_squared_error(y_train, y_pred_lasso_train)
mse_lasso_valid = mean_squared_error(y_valid, y_pred_lasso_valid)

print(f"Lasso Regression (PCA) - Train MSE: {mse_lasso_train}, Validation MSE: {mse_lasso_valid}")


###########################################
#3: San Francisco Clime Classification
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load the train and test datasets (use a subset of the data for faster testing)
train_data = pd.read_csv("/kaggle/input/san-francisco-crime-classification/train.csv").sample(frac=0.1, random_state=42)  # 10% of the data
test_data = pd.read_csv("/kaggle/input/san-francisco-crime-classification/test.csv")

# Display the columns to verify if 'Category' exists
print("Train Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

# Ensure the 'Category' column exists, or modify it to the correct target column
if 'Category' in train_data.columns:
    # Extract 'Category' as the target variable and features from 'Dates', 'DayOfWeek', 'PdDistrict', 'X', 'Y'
    X = train_data[['Dates', 'DayOfWeek', 'PdDistrict', 'X', 'Y']]
    y = train_data['Category']
else:
    raise KeyError("The 'Category' column does not exist in the training dataset. Check the correct column name.")

# For test data, exclude 'Id' from the features
X_test = test_data[['Dates', 'DayOfWeek', 'PdDistrict', 'X', 'Y']]

# Ensure 'Dates' column is converted to datetime format
X.loc[:, 'Dates'] = pd.to_datetime(X['Dates'], errors='coerce')
X_test.loc[:, 'Dates'] = pd.to_datetime(X_test['Dates'], errors='coerce')

# Drop rows where 'Dates' could not be converted to datetime
X = X.dropna(subset=['Dates'])
X_test = X_test.dropna(subset=['Dates'])

# Extract year, month, and hour from 'Dates'
X['Dates'] = pd.to_datetime(X['Dates'], errors='coerce')
X['Year'] = X['Dates'].dt.year
X['Month'] = X['Dates'].dt.month
X['Hour'] = X['Dates'].dt.hour
X_test['Dates'] = pd.to_datetime(X_test['Dates'], errors='coerce')
X_test['Year'] = X_test['Dates'].dt.year
X_test['Month'] = X_test['Dates'].dt.month
X_test['Hour'] = X_test['Dates'].dt.hour

# Drop the 'Dates' column after feature extraction
X = X.drop('Dates', axis=1)
X_test = X_test.drop('Dates', axis=1)

# One-Hot Encoding for 'DayOfWeek' and 'PdDistrict'
X = pd.get_dummies(X, columns=['DayOfWeek', 'PdDistrict'])
X_test = pd.get_dummies(X_test, columns=['DayOfWeek', 'PdDistrict'])

# Align the train and test data to have the same columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Ensure X and y are aligned and have the same number of samples
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the data into training and validation sets
if X.shape[0] == y.shape[0] and X.shape[0] > 0:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    raise ValueError("X and y must have the same number of rows.")

# Model 1: Decision Tree Classifier (no scaling required)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Evaluate the Decision Tree model on the validation set
y_pred_tree = decision_tree.predict(X_valid)
print("Decision Tree Classification Report:")
print(classification_report(y_valid, y_pred_tree))

# Model 2: Random Forest Classifier (reduce number of trees to 10 for faster execution)
random_forest = RandomForestClassifier(n_estimators=10, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions and evaluate the Random Forest model
y_pred_forest = random_forest.predict(X_valid)
print("Random Forest Classification Report:")
print(classification_report(y_valid, y_pred_forest))

# Model 3: Support Vector Machine (SVM) - Faster using LinearSVC
# Standardize the features before applying SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Fit LinearSVC (which is faster than SVC with kernel='linear')
svm = LinearSVC(max_iter=1000, random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions and evaluate the SVM model
y_pred_svm = svm.predict(X_valid_scaled)
print("SVM Classification Report:")
print(classification_report(y_valid, y_pred_svm))

# Final predictions for Kaggle submission (using the Random Forest model)
y_test_pred_forest = random_forest.predict(X_test)
