#####################  WEEK 8 - Course 8550 Final Assigment###############
#######  By: Tinsae Abdeta

#1. Store Sales Time Series Forecasting

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as ets
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Data: Australian tourists count over several quarters
austourists_data = [
    30.05251300, 19.14849600, 25.31769200, 27.59143700,
    32.07645600, 23.48796100, 28.47594000, 35.12375300,
    36.83848500, 25.00701700, 30.72223000, 28.69375900,
    36.64098600, 23.82460900, 29.31168300, 31.77030900,
    35.17787700, 19.77524400, 29.60175000, 34.53884200,
    41.27359900, 26.65586200, 28.27985900, 35.19115300,
    42.20566386, 24.64917133, 32.66733514, 37.25735401,
    45.24246027, 29.35048127, 36.34420728, 41.78208136,
    49.27659843, 31.27540139, 37.85062549, 38.83704413,
    51.23690034, 31.83855162, 41.32342126, 42.79900337,
    55.70835836, 33.40714492, 42.31663797, 45.15712257,
    59.57607996, 34.83733016, 44.84168072, 46.97124960,
    60.01903094, 38.37117851, 46.97586413, 50.73379646,
    61.64687319, 39.29956937, 52.67120908, 54.33231689,
    66.83435838, 40.87118847, 51.82853579, 57.49190993,
    65.25146985, 43.06120822, 54.76075713, 59.83447494,
    73.25702747, 47.69662373, 61.09776802, 66.05576122
]

# Index of the time series (quarterly frequency, starting from 1999-03)
index = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
austourists = pd.Series(austourists_data, index=index)

# Verify that the 'austourists' series is correctly defined
print(austourists.head())  # Check the first few values

# Plot the time series to ensure everything works
austourists.plot()
plt.ylabel("Australian Tourists")
plt.title("Australian Tourists Over Time")
plt.show()

# Splitting into training and test sets
mytrain = austourists[0:56]  # Training set
mytest = austourists[56:69]  # Test set

# Additive ETS Model Forecast
# Fit an additive ETS (Error-Trend-Seasonality) model in statsmodels
model = ets(mytrain, error="add", trend="add", seasonal="add", damped_trend=True, seasonal_periods=4)
fit = model.fit()

# Get predictions from the model for the test period
pred = fit.get_prediction(start="2013-03-01", end="2015-12-01")
df = pred.summary_frame(alpha=0.05)  # Confidence interval

# Add the test values for comparison
df['test'] = mytest

# Plot the actual data, fitted values, and forecasts
austourists.plot(label="Y", color='black')
plt.ylabel("Australian Tourists")
fit.fittedvalues.plot(label="Fitted (ETS)", color='blue')
df['test'].plot(label='Test (Actual)', color='red')
df['mean'].plot(label='Forecast', color='green')
plt.legend()
plt.title("ETS Model: Actual vs Forecasted Tourists")
plt.show()

# Multiplicative ETS Model Forecast
# Fit a multiplicative ETS model in statsmodels
model = ets(mytrain, error="mul", trend="mul", seasonal="mul", damped_trend=True, seasonal_periods=4)
fit = model.fit()
pred = fit.get_prediction(start="2013-03-01", end="2015-12-01")
df2 = pred.summary_frame()
df2['test'] = mytest

# Plot the actual data, fitted values, and forecasts for multiplicative model
austourists.plot(label="Y")
plt.ylabel("Australian Tourists")
fit.fittedvalues.plot(label="ETS(MMM)")
df2['test'].plot(label='Test y')
df2['mean'].plot(label='Forecast y')
plt.legend()
plt.title("ETS Model: Actual vs Forecasted Tourists (Multiplicative)")
plt.show()

# Metrics Calculation
def myf(mytest, ypred):
    error = mytest - ypred
    MSE = np.mean(error**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(error))
    MAPE = np.mean(np.abs(error) / mytest)
    ME = np.mean(error)
    MPE = np.mean((mytest - ypred) / mytest)
    answer = [MSE, RMSE, MAE, MAPE, ME, MPE]
    names = ["MSE", "RMSE", "MAE", "MAPE", "ME", "MPE"]
    myanswer = dict(zip(names, answer))
    return myanswer

# Call the function with mytest (actual test values) and predicted values
m1 = myf(mytest, df['mean'])
m2 = myf(mytest, df2['mean'])

# Create a DataFrame to compare the results
newdf = pd.DataFrame([m1, m2])
print(newdf)

# Autoregressive Integrated Moving Average (ARIMA) Model

# Plot the ACF and PACF
plot_acf(austourists)
plot_pacf(austourists)
plt.show()

# Fit an ARIMA model and forecast
model = ARIMA(mytrain, order=(1, 1, 1))
fit = model.fit()
pred = fit.get_prediction(start="2013-03-01", end="2015-12-01")
df3 = pred.summary_frame()
df3['test'] = mytest

# Plot the actual data, fitted values, and forecasts for ARIMA model
austourists.plot(label="Y")
plt.ylabel("Australian Tourists")
fit.fittedvalues.plot(label="ARIMA(1,1,1)")
df3['test'].plot(label='Test y')
df3['mean'].plot(label='Forecast y')
plt.legend()
plt.title("ARIMA Model: Actual vs Forecasted Tourists")
plt.show()

model = ARIMA(mytrain, order=(1,1,1), seasonal_order=(1,1,1,4),)
fit = model.fit()
pred = fit.get_prediction(start="2013-03-01", end="2015-12-01")
df4 = pred.summary_frame()
df4['test']=mytest
austourists.plot(label="Y")
plt.ylabel("Australian Tourists")
fit.fittedvalues.plot(label="ETS(AAA)")
df4['test'].plot(label='Test y')
df4['mean'].plot(label='Forecast y')
plt.legend()

# Define the custom function to calculate various error metrics
def myf(mytest, ypred):
    error = sum(mytest - ypred)
    MSE = error ** 2
    RMSE = MSE ** 0.5
    MAE = np.mean(np.abs(error))
    MAPE = np.mean((np.abs(mytest - ypred) / mytest))
    ME = np.mean(error)
    MPE = np.mean((mytest - ypred) / mytest)
    answer = [MSE, RMSE, MAE, MAPE, ME, MPE]
    names = ["MSE", "RMSE", "MAE", "MAPE", "ME", "MPE"]
    myanswer = dict(zip(names, answer))
    return myanswer

# Calculate error metrics for multiple models
m1 = myf(mytest, df['mean'])
m2 = myf(mytest, df2['mean'])
m3 = myf(mytest, df3['mean'])
m4 = myf(mytest, df4['mean'])

# Create a DataFrame to store the results
newdf = pd.DataFrame([m1, m2, m3, m4])
newdf.index = ['ETS(AAA)', 'ETS(MMM)', 'ARIMA(1,1,1)', 'ARIMA(1,1,1)(1,1,1)[4]']

# Display the DataFrame with error metrics for each model
print(newdf)



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
