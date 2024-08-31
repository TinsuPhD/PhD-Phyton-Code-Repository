import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the datasets
train_data_path = pd.read_csv(r'/kaggle/input/playground-series-s4e4/train.csv')
test_data_path = pd.read_csv(r'/kaggle/input/playground-series-s4e4/test.csv')
#train_data_path = r'C:\PhD\4. TIM-8555\Week 2\train.csv'
#test_data_path = r'C:\PhD\4. TIM-8555\Week 2\test.csv'

#train_df = pd.read_csv(train_data_path)
#test_df = pd.read_csv(test_data_path)

train_df = train_data_path
test_df = test_data_path

# Prepare the data
X_train_raw = pd.get_dummies(train_df.drop(columns=['id', 'Rings']))
y_train = train_df['Rings']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Regularization Models
# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_predictions = lasso.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_predictions))

print("Lasso RMSE:", lasso_rmse)

# Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
ridge_predictions = ridge.predict(X_val)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_predictions))

print("Ridge RMSE:", ridge_rmse)

# Predictions on test data
X_test_raw = pd.get_dummies(test_df.drop(columns=['id']))
X_test_scaled = scaler.transform(X_test_raw)

lasso_test_predictions = lasso.predict(X_test_scaled)
ridge_test_predictions = ridge.predict(X_test_scaled)



# Principal Components Regression (PCR)
# Define the PCA and Regression Pipeline
pca = PCA(n_components=10)
regression = LinearRegression()

pcr_pipeline = Pipeline([
    ('pca', pca),
    ('regression', regression)
])

# Fit the PCR model
pcr_pipeline.fit(X_train, y_train)
pcr_predictions = pcr_pipeline.predict(X_val)
pcr_rmse = np.sqrt(mean_squared_error(y_val, pcr_predictions))

print("PCR RMSE:", pcr_rmse)

# Predictions on test data using PCR
pcr_test_predictions = pcr_pipeline.predict(X_test_scaled)


