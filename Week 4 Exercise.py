#Import Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
X_train = pd.get_dummies(train_df.drop(columns=['id', 'Rings']))
y_train = train_df['Rings']
X_test = pd.get_dummies(test_df.drop(columns=['id']))

# Polynomial Regression
X_train_poly = X_train.copy()
X_train_poly['Length_2'] = X_train['Length'] ** 2
X_train_poly['Diameter_2'] = X_train['Diameter'] ** 2
X_train_poly['Interaction'] = X_train['Length'] * X_train['Diameter']

# Fit polynomial regression model
mod_poly = ols('Rings ~ Length + Diameter + Length_2 + Diameter_2 + Interaction', data=pd.concat([X_train_poly, y_train], axis=1))
res_poly = mod_poly.fit()
print("Polynomial Regression Summary:")
print(res_poly.summary())

#Prepare/assess the column
print(X_train)
print(X_train['Length'])

# Maximum value
max_value = X_train['Length'].max()
# Minimum value
min_value = X_train['Length'].min()
# Average (mean) value
mean_value = X_train['Length'].mean()
# Mode (most frequent value)
mode_value = X_train['Length'].mode()[0]  # [0] to get the first mode in case of multiple modes
# Median value
median_value = X_train['Length'].median()
# Print the results
print(f"Max: {max_value}")
print(f"Min: {min_value}")
print(f"Mean: {mean_value}")
print(f"Mode: {mode_value}")
print(f"Median: {median_value}")


# Step Functions

X_train_step = X_train.copy()
# Create new quantile-based categories based on provided min, median, and max values
X_train_step['Q1'] = np.where(X_train['Length'] <= 0.075, 1, 0)
X_train_step['Q2'] = np.where((X_train['Length'] > 0.075) & (X_train['Length'] <= 0.545), 1, 0)
X_train_step['Q3'] = np.where((X_train['Length'] > 0.545) & (X_train['Length'] <= 0.815), 1, 0)
X_train_step['Q4'] = np.where(X_train['Length'] > 0.815, 1, 0)

# Fit step functions model
mod_step = ols('Rings ~ Q1 + Q2 + Q3 + Q4 + Diameter', data=pd.concat([X_train_step, y_train], axis=1))
res_step = mod_step.fit()
print("Step Functions Summary:")
print(res_step.summary())

# Splines

X_train_spline = X_train.copy()

X_train_spline['Squared'] = X_train['Length'] ** 2
X_train_spline['Cubic'] = X_train['Length'] ** 3

# Adjust knots based on the new breakpoints (median and max values)
X_train_spline['Knot1'] = np.where(X_train['Length'] > 0.545, (X_train['Length'] - 0.545) ** 3, 0)
X_train_spline['Knot2'] = np.where(X_train['Length'] > 0.815, (X_train['Length'] - 0.815) ** 3, 0)

# Fit spline regression model
mod_spline = ols('Rings ~ Length + Squared + Cubic + Knot1 + Knot2 + Diameter', data=pd.concat([X_train_spline, y_train], axis=1))
res_spline = mod_spline.fit()
print("Spline Regression Summary:")
print(res_spline.summary())

# Predictions on test data using the best model (e.g., polynomial regression)
X_test_poly = X_test.copy()
X_test_poly['Length_2'] = X_test['Length'] ** 2
X_test_poly['Diameter_2'] = X_test['Diameter'] ** 2
X_test_poly['Interaction'] = X_test['Length'] * X_test['Diameter']

# Predict and save results
test_predictions = res_poly.predict(X_test_poly)
submission = pd.DataFrame({'id': test_df['id'], 'Rings': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created.")
