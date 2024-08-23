import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import scipy.stats as stats

# Load the datasets
train = pd.read_csv(r'/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s4e4/test.csv')
#train_data_path = r'C:\PhD\4. TIM-8555\Week 2\train.csv'
#test_data_path = r'C:\PhD\4. TIM-8555\Week 2\test.csv'
#train_df = pd.read_csv(train_data_path)
#test_df = pd.read_csv(test_data_path)

# Display the first few rows of the training data to understand its structure
print(train_df.head())
print(test_df.head())

# 1. Simple OLS Regression with Continuous Variables

# Define predictor and response variables for training
X_train_cont = train_df[['Length']]  # Predictor variable
y_train_cont = train_df['Rings']     # Response variable

# Add constant to the predictor variables
X_train_cont = sm.add_constant(X_train_cont)

# Fit the OLS regression model
model_train_cont = sm.OLS(y_train_cont, X_train_cont).fit()

# Print the summary of the regression model
print("\nOLS Regression with Continuous Variables (Training Data)")
print(model_train_cont.summary())

# Predict on the test data
X_test_cont = test_df[['Length']]
X_test_cont = sm.add_constant(X_test_cont)
y_pred_cont = model_train_cont.predict(X_test_cont)

# Add predictions to the test dataframe
test_df['Predicted_Rings_Cont'] = y_pred_cont

# Display predictions
print("\nTest Data with Predictions (Continuous Variables)")
print(test_df[['Length', 'Predicted_Rings_Cont']].head())

#######################-------------------------------------#######################

#import pandas as pd
#import statsmodels.api as sm
#from statsmodels.formula.api import ols

# Load the datasets
#train_data_path = r'C:\PhD\4. TIM-8555\Week 2\train.csv'
#test_data_path = r'C:\PhD\4. TIM-8555\Week 2\test.csv'

#train_df = pd.read_csv(train_data_path)
#test_df = pd.read_csv(test_data_path)
# 2. OLS Regression with Dichotomous Variables
# Create dummy variables for the 'Sex' column in training data
train_df_dich = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)
train_df_dich = sm.add_constant(train_df_dich)

# Define the dependent variable
train_df_dich['y'] = train_df['Rings']

# Fit the OLS model
mod = ols('y ~ Rings + Sex_M', data=train_df_dich)
res = mod.fit()
print("\nOLS Regression with Dichotomous Variables (Training Data)")
print(res.summary())

# Prepare the test data with dummy variables
test_df_dich = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)
test_df_dich = sm.add_constant(test_df_dich, has_constant='add')

# Ensure test data has the same columns as training data
test_df_dich = test_df_dich.reindex(columns=train_df_dich.columns, fill_value=0)

# Predict on the test data
test_df_dich['Predicted_Rings_Dich'] = res.predict(test_df_dich)

# Add predictions to the test dataframe
test_df['Predicted_Rings_Dich'] = test_df_dich['Predicted_Rings_Dich']

# Display predictions
print("\nTest Data with Predictions (Dichotomous Variables)")
print(test_df[['Length', 'Sex', 'Predicted_Rings_Dich']].head())

# Fit a model without the constant term
noconst = train_df_dich.copy()  # Make a copy to avoid modifying the original DataFrame
mod2 = ols('y ~ Rings + Sex_M', data=noconst).fit()

# Perform ANOVA on the model
table = sm.stats.anova_lm(mod2)
print("\nANOVA Table")
print(table)

#Uploaded at: https://www.kaggle.com/code/tinsuabdeta/regression-with-abalone-practice

