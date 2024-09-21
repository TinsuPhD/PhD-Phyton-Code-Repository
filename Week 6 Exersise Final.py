# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report as cr
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier as bag, RandomForestClassifier as rfc, GradientBoostingClassifier as gbc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Obesity Risk dataset (assuming the CSV file is saved locally)
#train = pd.read_csv('C:/PhD/4. TIM-8555/Week 5/train.csv')
#test = pd.read_csv('C:/PhD/4. TIM-8555/Week 5/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
# Display the first few rows of the datasets
print("Train Dataset Sample:\n", train.head())
print("Test Dataset Sample:\n", test.head())


# Preprocessing the data (encoding categorical variables, scaling numerical features)
train = pd.get_dummies(train, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])
test = pd.get_dummies(test, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])

# Define features and target variable
X_train = train.drop(columns=['id'])  # Features
y_train = train['NObeyesdad_Overweight_Level_II']  # Replace with the target column for prediction

# Split the dataset into training and testing sets (to validate model performance)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Cross-validation setup
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#-----------------------------#
# 1. Decision Tree Classifier  #
#-----------------------------#
print("\nDecision Tree Classifier:")
model_tree = DecisionTreeClassifier(random_state=1, max_depth=5)  # Limiting the depth to prevent overfitting and speed up the plot rendering
n_scores_tree = cross_val_score(model_tree, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy (Decision Tree): %.3f (%.3f)' % (np.mean(n_scores_tree), np.std(n_scores_tree)))

# Train and predict using the Decision Tree
tree_model = model_tree.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_val)
print(cr(y_val, tree_predictions))

# Plot the decision tree with improved rendering speed
plt.figure(figsize=(12,8))
plot_tree(tree_model, feature_names=X_train.columns, filled=True)
plt.savefig('plot.png', dpi=300)  # Save the plot first
plt.show(block=False)  # Render without blocking

#-----------------------------#
# 2. Bagging Classifier        #
#-----------------------------#
print("\nBagging Classifier:")
model_bagging = bag(estimator=DecisionTreeClassifier(), n_estimators=30, random_state=1)  # Corrected parameter name
n_scores_bagging = cross_val_score(model_bagging, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy (Bagging): %.3f (%.3f)' % (np.mean(n_scores_bagging), np.std(n_scores_bagging)))

# Train and predict using the Bagging Classifier
bag_model = model_bagging.fit(X_train, y_train)
bag_predictions = bag_model.predict(X_val)
print(cr(y_val, bag_predictions))

#-----------------------------#
# 3. Random Forest Classifier  #
#-----------------------------#
print("\nRandom Forest Classifier:")
model_rf = rfc(n_estimators=100, random_state=1)
n_scores_rf = cross_val_score(model_rf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy (Random Forest): %.3f (%.3f)' % (np.mean(n_scores_rf), np.std(n_scores_rf)))

# Train and predict using Random Forest
rf_model = model_rf.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_val)
print(cr(y_val, rf_predictions))

# Plot feature importances for Random Forest
feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()

#-----------------------------#
# 4. Boosting Classifier       #
#-----------------------------#
print("\nGradient Boosting Classifier:")
model_gbc = gbc(n_estimators=100, max_depth=1, random_state=1)
n_scores_gbc = cross_val_score(model_gbc, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy (Boosting): %.3f (%.3f)' % (np.mean(n_scores_gbc), np.std(n_scores_gbc)))

# Train and predict using Gradient Boosting
gbc_model = model_gbc.fit(X_train, y_train)
gbc_predictions = gbc_model.predict(X_val)
print(cr(y_val, gbc_predictions))

# Plot feature importances for Gradient Boosting
feat_importances_gbc = pd.Series(gbc_model.feature_importances_, index=X_train.columns)
feat_importances_gbc.nlargest(4).plot(kind='barh')
plt.show()

