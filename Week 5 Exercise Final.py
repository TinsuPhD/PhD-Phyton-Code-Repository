# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# Load datasets (assuming they are saved locally)
train = pd.read_csv('C:/PhD/4. TIM-8555/Week 5/train.csv')
test = pd.read_csv('C:/PhD/4. TIM-8555/Week 5/test.csv')

# Display the first few rows of the datasets
print("Train Dataset Sample:\n", train.head())
print("Test Dataset Sample:\n", test.head())

# Preprocessing: Encoding categorical variables
# Encode categorical columns like 'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', and 'NObeyesdad'
le = LabelEncoder()
for column in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']:
    train[column] = le.fit_transform(train[column])

# Splitting features and target variable
X_train = train.drop(columns=['id', 'NObeyesdad'])  # Features
y_train = train['NObeyesdad']  # Target variable

# Scale numerical features
scaler = StandardScaler()
X_train[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']] = scaler.fit_transform(X_train[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']])

# Define a cross-validation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# 1. Multinomial Logistic Regression
print("\nMultinomial Logistic Regression:")
log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_scores = cross_val_score(log_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(log_scores), np.std(log_scores)))
log_model.fit(X_train, y_train)
log_predictions = log_model.predict(X_train)
print(classification_report(y_train, log_predictions))

# 2. Linear Discriminant Analysis
print("\nLinear Discriminant Analysis:")
lda_model = LinearDiscriminantAnalysis()
lda_scores = cross_val_score(lda_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(lda_scores), np.std(lda_scores)))
lda_model.fit(X_train, y_train)
lda_predictions = lda_model.predict(X_train)
print(classification_report(y_train, lda_predictions))

# 3. Quadratic Discriminant Analysis
print("\nQuadratic Discriminant Analysis:")
qda_model = QuadraticDiscriminantAnalysis()
qda_scores = cross_val_score(qda_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(qda_scores), np.std(qda_scores)))
qda_model.fit(X_train, y_train)
qda_predictions = qda_model.predict(X_train)
print(classification_report(y_train, qda_predictions))

# 4. Naïve Bayes
print("\nNaïve Bayes:")
nb_model = BernoulliNB()
nb_scores = cross_val_score(nb_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(nb_scores), np.std(nb_scores)))
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_train)
print(classification_report(y_train, nb_predictions))

# 5. Support Vector Machine
print("\nSupport Vector Machine:")
svc_model = LinearSVC(max_iter=10000)
svc_scores = cross_val_score(svc_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(svc_scores), np.std(svc_scores)))
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_train)
print(classification_report(y_train, svc_predictions))
