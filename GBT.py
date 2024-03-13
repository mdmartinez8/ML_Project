import sys
import os
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

X = pd.read_csv('fraudTrain.csv')
Y = pd.read_csv('fraudTest.csv')
# Prep the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize Models
# For classification
modelC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# For regression
modelR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# Train models
modelC.fit(X_train, Y_train)
modelR.fit(X_train, Y_train)

# Make predictions
predictions_R = modelR.predict(X_test)
predictions_C = modelC.predict(X_test)

# For classification
accuracy = accuracy_score(Y_test, predictions_C)
print(f"Accuracy: {accuracy}")

# For regression
mse = mean_squared_error(Y_test, predictions_R)
print(f"Mean Squared Error: {mse}")


# Make predictions on new data
#new_predictions = model.predict(new_data)


print("Yoooo")
