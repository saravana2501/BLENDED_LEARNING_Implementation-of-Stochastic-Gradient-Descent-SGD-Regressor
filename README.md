Skip to content
Navigation Menu
aaliyafathimaa
BLENDED_LEARNING_Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

Type / to search
Code
Pull requests
Actions
Projects
Security
Insights
Owner avatar
BLENDED_LEARNING_Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor
Public
forked from AkilaMohan/BLENDED_LEARNING_Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor
aaliyafathimaa/BLENDED_LEARNING_Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor
Go to file
t
This branch is 1 commit ahead of AkilaMohan/BLENDED_LEARNING_Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor:main.
Name		
aaliyafathimaa
aaliyafathimaa
Update README.md
173c482
 · 
6 months ago
LICENSE
Initial commit
7 months ago
README.md
Update README.md
6 months ago
Repository files navigation
README
License
BLENDED_LEARNING
Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor
AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Import the necessary libraries.
Load the dataset.
Preprocess the data (handle missing values, encode categorical variables).
Split the data into features (X) and target (y).
Divide the data into training and testing sets.
Create an SGD Regressor model.
Fit the model on the training data.
Evaluate the model performance.
Make predictions and visualize the results.
Program:
/*
Program to implement SGD Regressor for linear regression.
Developed by: Saravana Kumar S
RegisterNumber:  212224220090
*/
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
# Define target variable (y) and features (X)
y = df['price']
X = df.drop(['price'], axis=1)

# Print the shape of X and y
print(X.shape, y.shape)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the training and test sets
print(X_train.shape, X_test.shape)
# Standardize the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize and train the SGD Regressor model
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_regressor.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = sgd_regressor.predict(X_test_scaled)
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}")
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}")
# Plot actual vs predicted car prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()
Output:
Screenshot 2024-10-06 193048

Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.


