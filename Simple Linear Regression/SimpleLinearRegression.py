import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../Dataset/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# print(X)
# print(Y)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Visualising the Training set results

plt.scatter(X_train, Y_train , color='red')
plt.plot(X_train , regressor.predict(X_train) , color='blue')
plt.title('Salary vs Expirence (Training Set')
plt.xlabel('Years Of Exprince')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results

plt.scatter(X_test, Y_test , color='red')
plt.plot(X_train , regressor.predict(X_train) , color='blue')
plt.title('Salary vs Expirence (Test Set')
plt.xlabel('Years Of Exprince')
plt.ylabel('Salary')
plt.show()

# print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
# Salary=9345.94×YearsExperience+26816.19