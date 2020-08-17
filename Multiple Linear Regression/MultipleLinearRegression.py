import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('../Dataset/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), axis=1))

# Visualising the Training set results

# plt.scatter(X_train, Y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Expirence (Training Set')
# plt.xlabel('Years Of Exprince')
# plt.ylabel('Salary')
# plt.show()
#
# # Visualising the Test set results
#
# plt.scatter(X_test, Y_test, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Expirence (Test Set')
# plt.xlabel('Years Of Exprince')
# plt.ylabel('Salary')
# plt.show()
