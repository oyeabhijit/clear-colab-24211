# -*- coding: utf-8 -*-

# -- Sheet --

# # Simple Linear Regression


# # Importing the Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Importing the dataset


dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
dataset.head()

# # Splitting the dataset into Train set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# # Training the Simple Linear Regression model on the Training set


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# # Predicting the Test set results


y_pred=regressor.predict(X_test)

# # Visualising the Training set results
# - We will use the matplotlib library. To be exact pyplot for plotting our training set reults.


plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('SALARY vs EXPERIENCE (Training Set)')
plt.xlabel('EXPERIENCE (Years)')
plt.ylabel('SALARY (USD)')

# # Visualising the Test set results
# - We will use the matplotlib library. To be exact pyplot for plotting our training set reults.


plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('SALARY vs EXPERIENCE (Test Set)')
plt.xlabel('EXPERIENCE (Years)')
plt.ylabel('SALARY (USD)')

