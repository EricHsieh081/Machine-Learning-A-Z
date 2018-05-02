# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
# do not change the result, random_state = 0, ten trees
# if trees number goes up, it will become more accurate
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

y_prediction = regressor.predict(6.5)


#Decision Tree split the interval, and there are several trees, for each interval
#it would be split further into a lot of steps, from the ans of average of trees
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
