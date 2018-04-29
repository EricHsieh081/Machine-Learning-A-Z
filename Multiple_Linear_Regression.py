#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:39:26 2018

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encode the categorial data
labelencoderx = LabelEncoder()
X[:,3] = labelencoderx.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Eliminate one of the dummy variables
X = X[:,1:]

#cross-validation and train data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#predict the test data
prediction = regressor.predict(xtest)


# next is for backward elimination
import statsmodels.formula.api as smodel
#add the x0 from b0*x0 for the feature, x0 is always 1, b0*x0 is at the head of function
#y = b0*x0 + b1*x1 ...
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# X_ideal means the features that have high impact on prediction

#auto tune ideal train feature
newX = X[:,[0, 1, 2, 3, 4, 5]]
X_ideal = newX[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()
regressor_OLS.summary()
type(regressor_OLS._results.pvalues)
SL_value = 0.05
flag = True
level = 0

while flag == True:
    print(regressor_OLS._results.pvalues)
    level+=1
    flag = False
    max_overflow = 0.0
    max_index = -1
    #print(regressor_OLS._results.pvalues.shape)
    for index, p_value in enumerate(regressor_OLS._results.pvalues):
        if p_value > SL_value:
            #print("find ", level," ", p_value)
            flag = True
            if p_value > max_overflow:
                max_overflow = p_value
                max_index = index
    
    if max_index >= 0:
        ask = list()
        for i in range(0, regressor_OLS._results.pvalues.shape[0]):
            if i != max_index:
                ask.append(i)
        print(ask)
        X_ideal = newX[:, ask]
        newX = newX[:, ask]
    
    regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()

        
# human tune ideal train feature
# =============================================================================
# X_ideal = X[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()
# regressor_OLS.summary()
# X_ideal = X[:, [0, 1, 3, 4, 5]]
# print(regressor_OLS._results.pvalues)
# regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()
# regressor_OLS.summary()
# X_ideal = X[:, [0, 3, 4, 5]]
# print(regressor_OLS._results.pvalues)
# regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()
# regressor_OLS.summary()
# print(regressor_OLS._results.pvalues)
# X_ideal = X[:, [0, 3, 5]]
# regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()
# regressor_OLS.summary()
# print(regressor_OLS._results.pvalues)
# X_ideal = X[:, [0, 3]]
# regressor_OLS = smodel.OLS(endog = y, exog = X_ideal).fit()
# regressor_OLS.summary()
# print(regressor_OLS._results.pvalues)
# regressor_OLS._results
# =============================================================================
