#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:25:08 2018


"""

from pandas import read_table
from sklearn.preprocessing import normalize
import numpy as np
from sklearn import metrics
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from LinearRegression import LinearRegressionGradientMethod
import statsmodels.api as sm
import seaborn as sns
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib import pylab
from numpy import arange,array,ones
from scipy import stats

## Data Preparation

dataFileName = "movehubqualityoflife_costofliving.csv"

'''
Takes In:
    CSV file

Returns:
    A DataFrame data struction of the passed CSV File.
    
'''
def download_data(fileLocation):
    frame = read_table(fileLocation,encoding='latin-1',sep=',',skipinitialspace=True,index_col=None,header=None)
    return frame



'''
Takes In:
0: City, 1: Movehub Rating, 2: Purchase Power, 3: Health Care, 4: Pollution, 5: Quality of Life, 6: Crime Rating,
7: City, 8: Cappuccino, 9: Cinema, 10: Wine, 11: Gasoline, 12: Avg Rent, 13: Avg Disposable Income

Returns:
    A data matrix without City names, without Column names, and is NOT normalized. 
    0: Movehub Rating, 1: Purchase Power, 2: Health Care, 3: Pollution, 4: Quality of Life, 5: Crime Rating,
    6: Cappuccino, 7: Cinema, 8: Wine, 9: Gasoline, 10: Avg Rent, 11: Avg Disposable Income
'''
def data_preparation(data):
    new_data = []
    new_data = data.iloc[1:]
    new_data = new_data.drop(new_data.columns[[0]], 1)
    return new_data

def get_Column_Names(data):
    new_data = []
    data = data.drop(data.columns[[0,1]], 1)
    new_data = data.iloc[0,:]
    return new_data


'''
Takes In:
    Stripped Data (No row/column Labels)

Returns:
    A data matrix with normalized columns based off of max column values.
'''
def data_normalization(data, method='max'):
    new_data = normalize(data, axis=0, norm=method)
    return new_data

raw_data = download_data(dataFileName)              # Gets data from file
prepared_data = data_preparation(raw_data)          # Cuts unwanted Data
prepared_data = data_normalization(prepared_data)   # Normalizes all Columns
column_names = get_Column_Names(raw_data)

## END of Data Preparation

## Data Splitting
total_samples = 216
total_training = 100
total_validating = 50
total_testing = total_samples - (total_training + total_validating)

S = np.random.permutation(total_samples)
X_train = prepared_data[S[:total_training], 1:(len(prepared_data[0]))]
Y_train = prepared_data[S[:total_training], :1]
X_validation = prepared_data[S[total_training:(total_training + total_validating)], 1:(len(prepared_data[0]))]
Y_validation = prepared_data[S[total_training:(total_training + total_validating)], :1]
X_test = prepared_data[S[(total_training + total_validating):total_samples], 1:(len(prepared_data[0]))]
Y_test = prepared_data[S[(total_training + total_validating):total_samples], :1]
## END Data Splitting
## Linear Regression, Gradient Decent



## Linear Regression, Gradient Decent
LRegress = LinearRegressionGradientMethod()
arrCost = LRegress.getGradient(X_train,Y_train)
LRegress.showCostGraph(arrCost)
theta = LRegress.getTheta()
theta = theta[len(theta)-1]
tVal = X_test.dot(theta)
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(X_train,Y_train)])
print('gradientdecent: {} ({})'.format(np.mean(tError), np.std(tError)))
print("gradientdecent Mean Absolute Error: ",metrics.mean_absolute_error(Y_test,tVal))
print("gradientdecent Mean Squared Error: ",metrics.mean_squared_error(Y_test,tVal))
print("gradientdecent: SQRT(Mean Squared Error:)",np.sqrt(metrics.mean_squared_error(Y_test,tVal)))
sns.regplot(Y_test,tVal, data=prepared_data, label='gradientdecent')

## END Linear Regression, Gradient Decent

##Linear Regression, statsmodel
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model
model.summary()
print(model.summary())

sns.regplot(Y_test,predictions, data=prepared_data, label='statsmodel',color='green')
#plt.legend(loc='upper right')
#plt.show()

print("Stat Model Mean Absolute Error::",metrics.mean_absolute_error(Y_test,predictions))
print("Stat Model Mean Squared Error:",metrics.mean_squared_error(Y_test,predictions))
print("Stat Model SQRT(Mean Squared Error):",np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
##END Linear Regression, statsmodel

##Linear Regression, Scikit-learn 
model = LinearRegression()
model.fit(X_train, Y_train)
predictions1=model.predict(X_test)

print("Scikit-learn Mean Absolute Error:",metrics.mean_absolute_error(Y_test,predictions1))
print("Scikit-learn Mean Squared Error:",metrics.mean_squared_error(Y_test,predictions1))
print("Scikit-learn SQRT(Mean Squared Error): ",np.sqrt(metrics.mean_squared_error(Y_test,predictions1)))

#sns.regplot(Y_test,predictions1,data=prepared_data,label='Scikit-learn',color='red')
#plt.legend(loc='upper right')
##END Linear Regression, Scikit-learn 

##Linear Regression, KNN
print("KNN model")
knn = neighbors.KNeighborsRegressor(10)
y_predict_knn = knn.fit(X_train, Y_train).predict(X_test)
regression_model_mse_knn = mean_squared_error(y_predict_knn, Y_test)

print("Knn Mean Absolute Error::",metrics.mean_absolute_error(Y_test,y_predict_knn))
print("Knn Stat Model Mean Squared Error:",metrics.mean_squared_error(Y_test,y_predict_knn))
print("Knn Stat Model SQRT(Mean Squared Error):",np.sqrt(metrics.mean_squared_error(Y_test,y_predict_knn)))

print(math.sqrt(regression_model_mse_knn))
print(" ")
#sns.regplot(Y_test,y_predict_knn,data=prepared_data,label='KNN',color='purple')
##END Linear Regression, KNN


