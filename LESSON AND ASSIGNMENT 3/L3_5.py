# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:27:40 2018

@author: Rene
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, grid_search
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# load dataset
df = pd.read_csv('cwurData.csv')
df=df.dropna(axis='columns')

df.drop(['world_rank', 'year', 'institution', 'national_rank','score'], axis = 1,inplace=True)
df['country'], tmp_indexer = pd.factorize(df['country'])
#
#considered_params= ['country', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations', 'patents']
#X = df[considered_params].copy()
#X_orig= df[considered_params].copy()
#X=X.values
#
#considered_params= ['quality_of_education']
#Y = df[considered_params].copy()
#Y=Y.values
#
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
#
#regr_1 = DecisionTreeRegressor(max_depth=20)
#regr_1= regr_1.fit(X_train, Y_train)
#predicted_outputs_test= regr_1.predict(X_test)
#
#
##plt.plot(predicted_outputs_test,color='red',label='predicted_values')
##plt.plot(Y_test,color='yellow',label='actual values')
##plt.xlabel('Sample')
##plt.ylabel('Value')
##plt.title('KNN sample Regression')
##plt.legend()
##plt.show()
##
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#
#importances= regr_1.feature_importances_
#print(importances)
#
#mse= mean_squared_error(Y_test, predicted_outputs_test)
#R2= r2_score(Y_test, predicted_outputs_test)
#
#print(mse)
#print(R2)
#
## Sort feature importances in descending order
#indices = np.argsort(importances)[::-1]
#
#considered_params= list(X_orig.columns.values)
## Rearrange feature names so they match the sorted feature importances
#input_names= considered_params#[0:-1]
#names = [input_names[i] for i in indices]
#
## Create plot
#plt.figure()
#
## Create plot title
#plt.title("Feature Importance")
#
## Add bars
#plt.bar(range(X.shape[1]), importances[indices])
#
## Add feature names as x-axis labels
#plt.xticks(range(X.shape[1]), names, rotation=90)
#
## Show plot
#plt.show()
#
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#
#
#mse= mean_squared_error(Y_test, predicted_outputs_test)
#R2= r2_score(Y_test, predicted_outputs_test)
#
#print(mse)
#print(R2)