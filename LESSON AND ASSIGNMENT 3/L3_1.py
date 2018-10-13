# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 08:31:09 2018

@author: Rene
"""

import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# load dataset
df = pandas.read_csv("history_export.csv", sep=";", engine="python")
#df= df.drop(df.index[:, 0:8], inplace=True) #dropping first 10 rows 
#df.drop(df.index[:11], inplace=True) #dropping first 10 rows 
df.drop(df.index[:10], inplace=True) #dropping first 9 rows coz they contain labels only

to_change = [c for c in df.columns if "46.6" in c] # numeric
# drop unwanted columns
# Notice that you need to specify inplace, otherwise pandas will return the data frame instead of changing it in place
#df.drop(to_remove, axis=1, inplace= True) 
# Change the target column data types
for c in to_change:
    df[c] = df[c].apply(lambda x: pandas.to_numeric(x))
#    
#Y=df.iloc[:,10]
#X=df.iloc[:,5:9]
#Y=Y.values
#X=X.values
#
#
#X_train=X[:round(0.7*len(X)),:]
#Y_train=Y[:round(0.7*len(Y))]
#
#X_test=X[round(0.7*len(X)):,:]
#Y_test=Y[round(0.7*len(Y)):]
#
#neigh = KNeighborsRegressor(n_neighbors=10)
#neigh.fit(X_train, Y_train) 
#predicted_outputs_test=neigh.predict(X_test)
#
#plt.plot(predicted_outputs_test,color='red',label='predicted_values')
#plt.plot(Y_test,color='yellow',label='actual values')
#plt.xlabel('Sample')
#plt.ylabel('Value')
#plt.title('SVM sample Regression')
#plt.legend()
#plt.show()
#
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#
#mse= mean_squared_error(Y_test, predicted_outputs_test)
#R2= r2_score(Y_test, predicted_outputs_test)
#
#print(mse)
#print(R2)