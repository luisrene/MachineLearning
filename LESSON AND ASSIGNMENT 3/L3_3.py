# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:39:32 2018

@author: Rene
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv('cwurData.csv')
df=df.dropna(axis='columns')

df.drop(['world_rank', 'year', 'institution', 'country', 'national_rank'], axis = 1,inplace=True)
X=df.iloc[:,0:7]
X=X.values
Y=df.iloc[:, 7]
Y=Y.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.05)

neigh = KNeighborsRegressor(n_neighbors=1)
neigh.fit(X_train, Y_train) 
predicted_outputs_test=neigh.predict(X_test)

plt.plot(predicted_outputs_test,color='red',label='predicted_values')
plt.plot(Y_test,color='yellow',label='actual values')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('KNN sample Regression')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

importances= neigh.feature_importances_
print(importances)

mse= mean_squared_error(Y_test, predicted_outputs_test)
R2= r2_score(Y_test, predicted_outputs_test)

print(mse)
print(R2)