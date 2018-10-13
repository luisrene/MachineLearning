# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:02:51 2018

@author: Rene
"""
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes() # Load the diabetes dataset
X=diabetes.data
Y=diabetes.target
X_train=X[:-20]
Y_train=Y[:-20]

X_test=X[-20:]
Y_test=Y[-20:]

svr_rbf = SVR(kernel= 'rbf', C= 1000, gamma= 8) # defining the support vector regression models
svr_rbf.fit(X_train, Y_train) # fitting the data points in the models
#### predictiong the same training data
predicted_outputs= svr_rbf.predict(X_train)
predicted_outputs_test= svr_rbf.predict(X_test)

plt.plot(predicted_outputs_test,color='red',label='predicted_values')
plt.plot(Y_test, color='yellow',label='Actual values')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('SVM Sample Regression')
plt.legend()
plt.show()

print("Mean squared error: "  )
mse= mean_squared_error(Y_test, predicted_outputs_test)
print(mse)