# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:13:04 2018

@author: Rene
"""

from sklearn import datasets, svm, grid_search
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

dataframe=pandas.read_csv("insurance.csv",header=None)

#Manage categorical values into fixed integers
#nums= {"sex":  {"female": 1,"male": 0},
            # "smoker": {"yes": 0, "no": 1},
             #"region": {"northeast": 0, "northwest":1, "southeast":2, "southwest":3} } 
             
#dataframe.replace(nums, inplace=True)

# Encoding categorical data
labelencoder_X = LabelEncoder()

#  I changed this line
dataframe.values[1:1338, 1] = labelencoder_X.fit_transform(dataframe.values[1:1338, 1])
dataframe.values[1:1338, 4] = labelencoder_X.fit_transform(dataframe.values[1:1338, 4])
dataframe.values[1:1338, 5] = labelencoder_X.fit_transform(dataframe.values[1:1338, 5])
print(dataframe.head())

X=dataframe.iloc[1:1338,0:5]
Y=dataframe.iloc[1:1338,6]

#Split data 80/20%
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.3)


#Find best parameters for model - I GOT AN ERROR WHEN TRYING TO FIND BEST PARAMETERS
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100]}
#svm = svm.SVC()
#clf = grid_search.GridSearchCV(svm, parameters)
#clf.fit(X_train, Y_train)
#print(clf.best_params_)



# Create the SVM model object
svm_model= svm.SVC(kernel='rbf', C=100, gamma= 'auto') #gamma never affects linear kernel
svm = svm_model.fit(X_train, Y_train)
predicted_labels = svm.predict(X_test)
predicted_labels=numpy.array(predicted_labels, dtype=float)
Y_test=numpy.array(Y_test,dtype=float)

plt.plot(predicted_labels,color='red',label='Predicted values')
plt.plot(Y_test,color='yellow',label='Actual values')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('SVM sample Regression')
plt.legend()
plt.show()

mse=mean_squared_error(Y_test,predicted_labels)
print (mse)
r2=r2_score(Y_test,predicted_labels)
print(r2)
