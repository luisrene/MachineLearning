# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:49:54 2018

@author: Rene
"""

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy

dataset=datasets.load_iris()

X=dataset.data
Y=dataset.target

#Test 1: First 70%
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.3)

# Create the SVM model object
svm_model= svm.SVC(kernel='rbf', C=100, gamma= 'auto') #gamma never affects linear kernel
svm = svm_model.fit(X_train, Y_train)
predicted_labels = svm.predict(X_test)

Good=0;Error=0;
for i in range(1,len(Y_test)):
    if predicted_labels[i] == Y_test[i]:
        Good=Good+1
    else:
        Error=Error+1
        
        Accuracy=(Good/(Good+Error))*100
print (Accuracy)


#Test 2: 70% from the medium

row15=round(0.15*len(X))
row85=round(0.85*len(X))

X_train2=X[row15:row85,:]
Y_train2=Y[row15:row85]

X_test2a=X[0:row15-1,:]
X_test2b=X[row85+1:len(X),:]
X_test2=numpy.concatenate((X_test2a,X_test2b),axis=0)

Y_test2a=Y[0:row15-1]
Y_test2b=Y[row85+1:len(X)]
Y_test2=numpy.concatenate((Y_test2a,Y_test2b),axis=0)

svm2 = svm_model.fit(X_train2, Y_train2)
predicted_labels2 = svm2.predict(X_test2)

Good=0;Error=0;
for i in range(1,len(Y_test2)):
    if predicted_labels2[i] == Y_test2[i]:
        Good=Good+1
    else:
        Error=Error+1
        
        Accuracy=(Good/(Good+Error))*100
print (Accuracy)


#Test 3: 70% from the end
row30=round(0.3*len(X))
X_train3=X[row30:,:]
Y_train3=Y[row30:]

X_test3=X[:row30,:]
Y_test3=Y[:row30]

svm3 = svm_model.fit(X_train3, Y_train3)
predicted_labels3 = svm3.predict(X_test3)

Good=0;Error=0;
for i in range(1,len(Y_test3)):
    if predicted_labels3[i] == Y_test3[i]:
        Good=Good+1
    else:
        Error=Error+1
        
        Accuracy=(Good/(Good+Error))*100
print (Accuracy)