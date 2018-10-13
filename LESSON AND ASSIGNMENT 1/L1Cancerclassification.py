# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:25:26 2018

@author: Rene
"""

from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
dataset = load_breast_cancer()

X= dataset.data
Y= dataset.target

## OR IF YOU WANT TO DIVIDE TO 70/30 RANDOMLY
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# Create the SVC model object
svm_model= svm.SVC(kernel='linear', C=100, gamma= 'auto') #gamma never affects linear kernel
svc = svm_model.fit(X_train, Y_train)
predicted_labels = svc.predict(X_test)
print("SVC Accuracy: {0:.2%}".format(accuracy_score(predicted_labels, Y_test)))
print(confusion_matrix(Y_test,predicted_labels)) 