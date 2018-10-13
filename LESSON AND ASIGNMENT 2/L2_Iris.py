# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 09:06:50 2018

@author: Rene
"""
from sklearn import datasets, svm, grid_search
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data # we only take the first two features.
Y = iris.target
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.3,random_state=10)

#
parameters = {'n_neighbors':[1, 10]}
neigh = KNeighborsClassifier()
clf = grid_search.GridSearchCV(neigh, parameters)
clf.fit(X_train,Y_train)
print(clf.best_params_)


neigh = KNeighborsClassifier(n_neighbors=1)
predicted_labels= clf.predict(X_test)
print(accuracy_score(predicted_labels, Y_test))

from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(Y_test,predicted_labels))  
print(classification_report(Y_test,predicted_labels)) 


#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svm = svm.SVC()
#clf = grid_search.GridSearchCV(svm, parameters)
#clf.fit(iris.data, iris.target)
#print(clf.best_params_)
#
#scores = cross_val_score(clf, iris.data, iris.target, cv=10)
#avg=(scores.mean())*100
#print(avg)
#
skf = KFold(n_splits=3)
ctr=0 
for train, test in skf.split(X, Y):
    ctr= ctr+ 1
    print ("fold#")
    print(ctr)
    print("%s %s" % (train, test))
##predicted = cross_val_predict(neigh, X, Y)
#
#skf = StratifiedKFold(n_splits=3)
#for train, test in skf.split(X, Y):
#    print("%s %s" % (train, test))
#    
#
#
##model=svm.SVC(kernel='rbf', C=1e3, gamma=3)
###model=svm.SVC(kernel='RBF', C=1e5, degree=200, gamma=0.01)
##model.fit(X, Y)
##
##X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.3,random_state=10)
##
#model.fit(X_train, Y_train)
#
#predicted_labels = model.predict(X_test)
#print("SVC Accuracy: {0:.2%}".format(accuracy_score(predicted_labels, Y_test)))
#
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#h = (x_max / x_min)/100
#
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#np.arange(y_min, y_max, h));X_plot = np.c_[xx.ravel(), yy.ravel()]
#
#Z = model.predict(X_plot);Z = Z.reshape(xx.shape)
#plt.figure(figsize=(15, 5));plt.subplot(121)
#plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
#plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Set1)
#plt.xlabel('Sepal length');plt.ylabel('Sepal width')
#plt.xlim(xx.min(), xx.max());plt.title('SVC with linear kernel')
#
## Create the SVC model object
#svm_model_poly= svm.SVC(kernel='poly', C= 10, degree=5)
#svc = svm_model_poly.fit(X_train, Y_train)
#
#Z = svc.predict(X_plot);Z = Z.reshape(xx.shape)
#
#plt.subplot(122)
#plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
#plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Set1)
#plt.xlabel('Sepal length');plt.ylabel('Sepal width')
#plt.xlim(xx.min(), xx.max());plt.title('SVC with Polynomial kernel of degree 5 ')
#plt.show()
#
#
#predicted_labels_linear = model.predict(X_test)
#print("SVC Accuracy Linear: {0:.2%}".format(accuracy_score(predicted_labels_linear, Y_test)))
#
#predicted_labels_poly = svm_model_poly.predict(X_test)
#print("SVC Accuracy  Polynomial: {0:.2%}".format(accuracy_score(predicted_labels_poly, Y_test)))