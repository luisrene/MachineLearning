# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 22:10:12 2018

@author: Rene
"""

from sklearn import datasets, svm, grid_search
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import LabelEncoder


dataframe=pd.read_csv("BlackFriday.csv", sep=',', header=None)
dataframe.drop([9,10], axis=1,inplace=True)
dataframe[0], tmp_indexer = pd.factorize(dataframe[0])
dataframe[1], tmp_indexer = pd.factorize(dataframe[1])
dataframe[2], tmp_indexer = pd.factorize(dataframe[2])
dataframe[3], tmp_indexer = pd.factorize(dataframe[3])
dataframe[4], tmp_indexer = pd.factorize(dataframe[4])
dataframe[5], tmp_indexer = pd.factorize(dataframe[5])
dataframe[6], tmp_indexer = pd.factorize(dataframe[6])
dataframe.drop([0], axis=0,inplace=True)

#pd.to_numeric(dataframe, errors='coerce')

#User_ID=dataframe.iloc[:,0]
#pd.to_numeric(User_ID, errors='coerce')

#Experiment 1
dataframeexp1=dataframe.copy()
Y=dataframeexp1.iloc[:,7]
dataframeexp1.drop([7], axis=1,inplace=True)

X=dataframeexp1
X.fillna(0)
X.astype('int64')
#
##skf = StratifiedKFold(n_splits=10)
##for train, test in skf.split(X, Y):
##    print("%s %s" % (train, test))


##    
###    
##parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
##svm = svm.SVC()
##clf = grid_search.GridSearchCV(svm, parameters)
##clf.fit(X, Y)
##print(clf.best_params_)
####
##scores = cross_val_score(clf, X, Y, cv=10)
##avg=(scores.mean())*100
##print(avg)
##
##
###predicted = cross_val_predict(neigh, X, Y)
#
#
#
#Experiment 2
dataframeexp2=dataframe.copy()
Y1=dataframeexp2.iloc[:,4]
dataframeexp2.drop([4], axis=1,inplace=True)

X1=dataframeexp2
X1.fillna(0)
X1.astype('int64')
#
##skf = StratifiedKFold(n_splits=10)
##for train, test in skf.split(X, Y):
##    print("%s %s" % (train, test))

##    
###    
##parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
##svm = svm.SVC()
##clf = grid_search.GridSearchCV(svm, parameters)
##clf.fit(X, Y)
##print(clf.best_params_)
####
##scores = cross_val_score(clf, X, Y, cv=10)
##avg=(scores.mean())*100
##print(avg)
##
##
###predicted = cross_val_predict(neigh, X, Y)
#
#
#
#
#
##Experiment 3
dataframeexp3=dataframe.copy()
Y2=dataframeexp3.iloc[:,3]
dataframeexp3.drop([3], axis=1,inplace=True)

X2=dataframeexp3
X2.fillna(0)
X2.astype('int64')
#
##skf = StratifiedKFold(n_splits=10)
##for train, test in skf.split(X, Y):
##    print("%s %s" % (train, test))

##    
###    
##parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
##svm = svm.SVC()
##clf = grid_search.GridSearchCV(svm, parameters)
##clf.fit(X, Y)
##print(clf.best_params_)
####
##scores = cross_val_score(clf, X, Y, cv=10)
##avg=(scores.mean())*100
##print(avg)
##
##
###predicted = cross_val_predict(neigh, X, Y)
#
#
#
##Experiment 4
dataframeexp4=dataframe.copy()
Y4=dataframeexp4.iloc[:,9]
Y4.astype('int64')

for x in range (1,len(Y4)):
    if  Y4[x] >= 1000:
        Y4[x]=1
    else:
        Y4[x]=0
        
dataframeexp3.drop([3], axis=1,inplace=True)

X2=dataframeexp3
X2.fillna(0)
X2.astype('int64')
#
##skf = StratifiedKFold(n_splits=10)
##for train, test in skf.split(X, Y):
##    print("%s %s" % (train, test))

##    
###    
##parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
##svm = svm.SVC()
##clf = grid_search.GridSearchCV(svm, parameters)
##clf.fit(X, Y)
##print(clf.best_params_)
####
##scores = cross_val_score(clf, X, Y, cv=10)
##avg=(scores.mean())*100
##print(avg)
##
##
###predicted = cross_val_predict(neigh, X, Y)
  