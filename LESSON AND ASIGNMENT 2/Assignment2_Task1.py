# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 21:49:28 2018

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


dataframe=pandas.read_csv("BlackFriday.csv", sep=',', header=None)
#Check data types and select only objects
dataframe.dtypes
obj_dataframe = dataframe.select_dtypes(include=['object']).copy()
obj_dataframe.head()


#Manage Product category 2 & 3
#Eliminate null values
modifieddataframe=obj_dataframe.fillna(0)
modifieddataframe.isnull().sum()
modifieddataframe.to_csv('modifiedBlackFriday.csv',index=False)

#Replace rows with most common value
Product_cat_2=modifieddataframe.iloc[1:537578,9]

Product_cat_2[Product_cat_2=='1']=1
Product_cat_2[Product_cat_2=='2']=2
Product_cat_2[Product_cat_2=='3']=3
Product_cat_2[Product_cat_2=='4']=4
Product_cat_2[Product_cat_2=='5']=5
Product_cat_2[Product_cat_2=='6']=6
Product_cat_2[Product_cat_2=='7']=7
Product_cat_2[Product_cat_2=='8']=8
Product_cat_2[Product_cat_2=='9']=9
Product_cat_2[Product_cat_2=='10']=10
Product_cat_2[Product_cat_2=='11']=11
Product_cat_2[Product_cat_2=='12']=12
Product_cat_2[Product_cat_2=='13']=13
Product_cat_2[Product_cat_2=='14']=14
Product_cat_2[Product_cat_2=='15']=15
Product_cat_2[Product_cat_2=='16']=16
Product_cat_2[Product_cat_2=='17']=17
Product_cat_2[Product_cat_2=='18']=18


MCV1=Counter(Product_cat_2)
print(MCV1)

#Replace array Product_cat_2 with most common value
Product_cat_2[Product_cat_2==0]=8

Product_cat_3=modifieddataframe.iloc[1:537578,10]

Product_cat_3[Product_cat_3=='1']=1
Product_cat_3[Product_cat_3=='2']=2
Product_cat_3[Product_cat_3=='3']=3
Product_cat_3[Product_cat_3=='4']=4
Product_cat_3[Product_cat_3=='5']=5
Product_cat_3[Product_cat_3=='6']=6
Product_cat_3[Product_cat_3=='7']=7
Product_cat_3[Product_cat_3=='8']=8
Product_cat_3[Product_cat_3=='9']=9
Product_cat_3[Product_cat_3=='10']=10
Product_cat_3[Product_cat_3=='11']=11
Product_cat_3[Product_cat_3=='12']=12
Product_cat_3[Product_cat_3=='13']=13
Product_cat_3[Product_cat_3=='14']=14
Product_cat_3[Product_cat_3=='15']=15
Product_cat_3[Product_cat_3=='16']=16
Product_cat_3[Product_cat_3=='17']=17
Product_cat_3[Product_cat_3=='18']=18

#MCV2=Counter(Product_cat_3)
#print(MCV2)

#Replace array Product_cat_3 with most common value
Product_cat_3[Product_cat_3==0]=16

# Encoding categorical data
labelencoder_X = LabelEncoder()

modifieddataframe.values[1:537577, 2] = labelencoder_X.fit_transform(modifieddataframe.values[1:537577, 2])
modifieddataframe.values[1:537577, 3] = labelencoder_X.fit_transform(modifieddataframe.values[1:537577, 3])
modifieddataframe.values[1:537577, 5] = labelencoder_X.fit_transform(modifieddataframe.values[1:537577, 5])
modifieddataframe.values[1:537577, 6] = labelencoder_X.fit_transform(modifieddataframe.values[1:537577, 6])
print(modifieddataframe.head())

Gender=modifieddataframe.iloc[1:537578,2] 
Gender=pandas.to_numeric(Gender, errors='coerce')

Age=modifieddataframe.iloc[1:537578,3] 
Age=pandas.to_numeric(Age, errors='coerce')

City_Category=modifieddataframe.iloc[1:537578,5] 
City_Category=pandas.to_numeric(City_Category, errors='coerce')

Stay_In=modifieddataframe.iloc[1:537578,6] 
Stay_In=pandas.to_numeric(Stay_In, errors='coerce')

#Manage Product_ID
Product_ID=modifieddataframe.iloc[1:537578,1]    
Product_ID = [e[1:] for e in Product_ID]
Product_ID=pandas.to_numeric(Product_ID, errors='coerce')

print (Product_ID)

#Convert to fixed integers
#Manage User_ID
User_ID=modifieddataframe.iloc[1:537578,0]
User_ID=pandas.to_numeric(User_ID, errors='coerce')

#Manage Occupation
Occupation=modifieddataframe.iloc[1:537578,4]
Occupation=pandas.to_numeric(Occupation, errors='coerce')

#Manage Marital Status
MaritalStat=modifieddataframe.iloc[1:537578,7]
MaritalStat=pandas.to_numeric(MaritalStat, errors='coerce')

#Manage Product_Category_1
Product_cat_1=modifieddataframe.iloc[1:537578,8] 
Product_cat_1=pandas.to_numeric(Product_cat_1, errors='coerce')

#Manage Purchase
Purchase=modifieddataframe.iloc[1:537578,11]
Purchase=pandas.to_numeric(Purchase, errors='coerce')

NewDataFrame=pandas.concat([User_ID, Gender, Age, Occupation, City_Category, Stay_In, MaritalStat, Product_cat_1, Product_cat_2, Product_cat_3, Purchase], axis=1)

NewDataFrame.convert_objects(convert_numeric=True)
a=NewDataFrame.iloc[:,0]
b=NewDataFrame.iloc[:,1:11]
FinalDataFrame=pandas.concat([a,Product_ID], axis=1)

print(FinalDataFrame)
#Product_cat_3=int(modifieddataframe.iloc[:,10])


#Xprod=int(dataframe.iloc[:,1])
#X1=dataframe.iloc[:,0:7]
#X2=dataframe.iloc[:,8:12]
#X=pandas.concat((X1,X2),axis=1)
#Y=dataframe.iloc[:,7]



#skf = KFold(n_splits=10)
#ctr=0 
#for train, test in skf.split(X, Y):
#    ctr= ctr+ 1
#    print ("fold#")
#    print(ctr)
#    print("%s %s" % (train, test))
#    
#    
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svm = svm.SVC()
#clf = grid_search.GridSearchCV(svm, parameters)
#clf.fit(X, Y)
#print(clf.best_params_)
#
#scores = cross_val_score(clf, X, Y, cv=10)
#avg=(scores.mean())*100
#print(avg)


#predicted = cross_val_predict(neigh, X, Y)

#skf = StratifiedKFold(n_splits=3)
#for train, test in skf.split(X, Y):
#    print("%s %s" % (train, test))
    