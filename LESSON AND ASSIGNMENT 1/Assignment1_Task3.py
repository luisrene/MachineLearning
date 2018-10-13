# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:42:21 2018

@author: Rene
"""

from sklearn.svm import SVR
import pandas

dataframe=pandas.read_csv("weatherhistory.csv",sep=';',header=None)
#Year;Month;Day;Hour;Minute;Temperature  [2 m above gnd];Relative Humidity  [2 m above gnd];Total Precipitation  [sfc];Total Cloud Cover  [sfc];Shortwave Radiation  [sfc]
dataset=dataframe.values

#100% data
X_train=dataset[:,5:9]
Y_train=dataset[:,9]

svr_model=SVR(kernel='linear',C=1e3,gamma='auto')
svr_model.fit(X_train,Y_train)  # fitting data points in python

X_temp_test=input("Please enter Temperature data: ")
X_hum_test=input("Please enter Humidity data: ")
X_prec_test=input("Please enter Precipitation data: ")
X_cloud_test=input("Please enter Cloud Cover data: ")
X_test=[X_temp_test,X_hum_test,X_prec_test,X_cloud_test]

predicted_outputs_test=svr_model.predict(X_test)

print("Solar radiance is: "+str(predicted_outputs_test))