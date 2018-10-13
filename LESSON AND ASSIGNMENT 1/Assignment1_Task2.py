# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:28:44 2018

@author: Rene
"""

from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split

dataframe=pandas.read_csv("weatherhistory.csv",sep=';',header=None)
#Year;Month;Day;Hour;Minute;Temperature  [2 m above gnd];Relative Humidity  [2 m above gnd];Total Precipitation  [sfc];Total Cloud Cover  [sfc];Shortwave Radiation  [sfc]
dataset=dataframe.values

X=dataset[:,5:9]
Y=dataset[:,9]

#Hours=list(range(157968))
Days=dataset[:,2]
Temperature=dataset[:,5]
Humidity=dataset[:,6]
Precipitation=dataset[:,7]
Cloudcover=dataset[:,8]
Solarradiance=Y
"""
plt.plot(Temperature,color='yellow',label='Temperature')
plt.plot(Humidity,color='red',label='Humidity')
plt.plot(Precipitation,color='blue',label='Precipitation')
plt.plot(Cloudcover,color='brown',label='Cloudcover')
plt.plot(Solarradiance,color='cyan',label='Solar radiance')

plt.xlabel('Hours')
plt.ylabel('Measure')
plt.title('Weather History')
plt.legend()
plt.show()
#Note: Cloudcover and Solar radiance cover almost all the plot"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

svr_model=SVR(kernel='linear',C=10,gamma='auto')
svr_model.fit(X_train,Y_train)  # fitting data points in python
predicted_outputs_test=svr_model.predict(X_test)

SOS=0
for i in range(1,len(Y_test)):
    SOS=SOS+(Y_test[i]-predicted_outputs_test[i])**2
    
MSE=SOS/len(Y_test)
print(MSE)

RMSE=MSE**0.5
print(RMSE)

plt.plot(predicted_outputs_test,color='red',label='Predicted Radiance values')
plt.plot(Y_test,color='yellow',label='Actual values')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.title('SV sample Regression')
plt.legend()
plt.show()