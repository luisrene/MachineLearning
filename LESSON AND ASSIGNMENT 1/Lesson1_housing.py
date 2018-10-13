from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import mean_squared_error

dataframe=pandas.read_csv("housing.csv",delim_whitespace=True,header=None)
dataset=dataframe.values
# Split our data into input (X) and output (Y)
X=dataset[:,0:13]
Y=dataset[:,13]

#Keeping 70% for training and remaining for testing
total_samples=len(Y)
trn_samples=round(0.7*total_samples)

X_trn=X[0:trn_samples]
Y_trn=Y[0:trn_samples]
X_tst=X[trn_samples:]
Y_tst=Y[trn_samples:]

svr_model=SVR(kernel='linear',C=10,gamma='auto')
svr_model.fit(X_trn,Y_trn)  # fitting data points in python
predicted_outputs=svr_model.predict(X_tst)

#plt.plot(predicted_outputs,color='red',label='predicted_values')
#plt.plot(Y_tst,color='yellow',label='actual values')
#plt.xlabel('Sample')
#plt.ylabel('Value')
#plt.title('SVM sample Regression')
#plt.legend()
#plt.show()

#  Find mean squared error
print ("Mean Squarred Error: ")
mse=mean_squared_error(Y_tst,predicted_outputs)
print(mse)

from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=7)
neigh.fit(X_trn, Y_trn) 
predicted_outputs_test=neigh.predict(X_tst)

plt.plot(predicted_outputs_test,color='red',label='predicted_values')
plt.plot(predicted_outputs,color='blue',label='SVR')
plt.plot(Y_tst,color='yellow',label='actual values')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('SVM sample Regression')
plt.legend()
plt.show()














