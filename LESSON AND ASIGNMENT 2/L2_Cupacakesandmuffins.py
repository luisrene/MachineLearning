# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 08:10:33 2018

@author: Rene
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


recipes=pd.read_csv("recipes_muffins_cupcakes.csv")

ingredients=recipes[['Flour','Sugar']].as_matrix()

type_label=np.where(recipes['Type']=='Muffin',1,0)
#model= LinearSVC(random_state=0)
model=svm.SVC(kernel='linear', C=1e5, degree=200, gamma=0.01)
model.fit(ingredients, type_label)

xx, yy=np.meshgrid(np.linspace(30,60), np.linspace(0,35))

Z=model.decision_function(np.c_[xx.ravel(),yy.ravel()])

Z=Z.reshape(xx.shape)

plt.scatter(ingredients[:,0], ingredients[:,1], c=type_label, cmap= plt.cm.Paired)
contours= plt.contour(xx, yy, Z, levels=[0])
plt.show()


X_train, X_test, Y_train, Y_test= train_test_split(ingredients,type_label,test_size=0.3,random_state=10)

model.fit(X_train, Y_train)

predicted_labels = model.predict(X_test)
print("SVC Accuracy: {0:.2%}".format(accuracy_score(predicted_labels, Y_test)))