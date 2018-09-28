# -*- coding: utf-8 -*-

# Imports 
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path

'''
# Setting paths
cwd = os.getcwd()
'''
# Importing the dataset

'''
# Iris Dataset
dataset = pd.read_csv(path.join('..','data','iris.csv'))
X_columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
y_columns = ['iris']

'''

# Social Network Ads Dataset
dataset = pd.read_csv(path.join('..','data','Social_Network_Ads.csv'))
X_columns = [ 'Age', 'EstimatedSalary']
y_columns = ['Purchased']

# Extracting Features and Labels
dataset_feature = dataset[X_columns]
dataset_labels = dataset[y_columns]



# Test Train Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset_feature,dataset_labels,test_size=0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the classifier and Training it
from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=5,metric = 'minkowski', p = 2)
knnclf.fit( X =X_train ,y = y_train.values.ravel())

# Predicting based on trained classifier
y_pred = knnclf.predict(X=X_test)

# Obtaining the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test ,y_pred= y_pred)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(['r','g'])
x_axis = dataset[X_columns[0]]
y_axis = dataset[X_columns[1]]
plt.scatter(x_axis.values,y_axis.values)
plt.show()
x_axis_grid = np.arange(x_axis.min()-1,x_axis.max()+1,step = 0.1)
y_axis_grid = np.arange(y_axis.min()-1,y_axis.max()+1,step = 0.1)
XX,yy = np.meshgrid(x_axis_grid,y_axis_grid)
z = knnclf.predict(X=np.array([XX.ravel(),yy.ravel()]).T).reshape(XX.shape)
plt.contourf(XX,yy,z,cmap = cmap)
plt.show()

