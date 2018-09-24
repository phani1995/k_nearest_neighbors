# -*- coding: utf-8 -*-


#Imports 
import pandas as pd
#import matplotlib.pyplot as plt

# Iris Dataset
# Importing the dataset
dataset = pd.read_csv('iris.csv')

# Extracting Features and Labels
dataset_feature = dataset.drop(labels = ['iris'],axis = 1)
dataset_labels = dataset['iris']

# Test Train Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset_feature,dataset_labels,test_size=0.25)

# Importing the classifier and Training it
from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier()
knnclf.fit(X =X_train.values ,y =y_train.values)

# Predicting based on trained classifier
y_pred = knnclf.predict(X=X_test.values)

# Obtaining the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test ,y_pred= y_pred)

