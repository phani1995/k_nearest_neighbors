# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# Imports 
import numpy as np
import pandas as pd
import math

'''
x_cordinates  = pd.Series(data = [11,1,2,1,8,9,10])
y_cordinates  = pd.Series(data = [11,1,1,3,8,10,9])
group = pd.Series(data = ['b','a','a','a','b','b','b'])
dataset = pd.concat([x_cordinates,y_cordinates,group],axis=1)
X_labels = [0,1]
y_labels =[2]
'''

# Iris Dataset
dataset = pd.read_csv('iris.csv')
x_labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
y_labels = ['iris']

# Shuffling the dataframe
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Creating features and Labels
X_dataset = dataset.drop(labels = y_labels,axis=1)
y_dataset = dataset[y_labels]

# Splitting the dataset into test and train sets
train_test_split_ratio = 0.75
X_train,X_test = X_dataset[:int(len(X_dataset)*train_test_split_ratio)],X_dataset[int(len(X_dataset)*train_test_split_ratio):]
y_train,y_test = y_dataset[:int(len(y_dataset)*train_test_split_ratio)],y_dataset[int(len(y_dataset)*train_test_split_ratio):]


# Function to calculate euclidean distance
def euclidian_distance(feature_1,feature_2):
    '''
    This function takes two fearuture vecotrs as input and caluculates the euclidian distance between them
    Args: feature_1 a numpy array of features
          feature_2 a numpy array of features
    returns: a float value of distance between the features
    '''
    sum_of_squares = 0
    for i in range(0,len(feature_1)):
        sum_of_squares = sum_of_squares + math.pow(feature_2[i]-feature_1[i],2)
    distance = math.sqrt(sum_of_squares)
    return distance


y_pred = []
for index_x_test,row_x_test in X_test.iterrows():
    # Iterating over all the features
    nearest_neighbors = []

    for index_x_train,row_x_train in X_train.iterrows():      
        dist = euclidian_distance(list(row_x_test),list(row_x_train))
        nearest_neighbors.append([dist,y_train.values[index_x_train]])
        
    # Sorting neighbors with least distance
    nearest_neighbors.sort(key = lambda ele : ele[0])
    
    # Slicing so we require only 1 nearest neighbors
    nearest_neighbors = nearest_neighbors[0]
                
    nearest_neighbor = nearest_neighbors[0]
    y_pred.append(nearest_neighbor)




classes = set(dataset[y_labels].values.ravel())
confusion_dataframe_1 = pd.DataFrame(np.zeros((len(classes),len(classes)),dtype=int),index= classes,columns=classes)

for (actual,pred) in zip(y_test.values.ravel(),y_pred):
    confusion_dataframe_1.at[actual,pred]+=1
    
y_test = pd.Series(y_test.values.ravel(), name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
confusion_dataframe = pd.crosstab(y_test, y_pred)