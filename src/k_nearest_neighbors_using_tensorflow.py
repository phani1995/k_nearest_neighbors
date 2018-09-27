# -*- coding: utf-8 -*-

# Imports
import numpy as np
import pandas as pd
import tensorflow as tf

# Iris Dataset
#dataset = pd.read_csv('iris.csv',header= None)

x_cordinates  = pd.Series(data = [11,1,2,1,8,9,10])
y_cordinates  = pd.Series(data = [11,1,1,3,8,10,9])
group = pd.Series(data = ['b','a','a','a','b','b','b'])
dataset = pd.concat([x_cordinates,y_cordinates,group],axis=1)

test_point =(0,1)
# K value
k=3

test_features = tf.placeholder(tf.float32)
features = tf.placeholder(tf.float32)
euclidain_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(features,test_features))))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    nearest_neighbors = []
    for index, row in dataset.iterrows():
        feed_dict={features:np.array([row[0],row[1]]),test_features:np.array(test_point)}
        dist = sess.run(euclidain_distance,feed_dict = feed_dict)
        nearest_neighbors.append([dist,row[2]])
    
    #print(distances_list)
    nearest_neighbors.sort(key = lambda ele : ele[0])
    #print(distances_list)   
    
    nearest_neighbors = nearest_neighbors[:k]
    print(nearest_neighbors)
        
    voting_dict = {}
    for ele in nearest_neighbors:
        if ele[1] not in list(voting_dict.keys()):
            voting_dict[ele[1]] = 1
        else:
            voting_dict[ele[1]]+=1
    print(voting_dict)
    
    max_value = 0 
    max_key = None
    for key,value in voting_dict.items():
        if value >max_value :
            max_value = value
            max_key = key
    
    print(max_key)    
    
    
    
