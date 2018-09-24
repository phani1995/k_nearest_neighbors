# -*- coding: utf-8 -*-

import pandas as pd
import math

#Iris Dataset
#dataset = pd.read_csv('iris.csv',header= None)

x_cordinates  = pd.Series(data = [11,1,2,1,8,9,10])
y_cordinates  = pd.Series(data = [11,1,1,3,8,10,9])
group = pd.Series(data = ['b','a','a','a','b','b','b'])

test_point =(0,1)
k=3

dataset = pd.concat([x_cordinates,y_cordinates,group],axis=1)

def euclidian_distance(point_1,point_2):
    sum_of_squares = 0
    for i in range(0,len(point_1)):
        sum_of_squares = sum_of_squares + math.pow(point_2[i]-point_1[i],2)
    distance = math.sqrt(sum_of_squares)
    return distance

distances_list = []
for index, row in dataset.iterrows():
    dist = euclidian_distance(test_point,(row[0],row[1]))
    distances_list.append([dist,row[2]])

#print(distances_list)
distances_list.sort(key = lambda ele : ele[0])
#print(distances_list)   

distances_list = distances_list[:k]
print(distances_list)
voting_dict = {}
for ele in distances_list:
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