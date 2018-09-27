# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    print( len(data),'Length of data+')
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        print(group ,'Group being searched')
        for features in data[group]:
            print(features,'Features being compared')
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #print(euclidean_distance,' this distance for feature',features)
            distances.append([euclidean_distance,group])
    
    print(distances)
    print('---------------------------')
    for i in sorted(distances)[:k]:
        print(i[1])
    print('---------------------------')

    votes = [i[1] for i in sorted(distances)[:k]]
    print(votes)
    print(Counter(votes))
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
        
plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
print(result)
plt.scatter(new_features[0], new_features[1], s=100, color = result)  
plt.show()