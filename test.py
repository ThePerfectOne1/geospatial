import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('ggplot')
import pandas as pd
from pandas import *


def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #print(np.array(features),'  ', np.array(predict), ' ' , euclidean_distance,'\n')
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[89,70,77], 'r':[97,91,118], 'g':[99,81,238], 'c':[140,149,170]}
new_features = [89,95,115]
#result= k_nearest_neighbors(dataset, new_features)
#print(result)

# Reading and converting data into required format
df= pd.read_csv('C:/Users/Pooja/dataset.csv') 
print('Size : ', df.size,'\n')
#convert data into list of list where each inner list is made of row
df= df.astype(int).values.tolist()
print('Data Frame : ' ,df, '\n')

#convert this dataframe into list of columns
X = []
Y = []
Z = []
for i in df:
    X.append(i[0])
    Y.append(i[1])
    Z.append(i[2]) 
'''
# Calculating the group to which a point belongs 
for i in df:
    result = k_nearest_neighbors(dataset, i)
    print(result)
'''

from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

#[[ax.scatter(ii[0],ii[1],ii[2], color=i) for ii in dataset[i]] for i in dataset]
# same as:

# Calculating the group to which a point belongs 
j=0
for i in df:
    result = k_nearest_neighbors(dataset, i)
    print(result)
    #ax.scatter(df[i[0]], df[i[1]], df[i[2]], color = result, marker='o')
#j=0
#for j in range(81):
    
    ax.scatter(df[j][0], df[j][1], df[j][2],color= result ,marker = 'o')
    j = j + 1
    print(j)
'''
#access value of a point rather that individual element in list value
for i in dataset:
    #print(i[0],' ',i[1],' ',i[2],'\n')
        ax.scatter(dataset[i][0],dataset[i][1],dataset[i][2],color= i ,marker = 'o')
    
    
ax.scatter(new_features[0], new_features[1],new_features[2] , marker = 'o')

result = k_nearest_neighbors(dataset, new_features)
ax.scatter(new_features[0], new_features[1],new_features[2], c = result , marker = 'o')

'''
#[[ax.scatter(ii[0],ii[1],ii[2], color=i) for ii in dataset[i]] for i in dataset]
# same as:

#access value of a point rather that individual element in list value
for i in dataset:
    #print(i[0],' ',i[1],' ',i[2],'\n')
        ax.scatter(dataset[i][0],dataset[i][1],dataset[i][2],color= i ,marker = 'o')
    
'''
ax.scatter(new_features[0], new_features[1],new_features[2] , marker = 'o')

result = k_nearest_neighbors(dataset, new_features)
ax.scatter(new_features[0], new_features[1],new_features[2], c = result , marker = 'o')
'''
'''# Calculating the group to which a point belongs 
for i in df:
    result = k_nearest_neighbors(dataset, i)
    print(result)
    ax.scatter(df[i[0]], df[i[1]], df[i[2]], color=i, marker='o')
    #ax.scatter(df[i][0], df[i][1], df[i][2],color= i ,marker = 'o')
'''

#ax.scatter(x,y,z,c='r',marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# ,[2,3,4],[3,1,5]   ,[7,7,4],[8,6,5]
