import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def knn(dataset , new_features , k = 3):
    if len(dataset) >= k:
        warnings.warn('K has not enough value for voting')
        
    
    distance = []
    for i in dataset:
        for features in dataset[i]: 
            ed = np.linalg.norm(np.array(features) - np.array(new_features))
            distance.append([ed , i])
            
    
    vote = [i[1] for i in sorted(distance) [:k]]
    vote_result = Counter(vote).most_common(1)[0][0]
    
    return vote_result
    


#dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Breast_Cancer/breast-cancer.csv')
dataset = dataset.drop(['id'] , axis = 1)
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1 , 'B': 0})

dataset = np.asarray(dataset)

random.shuffle(dataset)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = dataset[:-int(test_size*len(dataset))]
test_data = dataset[-int(test_size*len(dataset)):]



"""for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])"""
    
    

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = knn(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)