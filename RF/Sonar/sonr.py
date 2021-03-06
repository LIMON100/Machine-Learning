# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:20:33 2020

@author: limon
"""


from random import seed , randrange
from math import sqrt
from csv import reader


def load_csv(filename):
    
    dataset = list()
    
    with open(filename , 'r') as file:
        csv_reader = reader(file)
        
        for row in csv_reader:
            if not row:
                continue
            
            dataset.append(row)
            
            
    return dataset



def str_to_float(dataset , column):
    
    for row in dataset:
        row[column] = float(row[column].strip())
        


def str_to_int(dataset , column):
    
    class_value = [row[column] for row in dataset]
    
    unique = set(class_value)
    lookup = dict()
    
    for i , value in enumerate(unique):
        lookup[value] = i
        
    for row in dataset:
        row[column] = lookup[row[column]]
        
    return lookup
        


def cross_validation_split(dataset , n_folds):
    
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset_copy))
    
    for i in range(n_folds):
        fold = list()
        
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            
        
        dataset_split.append(fold)
        
    return dataset_split
        



def accuracy_matrics(actual , predicted):
    
    correct = 0
    
    for i in range(len(actual)):
        
        if(actual[i] == predicted[i]):
            correct+=1
        
    return correct / float(len(actual)) * 100.0





def evaluate_algorithm(dataset , algorithm , n_folds , *args):
    
    folds = cross_validation_split(dataset , n_folds)
    scores = list()
    
    for fold in folds:
        
        train_set = list(folds)
        train.set.remove(fold)
        train_set = list()
        
        for row in fold:
            
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
            
        predicted = algorithm(train_set , test_set , *args)
        actual = [row[-1] for row in fold]
        
        accuracy = accuracy_matrics(actual , predicted)
        
        scores.append(accuracy)

    return scores




def test_split(index , value , dataset):
    
    left , right = list() , list()
    
    for row in dataset:
        
        if row in dataset:
            
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
                
    return left , right




def gini_index(groups , classes):
    
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    
    for group in groups:
        size = float(len(group))
        
        if size == 0.0:
            continue
        
        score = 0.0
        
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            
            score += p*p
            gini += (1.0 - score) * (size / n_instances)
            
            
    return gini




def get_split(dataset , n_features):
    
    class_values = list(set(row[-1] for row in dataset))
    
    b_index , b_values , b_score , b_groups = 999 , 999 , 999 , None
            
    features = list()
    
    while(len(features) < n_features):
        
        index = randrange(len(dataset[0]) - 1)
        
        if index not in features:
            features.append(index)
            
        
        for index in features:
            features.append(index)
        
        return{"Index":b_index , "values":b_values , "Groups":b_groups}
    
        


def to_terminal(groups):
    
    outcomes = [row[-1] for row in groups]
    return max(set(outcomes) , key = outcomes.count)




def split(node , max_depth , min_size , n_features , depth):
    
    left , right = node['groups']
    del(node['groups'])
    
    if not left or not right:
        node['left'] = node['right']
        return
    
    if depth >= max_depth:
        node['left'] = to_terminal(left)
    
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
        



def build_tree(train , max_depth , min_size , n_features):
    
    root = get_split(train , n_features)
    split(root , max_depth , min_size , n_features , 1)
    
    return root



def predict(node , row):
    
    if row[node[index]] < node['value']:
        if isinstance(node['left'] , dict):
            return predict(node['left'] , row)
        
        else:
            return node['left']
        
        
    else:
        if isinstance(node['right'] , dict):
            return predict(node[right] , row)
        
        

def subsample(dataset , ratio):
    
    sample = list()
    n_samples = round(len(dataset) * ratio)
    
    while len(sample) < n_samples:
        index = randrange(len(dataset))
        sample.append(dataset[index])
        
    return sample



def bagging_predict(trees , row):
    
    predictions = [predict(trees , row) for tree in trees]
    return max(set(predictions) , key = predictions.count)




def random_forest_algorithm(train , test , max_depth , min_size , sample_size , n_trees , n_features):
    
    
    trees = list()
    
    for i in range(n_trees):
        
        sample = subsample(train , sample_size)
        tree = build_tree(sample , max_depth , min_size)
        trees.append(tree)
        predictions = [bagging_predict(trees , row) for row in test]
        
    return(predictions)
    


seed(2)



filename = 'sonar.all-data.csv'
dataset = load_csv(filename)


for i in range(0 , len(dataset[0] - 1)):
    str_to_float(dataset , i)
    
    
    
str_to_int(datset , len(dataset[0] -1))

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0] - 1)))


for n_trees in [1 , 5 , 10]:
    
    scores = evaluate_algorithm(dataset , random_forest_algorithm , n_folds , max_depth , min_size , n_features)















