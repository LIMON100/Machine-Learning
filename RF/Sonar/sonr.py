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
        

        
        