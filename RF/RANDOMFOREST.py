import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns


dataset = pd.read_csv('H:/Software/Machine learning/Dataset/credit-card-default.csv')
#dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Iris.csv')
dataset = dataset.drop("ID" , axis = 1)
dataset = dataset.rename(columns={"defaulted":  "label"})


def train_test_split(dataset , test_size):
    
    if isinstance(test_size , float):
        test_size = round(test_size * len(dataset) )
        
    indices = dataset.index.tolist()
    test_indices = random.sample(population = indices , k = test_size)
    
    y_train = dataset.loc[test_indices]
    x_train = dataset.drop(test_indices)
    
    return x_train , y_train


x_train , y_train = train_test_split(dataset , test_size = 5000)
#x_train , y_train = train_test_split(dataset , test_size = 30)


data = x_train.values
data[:5]

def check_purity(data):
    
    label_col = data[:,-1]
    unique_class = np.unique(label_col)
    
    if len(unique_class) == 1:
        return True
    else:
        return False




def classification(data):
    label_col = data[:, -1]
    unique_class , unique_value_count = np.unique(label_col , return_counts = True)
    
    index = unique_value_count.argmax()
    classyfy_data = unique_class[index]
    
    return classyfy_data


#print(classification(x_train[x_train.AGE == 30].values))




def get_potential_split(data):
    potential_splits = {}
    _, n_columns = data.shape

    for columns_index in range(n_columns):
        potential_splits[columns_index] = {}
        values = data[:, columns_index]
        unique_values = np.unique(values)
        
        for index in range(len(unique_values)):
            if index != 0:
                current_values = unique_values[index]
                previous_values = unique_values[index - 1]
                
                potential_split = (current_values + previous_values) / 2
                
                potential_splits[columns_index] = potential_split
        
        
    return potential_splits
        
        
#potential_split = get_potential_split(x_train.values)





def split_data(data , split_col , split_value):
    split_col_value = data[:, split_col]
    
    data_below = data[split_col_value <= split_value]
    data_above = data[split_col_value > split_value]
    
    return data_below , data_above


split_col = 3
split_value = data[:, split_col]
data_below , data_above = split_data(data , split_col , split_value)





def calculate_entropy(data):
    label_col = data[:, -1]
    _,counts = np.unique(label_col , return_counts = True)
    
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    
    return entropy
    
    
    

def calculate_overall_entropy(data_below , data_above):
    
    n_data_point = len(data_below) + len(data_above)
    
    p_data_below = len(data_below) / n_data_point
    p_data_above = len(data_above) / n_data_point
    
    overall_entropy = p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above)
    
    return overall_entropy
    
    
    
    
    
    
def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value
    
    

potential_splits = get_potential_split(data)
l = determine_best_split(data , potential_splits)    
    
    
    
    
    


def decision_tree_algorithm(dataset , counter = 0):
    
    if counter == 0:
        data = dataset.values
    else:
        data = dataset
        
    
    
    
    
    
    














