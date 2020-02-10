import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('H:\Software\Machine learning\Dataset\Multiple Linear Rregression/Housing.csv')

dataset['mainroad'] = dataset['mainroad'].map({'yes': 1, 'no': 0})
dataset['guestroom'] = dataset['guestroom'].map({'yes': 1, 'no': 0})
dataset['basement'] = dataset['basement'].map({'yes': 1, 'no': 0})
dataset['hotwaterheating'] = dataset['hotwaterheating'].map({'yes': 1, 'no': 0})
dataset['airconditioning'] = dataset['airconditioning'].map({'yes': 1, 'no': 0})
dataset['prefarea'] = dataset['prefarea'].map({'yes': 1, 'no': 0})


status = pd.get_dummies(dataset['furnishingstatus'],drop_first=True)
dataset = pd.concat([dataset,status],axis=1)
dataset.drop(['furnishingstatus'],axis=1,inplace=True)


dataset = (dataset - dataset.mean())/dataset.std()


x = dataset.drop(['price'] , axis = 1)
y = dataset['price']

x['intercept'] = 1
x = x.reindex_axis(['intercept','area','bedrooms' , 'bathrooms' , 'stories' , 'mainroad' , 'basement' , 'parking'], axis=1)

x = np.array(x)
y = np.array(y)


theta = np.matrix(np.array([0,0,0,0,0,0,0,0])) 
alpha = 0.01
iterations = 1000


def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))





def gradient_descent_multi(X, y, theta, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)
    gdm_df = pd.DataFrame( columns = ['Bets','cost'])

    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        gdm_df.loc[i] = [theta,cost]

    return gdm_df




def find_accuracy(real_value , predicted_value , y):
    
    ac1 = 0
    ac2 = 0
    
    mean_value_of_Y = np.mean(y)
        
    ac1 = sum((np.asarray(predicted_value) - mean_value_of_Y) * (np.asarray(predicted_value) - mean_value_of_Y))
    ac2 = sum((real_value - mean_value_of_Y) * (real_value - mean_value_of_Y))
        
    ac = ac1 / ac2
    return ac



b = gradient_descent_multi(x, y, theta, alpha, iterations)
