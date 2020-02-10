import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Logistice Regression/net.csv')


x = dataset.drop(['User ID' , 'Purchased'] , axis = 1)
y = dataset['Purchased']

x = np.array(x)
y = np.array(y)


x = (x - x.mean()) / x.std()

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 0)


one_column = np.ones((x_test.shape[0],1))
x_test = np.concatenate((one_column, x_test), axis = 1)



def hypothesis(x0 , x , n):
    
    h = np.ones((x.shape[0] , 1))
    
    x0 = x0.reshape(1 , n+1) #Adding one more column for w0
    
    for i in range(0 , x.shape[0]):
        h[i] = 1 / (1 + math.exp(-float(np.matmul(x0 , x[i]))))
    
    h = h.reshape(x.shape[0])
    
    return h



def gradientDescent(x , y , n , coef , predicted , alpha , iteration):
    
    history = np.ones((iteration,n+1)) #create a list for history
    cost = np.ones(iteration)          # create a vector-list for cost

    for i in range(0 , iteration):
        
        coef[0] = coef[0] - (alpha/x.shape[0]) * sum(predicted - y)
        
        for j in range(1 , n+1):
            
            coef[j] = coef[j] - (alpha/x.shape[0]) * sum((predicted - y) * x.transpose()[j])
        
        
        history[i] = coef
        h = hypothesis(coef , x , n)
        
        cost[i] = (-1/x.shape[0]) * sum(y * np.log(h) + (1 - y) * np.log(1 - h))
            
    coef = coef.reshape(1 , n+1)
    
    #print('Coefficient are: ',history)
    
    return coef , history , cost



def find_prediction(x_test , coef , n):
    
     #predict_value = np.ones((x_test.shape[0] , 1))
     
     #l = x_test.shape[0]
     
     predict_value = hypothesis(coef , x_test , n)
     
     #predict_value.reshape(x.shape[0])
     
     return predict_value
    



def find_accuracy(real_value , predicted_value , y):
    
    pass



def logisticRegression(x , y , alpha , iteraiton):
    
    n = x.shape[1]
    
    one_column = np.ones((x.shape[0],1))
    x = np.concatenate((one_column, x), axis = 1) #Adding one more column
    
    coef = np.zeros(n+1)
    
    h = hypothesis(coef , x , n)
    
    coef , history , cost = gradientDescent(x , y , n , coef , h , alpha , iteraiton)
    
    y_predict = find_prediction(x_test , coef , n)
    
    ac= find_accuracy(y_test , y_predict , y)
    print(ac*100)
    
    
    return coef , cost
    
    

def main(): 

    coefficient = logisticRegression(x , y , 0.01 , 20)
    print(coefficient[0])
    
    
    
main()




"""mean = np.ones(x.shape[1])
std = np.ones(x.shape[1])

for i in range(0, x.shape[1]):
    mean[i] = np.mean(x.transpose()[i]) 
    std[i] = np.std(x.transpose()[i])
    for j in range(0, x.shape[0]):
        x[j][i] = (x[j][i] - mean[i])/std[i]
"""
