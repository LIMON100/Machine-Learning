import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Linear Regression/tvmarketing.csv')

x = dataset['TV']
y = dataset['Sales']


def estimateCof(x , y):
    n = np.size(x)
    
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    Slop_m1 = np.sum(y*x) - n*m_x*m_y
    Slop_m2 = np.sum(x*x) - n*m_x*m_x
    
    Slop = Slop_m1 / Slop_m2
    
    intercept = m_y - Slop*m_x
    
    return(intercept , Slop)


def plot_regression_line(x, y, b): 
    
    plt.scatter(x, y, color = "m", marker = "o", s = 30) 
    
    y_pred = b[0] + b[1]*x 
  
    plt.plot(x, y_pred, color = "g") 
   
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    plt.show() 
    
  


def find_prediction(x , m , c):
    
    y_predict = m*x + c
     
    return y_predict
        


def find_accuracy(real_value , predicted_value , y):
    
    ac1 = 0
    ac2 = 0
    
    mean_value_of_Y = np.mean(y)
        
    ac1 = sum((np.asarray(predicted_value) - mean_value_of_Y) * (np.asarray(predicted_value) - mean_value_of_Y))
    ac2 = sum((real_value - mean_value_of_Y) * (real_value - mean_value_of_Y))
        
    ac = ac1 / ac2
    return ac



x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)

b = estimateCof(x , y)
print("Slop and Intercept are: ",b[0] , b[1])

y_predict= find_prediction(x_test , b[1] , b[0])


accuracy = find_accuracy(y_test , y_predict , y)
print("ACCURACY is: ",accuracy*100)


plot_regression_line(y_test, y_predict, b)