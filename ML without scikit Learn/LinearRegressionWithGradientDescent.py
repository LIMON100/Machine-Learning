import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Linear Regression/tvmarketing.csv')

x = dataset['TV']
y = dataset['Sales']


def gradient_descent(x , y):
    
    m_current = c_current = 0
    iterations = 60
    n = len(x)
    learning_rate = 0.001
    
    for i in range(iterations):
        
        y_predicted = m_current * x + c_current
        
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        
        #w1 = (1/n) * sum(x * (y_predicted - y))
        #w0 = (1/n) * sum(y - (y_predicted - y))
        
        w1 = -(2/n) * sum(x * (y - y_predicted))
        w0 = -(2/n) * sum(y - y_predicted)
        
        m_current = m_current - learning_rate * w1
        c_current = c_current - learning_rate * w0
        
        
        print("m {} , c {} , cost {} , iterations {}".format(m_current,c_current,cost,i))
    
    y_predicted = m_current * x + c_current
    
    
    plt.scatter(x , y)
    plt.scatter(x , y_predicted)
    plt.plot([min(x),max(x)] , [min(y_predicted),max(y_predicted)] , color='yellow')
    plt.show()
    
    return (m_current , c_current)




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




def plot_regression_line(x, y, b): 
    
    plt.scatter(x, y, color = "m", marker = "o", s = 30) 
    
    y_pred = b[0] + b[1]*x 
  
    plt.plot(x, y_pred, color = "g") 
   
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    plt.show() 




x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)


model = gradient_descent(x , y)

y_predict= find_prediction(x_test , model[1] , model[0])


accuracy = find_accuracy(y_test , y_predict , y)
print("ACCURACY is: ",accuracy*100)



plot_regression_line(y_test, y_predict, model)
