import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#df = pd.read_csv('H:/Software/Machine learning/Dataset/Logistice Regression/net.csv')
df = pd.read_csv('H:/Software/Machine learning/Dataset/Diabetis/diabetes.csv')

X = df.values[:,2:-1].astype('float64')
X = (X - np.mean(X, axis =0)) /  np.std(X, axis = 0)

## Add a bias column to the data
X = np.hstack([np.ones((X.shape[0], 1)),X])
X = MinMaxScaler().fit_transform(X)
Y = df["Outcome"]
Y = np.array(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)


def Sigmoid(z):
    return 1/(1 + np.exp(-z))


def Hypothesis(theta, x):   
    return Sigmoid(x @ theta) 


def Cost_Function(X,Y,theta,m):
    hi = Hypothesis(theta, X)
    _y = Y.reshape(-1, 1)
    J = 1/float(m) * np.sum(-_y * np.log(hi) - (1-_y) * np.log(1-hi))
    return J


def Cost_Function_Derivative(X,Y,theta,m,alpha):
    hi = Hypothesis(theta,X)
    _y = Y.reshape(-1, 1)
    J = alpha/float(m) * X.T @ (hi - _y)
    return J



def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = theta - Cost_Function_Derivative(X,Y,theta,m,alpha)
    return new_theta



def Accuracy(theta):
    correct = 0
    length = len(X_test)
    prediction = (Hypothesis(theta, X_test) > 0.5)
    _y = Y_test.reshape(-1, 1)
    correct = prediction == _y
    my_accuracy = (np.sum(correct) / length)*100
    print ('LR Accuracy %: ', my_accuracy)



def Logistic_Regression(X,Y,alpha,theta,num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        if x % 100 == 0:
            #print ('theta: ', theta)    
            print ('cost: ', Cost_Function(X,Y,theta,m))
    Accuracy(theta)



ep = .012

initial_theta = np.random.rand(X_train.shape[1],1) * 2 * ep - ep
alpha = 0.5
iterations = 2000
Logistic_Regression(X_train,Y_train,alpha,initial_theta,iterations)





"""x = dataset.drop(['User ID' , 'Purchased'] , axis = 1)
y = dataset['Purchased']

x = np.array(x)
y = np.array(y)


x = (x - x.mean()) / x.std()


def hypothesis(coef , x , n):
    
    h = np.ones((x.shape[0] , 1))
    
    coef = coef.reshape(1 , n+1) #Adding one more column for w0
    
    for i in range(0 , x.shape[0]):
        h[i] = 1 / (1+math.exp(-float(np.matmul(coef , x[i]))))
    
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
    
    print('Cost are: ',cost)
    
    return coef , history , cost


def logisticRegression(x , y , alpha , iteraiton):
    
    n = x.shape[1]
    
    one_column = np.ones((x.shape[0],1))
    x = np.concatenate((one_column, x), axis = 1) #Adding one more column
    
    coef = np.zeros(n+1)
    
    h = hypothesis(coef , x , n)
    
    coef , history , cost = gradientDescent(x , y , n , coef , h , alpha , iteraiton)
    
    return coef , cost
    
    
    

def main():
    
    #x0 = np.matrix(np.zeros(x.shape[1])) 

    coefficient , cost = logisticRegression(x , y , 0.01 , 20)


main()

"""