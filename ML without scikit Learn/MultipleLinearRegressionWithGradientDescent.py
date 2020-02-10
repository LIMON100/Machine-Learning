import pandas as pd
import numpy as np

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



def hypothesis(x , n , Coef_W): #Initial Hypothesis for 0 or PREDICTED -> y = mx+c
    
    h = np.ones((x.shape[0] , 1)) #Create a new ROW
    
    Coef_W = Coef_W.reshape(1 , n+1) 
    
    for i in range(0 , x.shape[0]):
        h[i] = float(np.matmul(Coef_W , x[i]))
    
    h = h.reshape(x.shape[0])

    return h
    
    
    
def GradientDescent(x , y , n , alpha , iteration , h , Coef_W):
    
    cost = np.ones(iteration) #Create an Vector with no of iteration
    
    for i in range(0 , iteration):
        
        Coef_W[0] = Coef_W[0] - (alpha/x.shape[0]) * sum(h - y)
        
        for j in range(1 , n+1):
            
            Coef_W[j] = Coef_W[j] - (alpha/x.shape[0]) * sum((h - y) * x.transpose()[j])
            
        
        h = hypothesis(x , n , Coef_W)
        cost[i] = (1/x.shape[0]) * 0.5 * sum(np.square(h - y))
        
    
    Coef_W = Coef_W.reshape(1,n+1)
    print(cost)
    
    return Coef_W , cost
        
        



def Multiplelinearegression(x , y , alpha , iteration):
    
    n = x.shape[1]
    
    col = np.ones((x.shape[0] , 1))   #Ntun akta col add kora holo WO calculate korar jnno"""
    
    x = np.concatenate((col , x) , axis = 1) 
    
    Coef_W = np.zeros(n+1)  #No of Co-efficient"""
    
    h = hypothesis(x , n , Coef_W)
    
    theta , cost = GradientDescent(x , y , n , alpha , iteration , h , Coef_W)
    
    return theta , cost
    
    
    
    
theta , cost = Multiplelinearegression(x , y , 0.01 , 100)







"""theta = np.matrix(np.array([0,0,0,0,0,0,0,0])) 

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


print(gradient_descent_multi(x, y, theta, 0.01, 100))"""

