import numpy as np  
import pandas as pd 

def hypothesis(theta, X, n):
    
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    
    return h


def BGD(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    
    #gdm_df = pd.DataFrame( columns = ['Cost'])
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)   ###'''X.shape[0]==size of X'''
        
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
            
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
        
        #gdm_df.loc[i] = [cost]
        
        #print("Theta {} , cost {} , iterations {}".format(theta,cost,i))
        
    theta = theta.reshape(1,n+1)
    
    #print(gdm_df)
    return theta, cost


def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1) ## This is mainly for Constant w0=c=b
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,alpha,num_iters,h,X,y,n)
    
    return theta, cost


def main():
    
    dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/Book1.csv')
    
    #dataset = np.loadtxt('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/start.txt')

    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    x = np.array(x)
    y = np.array(y)

    
    mean = np.ones(x.shape[1])
    std = np.ones(x.shape[1])
    
    for i in range(0, x.shape[1]):
        mean[i] = np.mean(x.transpose()[i]) ##Find the mean of first col
        std[i] = np.std(x.transpose()[i])
        for j in range(0, x.shape[0]):
            x[j][i] = (x[j][i] - mean[i])/std[i]
            
    
    theta, cost = linear_regression(x, y,0.01, 100)
    
    #print(cost)


if __name__ == "__main__": 
    main() 



