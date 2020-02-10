import numpy as np
import matplotlib.pyplot as plt

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
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
   
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    plt.show() 

def main():
    
    x = np.array([6 , 8 , 12 , 14 , 18])
    y = np.array([350 , 775 , 1150 , 1395 , 1675])
    
    b = estimateCof(x , y)
    print("Estimated coefficients:\nb_0 = {}  \ \nb_1 = {}".format(b[0], b[1])) 
    
    plot_regression_line(x, y, b)
    
if __name__ == "__main__": 
    main()