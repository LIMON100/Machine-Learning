import numpy as np
import matplotlib.pyplot as pt


def gradientDescent(x , y):
    
    m_current = c_current = 0
    iterations = 100
    n = len(x)
    learning_rate = 0.01
    
    for i in range(iterations):
        
        y_predicted = m_current * x + c_current
        
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        
        md = -(2/n) * sum(x * (y - y_predicted))
        cd = -(2/n) * sum(y - y_predicted)
        
        m_current = m_current - learning_rate * md
        c_current = c_current - learning_rate * cd
        
        
        print("m {} , c {} , cost {} , iterations {}".format(m_current,c_current,cost,i))
    
    y_predicted = m_current * x + c_current
    pt.scatter(x , y)
    pt.scatter(x , y_predicted)
    pt.plot([min(x),max(x)] , [min(y_predicted),max(y_predicted)] , color='yellow')
    pt.show()

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradientDescent(x , y)