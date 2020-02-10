from collections import Counter 

a = Counter("geeksforgeeks") 
  
for i in a.elements(): 
    print ( i, end = " ") 
print() 
      
b = Counter({'geeks' : 4, 'for' : 1,  
            'gfg' : 2, 'python' : 3}) 
  
for i in b.elements(): 
    print ( i, end = " ") 
print() 
  
c = Counter([1, 2, 21, 12, 2, 44, 5, 
              13, 15, 5, 19, 21, 5]) 
  
for i in c.elements(): 
    print ( i, end = " ") 
print()               
                
d = Counter( a = 2, b = 3, c = 6, d = 1, e = 5) 
  
for i in d.elements(): 
    print ( i, end = " ")
    
    
    
import numpy as np
x = np.array((3,5,7))
y = np.array((5,7,9))
p = np.hstack((x,y))


x = np.array([[3], [5], [7]])
y = np.array([[5], [7], [9]])
np.hstack((x,y))