import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(5 , 100 , 100)
y = np.linspace(10 , 1000 , 100)


plt.plot([1,4,6,8] , [3,8,3,5])

plt.plot(x , y)
plt.xlabel("current")
plt.ylabel("voltage")
plt.title("Ohn-Law")


#####DIfferent Shape of Point
x = np.linspace(0 , 10 , 20)
y = x*2
plt.plot(x , y , 'r^')


#### SUBPLOT
x = np.linspace(1 , 10 , 100)
y = np.log(x)

plt.figure(1)

plt.subplot(121)
plt.title("y = log(x)")
plt.plot(x , y)

plt.subplot(122)
plt.title("y = log(x)**2")
plt.plot(x , y**2)
plt.show()

