import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Iris.csv')

x = dataset.iloc[:,:-1]
y = dataset['Species']


plt.hist([x['SepalLengthCm'] , x['SepalWidthCm']] , color = ['black' , 'red'] , label = ['SepalLengthCm' , 'SepalWidthCm'])
plt.legend()

a = [34,34,52,23,55,45,32,23,12,14,101,90,76,86,44]
#ct = [x for x in range (len(a))]
#plt.bar(a , ct)

bins = [0 , 10 , 20 , 30 , 40 ,50 ,60 , 70 , 80 , 90 , 100 , 120]

plt.hist(a , bins , histtype = 'bar' , rwidth = 0.8)


x1 = [2,4,4,12,44,4,414,1]
y1 = [5,6,35,234,54,23,4,2]
plt.scatter(x1 , y1  , color = 'k')


plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()