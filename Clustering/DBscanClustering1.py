import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np

x , y = datasets.make_moons(n_samples = 1500 , noise = 0.05)

x1 = x[:,0]
x2 = x[:,1]

print('This is the dataset we want to Cluster.')
plt.scatter(x1 , x2 , s = 30)
plt.show()


db = DBSCAN()
db.fit(x)
y_db = db.labels_.astype(np.int)

colors = np.array(['#ff0000' , '#00ff00'])
                   

print('This is the dataset we want to DBSCAN-Cluster.')
plt.scatter(x1 , x2 , s = 20 , color = colors[y_db])
plt.show()




k = KMeans(n_clusters = 2)
k.fit(x)
y_k = k.labels_.astype(np.int)

colors = np.array(['#ff0000' , '#00ff00'])
                   

print('This is the dataset we want to DBSCAN-Cluster.')
plt.scatter(x1 , x2 , s = 20 , color = colors[y_k])
plt.show()