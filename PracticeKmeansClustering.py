import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('H:/Software/Machine learning/1.1/21 K-Means Clustering/K_Means/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values


'''Elbow Method'''
from sklearn.cluster import KMeans
wcss = []
for i in range(1 , 11):
    model = KMeans(n_clusters = i , init = 'k-means++' , max_iter = 300 , n_init = 10 , random_state = 0).fit(x)
    wcss.append(model.inertia_)
    
plt.plot(range(1 , 11) , wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


'''Apply k-means'''
kmeans = KMeans(n_clusters = 5 , init = 'k-means++' , max_iter = 300 , n_init = 10 , random_state = 0)
y_kmeans = kmeans.fit_predict(x)


'''Visualization'''
plt.scatter(x[y_kmeans == 0 , 0] , x[y_kmeans == 0 , 1] , s = 100 , c = 'magenta' , label = 'Careful')
plt.scatter(x[y_kmeans == 1 , 0] , x[y_kmeans == 1 , 1] , s = 100 , c = 'blue' , label = 'Standard')
plt.scatter(x[y_kmeans == 2 , 0] , x[y_kmeans == 2 , 1] , s = 100 , c = 'green' , label = 'Target')
plt.scatter(x[y_kmeans == 3 , 0] , x[y_kmeans == 3 , 1] , s = 100 , c = 'cyan' , label = 'Careless')
plt.scatter(x[y_kmeans == 4 , 0] , x[y_kmeans == 4 , 1] , s = 100 , c = 'black' , label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[:,1] , s = 300 , c = 'red' , label = 'Centroids')
plt.title('Clusters difference')
plt.xlabel('Annual income of client')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()