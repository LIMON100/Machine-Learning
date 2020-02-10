import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sc


dataset = pd.read_csv('G:/Software/Machine learning/1.1/21 K-Means Clustering/K_Means/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values


'''Finding cluster using DENDROGRAM'''
dendrogram = sc.dendrogram(sc.linkage(x , method = 'complete'))
plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()

dendrogram = sc.dendrogram(sc.linkage(x , method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()



'''Apply Hierarchical Clustering'''
from sklearn.cluster import AgglomerativeClustering
HC = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean' , linkage = 'ward')
y_hc = HC.fit_predict(x)



'''Visualization'''
plt.scatter(x[y_hc == 0 , 0] , x[y_hc == 0 , 1] , s = 100 , c = 'magenta' , label = 'Careful')
plt.scatter(x[y_hc == 1 , 0] , x[y_hc == 1 , 1] , s = 100 , c = 'blue' , label = 'Standard')
plt.scatter(x[y_hc == 2 , 0] , x[y_hc == 2 , 1] , s = 100 , c = 'green' , label = 'Target')
plt.scatter(x[y_hc == 3 , 0] , x[y_hc == 3 , 1] , s = 100 , c = 'cyan' , label = 'Careless')
plt.scatter(x[y_hc == 4 , 0] , x[y_hc == 4 , 1] , s = 100 , c = 'black' , label = 'Sensible')
plt.title('Clusters difference')
plt.xlabel('Annual income of client')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()