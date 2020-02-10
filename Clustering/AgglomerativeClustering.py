import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs
x , y = make_blobs(n_samples = 600 , centers = 5 , cluster_std = 0.60 , random_state = 45)
plt.scatter(x[:,0] , x[:,1] , s = 10)


from scipy.cluster.hierarchy import ward,linkage,dendrogram
np.set_printoptions(precision = 4 , suppress = True)

distance = linkage(x , 'ward')

plt.figure(figsize = (25 , 10))
plt.title('Hierarchical clustering')
plt.xlabel('index')
plt.ylabel('ward distance')
dendrogram(distance , orientation = 'left' , leaf_rotation = 90., leaf_font_size = 4.,)




'''TRUNCATING Dendrogram'''
plt.figure(figsize = (25 , 10))
plt.title('Hierarchical clustering')
plt.xlabel('index')
plt.ylabel('ward distance')
dendrogram(distance , leaf_rotation = 90., leaf_font_size = 4.,)
plt.axhline(50 , c = 'k');



plt.title('Hierarchical Clustering with truncating')
plt.xlabel('index')
plt.ylabel('Wards Distance')
dendrogram(distance , truncate_mode = 'lastp' , p = 5 , leaf_rotation = 0. , leaf_font_size = 12., show_contracted = True)



'''By Distance'''
from scipy.cluster.hierarchy import fcluster
Max_distance = 25
clusters = fcluster(distance , Max_distance , criterion = 'distance')


from scipy.cluster.hierarchy import fcluster
Max_distance = 100
clusters = fcluster(distance , Max_distance , criterion = 'distance')


plt.figure(figsize = (10 , 8))
plt.scatter(x[:,0] , x[:,1] , c = clusters , cmap = 'prism')



'''By Cluster'''
k = 5
clusters = fcluster(distance , k , criterion = 'maxclust')
plt.figure(figsize = (10 , 8))
plt.scatter(x[:,0] , x[:,1] , c = clusters , cmap = 'prism')




'''K-Means'''
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 5)
model.fit(x)
y_kmeans = model.predict(x)

plt.scatter(x[:,0] , x[:,1] , c = y_kmeans , s=10 , cmap = 'inferno')
centers = model.cluster_centers_
plt.scatter(centers[:,0] , centers[:,1] , c = 'cyan' , s = 300)



from mlxtend.plotting import plot_decision_regions

print(model.inertia_)

elbow = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i).fit(x)
    elbow.append([i , kmeans.inertia_])
    
plt.plot(pd.DataFrame(elbow)[0] , pd.DataFrame(elbow)[1])



from sklearn.metrics import silhouette_score
silhoutte = []
for i in range(2,8):
    kmeans = KMeans(n_clusters = i).fit(x)
    silhoutte.append([i , silhouette_score(x , kmeans.labels_)])
    
plt.plot(pd.DataFrame(silhoutte)[0] , pd.DataFrame(silhoutte)[1])