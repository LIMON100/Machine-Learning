from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn import metrics

dataset = pd.read_csv('H:/Software/Machine learning/1.1/22 Hierarchical Clustering/Hierarchical_Clustering/Mall_Customers.csv')

x = dataset.iloc[:,[3 , 4]]


dbs = DBSCAN(eps = 3 , min_samples = 4)
model = dbs.fit(x)

label = model.labels_

sample_cores = np.zeros_like(label , dtype = bool)

sample_cores[dbs.core_sample_indices_] = True

n_cluster = len(set(label)) - (1 if -1 in label else 0)

print(metrics.silhouette_score(x , label))