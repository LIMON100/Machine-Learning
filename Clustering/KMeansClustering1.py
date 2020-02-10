import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt

dataset = pd.read_csv('H:/Software/Machine learning/1/22. Unsupervised Learning/Unsupervised/Online+Retail.csv' , sep = ',' , encoding = "ISO-8859-1" , header = 0)

dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'] , infer_datetime_format=True)


new_dataset = dataset.dropna()



amount = pd.DataFrame(new_dataset.Quantity * new_dataset.UnitPrice , columns = ['Amount'])
amount.head()


new_dataset = pd.concat(objs = [new_dataset , amount] , axis = 1 , ignore_index = False)

monetary = new_dataset.groupby('CustomerID').Amount.sum()
monetary = monetary.reset_index()



frequency = new_dataset[['CustomerID' , 'InvoiceNo']]

k = frequency.groupby('CustomerID').InvoiceNo.count()
k = pd.DataFrame(k)
k = k.reset_index()
k.columns = ["CustomerID", "Frequency"]

master = monetary.merge(k , on = 'CustomerID' , how = 'inner' )



recency = new_dataset[['CustomerID' , 'InvoiceDate']]
maximum = max(recency.InvoiceDate)

maximum = maximum + pd.DateOffset(days = 1)
recency['diff'] = maximum - recency.InvoiceDate 

a = recency.groupby('CustomerID')

df = pd.DataFrame(recency.groupby('CustomerID').diff.min())
df = df.reset_index()
df.columns = ["CustomerID", "Recency"]




RFM = k.merge(monetary , on = 'CustomerID')
RFM = RFM.merge(df , on = 'CustomerID')



plt.boxplot(RFM.Amount)
Q1 = RFM.Amount.quantile(0.25)
Q3 = RFM.Amount.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Amount >= Q1 - 1.5*IQR) & (RFM.Amount <= Q3 + 1.5*IQR)]




plt.boxplot(RFM.Frequency)
Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (RFM.Frequency <= Q3 + 1.5*IQR)]






RFM_norm1 = RFM.drop("CustomerID", axis=1)
RFM_norm1.Recency = RFM_norm1.Recency.dt.days

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)


RFM_norm1 = pd.DataFrame(RFM_norm1)
RFM_norm1.columns = ['Frequency','Amount','Recency']
RFM_norm1.head()




from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H




model_clus5 = KMeans(n_clusters = 5, max_iter=50)
model_clus5.fit(RFM_norm1)



from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(RFM_norm1)
    sse_.append([k, silhouette_score(RFM_norm1, kmeans.labels_)])
    

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])


ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(RFM_norm1)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)