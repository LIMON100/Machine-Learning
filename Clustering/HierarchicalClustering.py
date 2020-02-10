import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


dataset = pd.read_csv('H:/Software/Machine learning/1/22. Unsupervised Learning/Unsupervised/Online+Retail.csv' , sep = ',' , encoding = 'ISO-8859-1' , header = 0)
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'] , infer_datetime_format=True)


print(dataset.isnull().sum())
print(dataset.isnull().sum()*100/dataset.shape[0])


without_missing = dataset.dropna()
print(without_missing.isnull().sum())



amount = pd.DataFrame(without_missing.Quantity * without_missing.UnitPrice , columns = ['Amount'])
amount.head()


without_missing = pd.concat(objs = [without_missing , amount] , axis =1 , ignore_index = False)


monetor = without_missing.groupby('CustomerID').Amount.sum()
monetor = monetor.reset_index()


frequency = without_missing[['CustomerID' , 'InvoiceNo']]
k = pd.DataFrame(frequency.groupby('CustomerID').InvoiceNo.count())
k = k.reset_index()
k.columns = ["CustomerID", "Frequency"]


new_dataset = monetor.merge(k , on = 'CustomerID' , how = 'inner')


recency = without_missing[['CustomerID' , 'InvoiceDate']]
maximum = max(recency.InvoiceDate)
maximum = maximum+pd.DateOffset(days = 1)
recency['diff'] = maximum - recency.InvoiceDate

a = recency.groupby('CustomerID')


df = pd.DataFrame(recency.groupby('CustomerID').diff.min())
df = df.reset_index()
df.columns = ['CustomerID' , 'Recency']


RFM = k.merge(monetor , on = 'CustomerID')
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



plt.boxplot(RFM.Recency)
Q1 = RFM.Recency.quantile(0.25)
Q3 = RFM.Recency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (RFM.Recency <= Q3 + 1.5*IQR)]


RFM_normal = RFM.drop('CustomerID' , axis = 1)
RFM_normal.Recency = RFM_normal.Recency.dt.days



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
RFM_normal = sc.fit_transform(RFM_normal)

RFM_normal = pd.DataFrame(RFM_normal)
RFM_normal.columns = ['Frequency' , 'Amount' , 'Recency']



##Hopkins Test##
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

def hopkinS(x):
    d = x.shape[1]
    
    n = len(x)
    m = int(0.1  * n)
    nbr = NearestNeighbors(n_neighbors = 1).fit(x.values)
    
    rand_x = sample(range(0 , n , 1) , m)
    
    ujd = []
    wjd = []
    
    for j in range(0 , m):
        u_dist, _ = nbr.kneighbors(uniform(np.amin(x , axis = 0) , np.amax(x , axis = 0) , d).reshape(1 , -1) , 2 , return_distance = True)
        ujd.append(u_dist[0][1])
        
        w_dist, _ = nbr.kneighbors(x.iloc[rand_x[j]].values.reshape(1 , -1) , 2 , return_distance = True)
        wjd.append(w_dist[0][1])
        
    
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    
    if isnan (H):
        print(ujd , wjd)
        H = 0
        
    return H

print(hopkinS(RFM_normal))



model = KMeans(n_clusters = 5 , max_iter = 50)
model.fit(RFM_normal)


from sklearn.metrics import silhouette_score
sse = []
for k in range(2 , 15):
    kmeans = KMeans(n_clusters = k).fit(RFM_normal)
    sse.append([k , silhouette_score(RFM_normal , kmeans.labels_)])
    
    
plt.plot(pd.DataFrame(sse)[0] , pd.DataFrame(sse)[1])




ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(RFM_normal)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)


link = linkage(RFM_normal , method = 'single' , metric = 'euclidean')
dendrogram(link)
plt.show()


link = linkage(RFM_normal , method = 'complete' , metric = 'euclidean')
dendrogram(link)
plt.show()


clusterCut = pd.Series(cut_tree(link, n_clusters = 5).reshape(-1,))
RFM_hc = pd.concat([RFM, clusterCut], axis=1)
RFM_hc.columns = ['CustomerID', 'Frequency', 'Amount', 'Recency', 'ClusterID']

RFM_hc.Recency = RFM_hc.Recency.dt.days
km_clusters_amount = pd.DataFrame(RFM_hc.groupby(["ClusterID"]).Amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_hc.groupby(["ClusterID"]).Frequency.mean())
km_clusters_recency = pd.DataFrame(RFM_hc.groupby(["ClusterID"]).Recency.mean())


df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]
df.head()


sns.barplot(x=df.ClusterID, y=df.Amount_mean)
sns.barplot(x=df.ClusterID, y=df.Frequency_mean)
sns.barplot(x=df.ClusterID, y=df.Recency_mean)