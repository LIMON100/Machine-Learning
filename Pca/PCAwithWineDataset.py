import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/wine.csv')


x = dataset.iloc[:,1:]
y = dataset['1']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)


sc = StandardScaler()
sc.fit(x_train)
x_train_sc = sc.transform(x_train)
x_test_sc = sc.transform(x_test)
#x_train_sc = sc.fit_transform(x_train) 
#x_test_sc = sc.fit_transform(x_test) 



pca = PCA(n_components = None)
pca.fit(x_train_sc)
pca.transform(x_train_sc)

print(np.round(pca.explained_variance_ratio_ , 3))

pd.DataFrame(np.round(pca.components_ , 3) , columns = x.columns).T


plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel('number of components')
plt.ylabel('cummulative explain')


res = pca.transform(x_train_sc)

index_name = ['PCA_'+str(k) for k in range(0 , len(res))]

d = pd.DataFrame(res , columns = dataset.columns[1:] , index = index_name)[0:4]
d.T.sort_values(by = 'PCA_0')
print(d)