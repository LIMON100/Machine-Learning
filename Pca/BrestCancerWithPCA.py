import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
df = load_breast_cancer()

type(df)
print(df['DESCR'])

dataset = pd.DataFrame(df['data'] , columns = df['feature_names'])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(dataset)
scled_data = sc.transform(dataset)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scled_data)
x_pca = pca.transform(scled_data)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0] , x_pca[:,1] , c = df['target'])
plt.xlabel('1st principal component.')
plt.ylabel('2nd principal component.')
plt.show()


df_comp = pd.DataFrame(pca.components_,columns = df['feature_names'])

plt.figure(figsize = (12 , 6))
sns.heatmap(df_comp , cmap = 'plasma')