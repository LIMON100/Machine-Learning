import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('G:/Software/Machine learning/1/23. Dimension Reduction/PCA Dataset/train.csv')

x = df.drop('label' , axis = 1)
y = df['label']


plt.figure(figsize=(7,7))
idx = 10

grid_data = x.iloc[idx].as_matrix().reshape(28,28)
plt.imshow(grid_data , interpolation = 'none' , cmap = 'gray')
plt.show()

print(y[idx])


labels = y.head(15000)
data = x.head(15000)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit_transform(data)
print(sc.shape)


sample_data = sc
covar_matrix = np.matmul(sample_data.T , sample_data)


'''Finding Eigen-value and vector'''
from scipy.linalg import eigh

values , vectors = eigh(covar_matrix , eigvals=(782 , 783))
print('Shape of the eigen vectors' , vectors.shape)

tranpse = vectors.T
print('Shape of the eigen vectors' , tranpse.shape)


new_cordinates = np.matmul(tranpse , sample_data.T)

new_cordinates = np.vstack((new_cordinates , labels)).T
dataframe = pd.DataFrame(data = new_cordinates , columns = ('1st principal' , '2nd Principal' , 'Label'))


import seaborn as sns
sns.FacetGrid(dataframe , hue = "Label" , size = 6).map(plt.scatter , '1st principal' , '2nd Principal' , 'Label')
plt.show()




'''Using SKLEARN'''
from sklearn import decomposition
pca = decomposition.PCA()

pca.n_components = 2
pca_data = pca.fit_transform(sample_data)


pca_data = np.vstack((pca_data.T , labels)).T

pca_df = pd.DataFrame(data = pca_data , columns = ('1st principal' , '2nd Principal' , 'Label'))
sns.FacetGrid(pca_df , hue = "Label" , size = 6).map(plt.scatter , '1st principal' , '2nd Principal' , 'Label')
plt.show()


pca.n_components = 784
pca_data = pca.fit_transform(sample_data)

percent_var_expalined = pca.explained_variance_/np.sum(pca.explained_variance_)

cum_var_explained = np.cumsum(percent_var_expalined)

plt.figure(1 , figsize = (6,4))
plt.clf()
plt.plot(cum_var_explained , linewidth = 2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_variance')
plt.show()