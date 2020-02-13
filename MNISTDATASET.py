import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import eigh
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('G:/Software/Machine learning/1/23. Dimension Reduction/PCA Dataset/train.csv')

x = dataset.drop(['label'] , axis = 1)
y = dataset['label']


plt.figure(figsize=(6,6))
idx = 1

data = x.iloc[idx].as_matrix().reshape(28 , 28)
plt.imshow(data , interpolation = 'none' , cmap  = 'gray')
plt.show()


sc = StandardScaler().fit_transform(x)
print(sc.shape)


simple_data = sc
cov_matrix = np.matmul(simple_data.T , simple_data)
print(cov_matrix.shape)


values , vectors = eigh(cov_matrix , eigvals = (782 , 783))
print("Shape of eigen vectors = ",vectors.shape)

vectors = vectors.T

print(vectors.shape)



new_cordinates = np.matmul(vectors , simple_data.T)
print(new_cordinates.shape)



new_condinates = np.vstack((new_cordinates , y)).T
dataframe = pd.DataFrame(data=new_condinates, columns=("1st_principal", "2nd_principal", "label"))
print(dataframe.head())


sns.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


from sklearn import decomposition
pca = decomposition.PCA()


pca.n_components = 2
pca_data = pca.fit_transform(simple_data)
print("shape of pca_reduced.shape = ", pca_data.shape)






pca_data = np.vstack((pca_data.T, y)).T

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sns.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()




pca.n_components = 784
pca_data = pca.fit_transform(simple_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20)

lr = LogisticRegression()
lr.fit(x_train , y_train)
y = lr.predict(x_test)
ac = accuracy_score(y_test , y)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train , y_train)
y_rf = rf.predict(x_test)
ac = accuracy_score(y_test , y_rf)
print(ac*100)