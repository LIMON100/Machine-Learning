import numpy as np
import pandas as pd

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Breast_Cancer/breast-cancer.csv')

np.where(dataset.isnull())

convert = {'M' : 0 , 'B' : 1}

dataset['diagnosis'] =  dataset['diagnosis'].map(convert)

X = dataset.drop(['id' , 'diagnosis'] , axis = 1)
Y = dataset['diagnosis']

from sklearn.preprocessing import StandardScaler
X = StandardScaler.fit_transform(X.values)

from sklearn.model_selection import train_test_split
df1 = pd.DataFrame(X , columns = X.columns)


X_train , X_test , Y_train , Y_test = train_test_split(df1 , Y , train_size = 0.8 , random_state = 0)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7 , p = 2 , metric = 'minkowski')
knn.fit(X_train , Y_train)


from sklearn.model_selection import cross_val_predict , cross_val_score
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

ac = accuracy_score(Y_train , knn.predict(X_train))
con = confusion_matrix(Y_train , knn.predict(X_train))
print(ac)
print(con)

ac1 = accuracy_score(Y_test , knn.predict(X_test))
con1 = confusion_matrix(Y_test , knn.predict(X_test))
print(ac1)
print(con1)