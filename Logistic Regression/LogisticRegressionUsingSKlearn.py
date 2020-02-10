import pandas as pd

dataset = pd.read_csv('H:/Software/Machine learning/1.1/Machine Learning A-Z™ Hands-On Python & R In Data Science/12 Logistic Regression/Code/Logistic_Regression/Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split  #### cross_validation==model_selection
X_train , X_test , y_train , y_test =  train_test_split(X , y , test_size = 0.3 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

from matplotlib.colors import ListedColormap
X_test , y_test = X_train , y_train