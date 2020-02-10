import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
#import re

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline



dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/Housing.csv')

dataset1 = dataset

dataset = dataset.loc[:, ['area', 'price']]


df_columns = dataset.columns
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

dataset = pd.DataFrame(dataset)
dataset.columns = df_columns


sns.regplot(x="area", y="price", data=dataset, fit_reg=False)


df_train, df_test = train_test_split(dataset , train_size = 0.7, test_size = 0.3, random_state = 10)


X_train = df_train['area']
X_train = X_train.values.reshape(-1, 1)
y_train = df_train['price']

X_test = df_test['area']
X_test = X_test.values.reshape(-1, 1)
y_test = df_test['price']



degrees = [1, 2, 3, 6, 10, 20]

y_train_pred = np.zeros((len(X_train), len(degrees)))
y_test_pred = np.zeros((len(X_test), len(degrees)))

for i, degree in enumerate(degrees):
    
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    y_train_pred[:, i] = model.predict(X_train)
    y_test_pred[:, i] = model.predict(X_test)
    
    
    

plt.figure(figsize=(16, 8))


plt.subplot(121)
plt.scatter(X_train, y_train)
plt.yscale('log')
plt.title("Train data")
for i, degree in enumerate(degrees):    
    plt.scatter(X_train, y_train_pred[:, i], s=15, label=str(degree))
    plt.legend(loc='upper left')
    

plt.subplot(122)
plt.scatter(X_test, y_test)
plt.yscale('log')
plt.title("Test data")
for i, degree in enumerate(degrees):    
    plt.scatter(X_test, y_test_pred[:, i], label=str(degree))
    plt.legend(loc='upper left')
    
    
    
print("R-squared values: \n")

for i, degree in enumerate(degrees):
    train_r2 = round(sklearn.metrics.r2_score(y_train, y_train_pred[:, i]), 2)
    test_r2 = round(sklearn.metrics.r2_score(y_test, y_test_pred[:, i]), 2)
    print("Polynomial degree {0}: train score={1}, test score={2}".format(degree, train_r2, test_r2))
    



binary_vars_list =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

def binary_map(x):
    return x.map({'yes': 1, "no": 0})

dataset1[binary_vars_list] = dataset1[binary_vars_list].apply(binary_map)
dataset1.head()


status = pd.get_dummies(dataset1['furnishingstatus'], drop_first = True)

dataset1 = pd.concat([dataset1, status], axis = 1)

dataset1.drop(['furnishingstatus'], axis = 1, inplace = True)



df_train, df_test = train_test_split(dataset1, train_size = 0.7, test_size = 0.3, random_state = 100)


scaler = MinMaxScaler()

numeric_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_train[numeric_vars] = scaler.fit_transform(df_train[numeric_vars])
df_test[numeric_vars] = scaler.fit_transform(df_test[numeric_vars])


y_train = df_train.pop('price')
X_train = df_train

y_test = df_test.pop('price')
X_test = df_test



lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=10)             
rfe = rfe.fit(X_train, y_train)


y_pred = rfe.predict(X_test)

r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)



lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=8)             
rfe = rfe.fit(X_train, y_train)

y_pred = rfe.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)





lm = LinearRegression()
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=5)


folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=folds)


scores = cross_val_score(lm, X_train, y_train, scoring='mean_squared_error', cv=5)




folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
hyper_params = [{'n_features_to_select': list(range(1, 14))}]


lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)             

model_cv = GridSearchCV(estimator = rfe, param_grid = hyper_params, scoring= 'r2', cv = folds, verbose = 1,return_train_score=True)      

model_cv.fit(X_train, y_train)  


cv_results = pd.DataFrame(model_cv.cv_results_)



plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')





n_features_optimal = 10

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(X_train, y_train)

y_pred = lm.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)