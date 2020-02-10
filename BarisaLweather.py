import pandas as pd
import gc
import numpy as np
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score , KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression


train_dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/65 Years of Weather Data Bangladesh (1948 - 2013).csv')
test_dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/65 Years of Weather Data of Barisal.csv')



train_dataset = train_dataset.drop(['Max Temp' , 'Min Temp' , 'Bright Sunshine' , 'Period'  , 'X_COR' , 'ALT'] , axis = 1)
test_dataset = test_dataset.drop(['Max Temp' , 'Min Temp' , 'Bright Sunshine' , 'Period'  , 'X_COR' , 'ALT'] , axis = 1)


labelencoder_X = LabelEncoder()
train_dataset['Station Names']= labelencoder_X.fit_transform(train_dataset['Station Names'])
test_dataset['Station Names']= labelencoder_X.fit_transform(test_dataset['Station Names'])



df_train = train_dataset
df_test = test_dataset


mm_scaler = MinMaxScaler()
features_names_train = ['Station Names' , 'YEAR' , 'Month' , 'Rainfall' , 'Relative Humidity' , 'Wind Speed' , 'Cloud Coverage' , 'Station Number', 'Y_COR' , 'LATITUDE' , 'LONGITUDE']
features_names_test = ['Station Names' , 'YEAR' , 'Month' , 'Rainfall' , 'Relative Humidity' , 'Wind Speed' , 'Cloud Coverage' , 'Station Number', 'Y_COR' , 'LATITUDE' , 'LONGITUDE']


df_train[features_names_train] = mm_scaler.fit_transform(df_train[features_names_train])
df_test[features_names_test] = mm_scaler.fit_transform(df_test[features_names_test])


gc.collect()


x_train = df_train
y_train = df_train.pop('Rainfall')


x_test = df_test
y_test = df_test.pop('Rainfall')




n_folds = 5
parameters = {
        'n_neighbors': range (2 , 50 , 2)
        }

knn = KNeighborsRegressor()

tree = GridSearchCV(estimator = knn , param_grid = parameters , cv = n_folds , n_jobs = -1)
tree.fit(x_train , y_train)

score1 = tree.cv_results_

print(pd.DataFrame(score1).head())
print(tree.best_params_)

knn = KNeighborsRegressor(n_neighbors = 18)
knn.fit(x_train , y_train)

y_pred_knn = knn.predict(x_test)

ac_knn = r2_score(y_test , y_pred_knn)
print('After Cross-validation: ',ac_knn*100)








sv = SVR()
sv.fit(x_train , y_train)

y_pred_svm = sv.predict(x_test)

ac_svr = r2_score(y_test , y_pred_svm)
print(ac_svr*100)


folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
params = {"C": [0.01 , 0.1, 1, 10, 100, 1000]}

model = SVR()

model_cv_C = GridSearchCV(estimator = model, param_grid = params, cv = folds , verbose = 1 , return_train_score=True)
model_cv_C.fit(x_train, y_train) 


cv_results = pd.DataFrame(model_cv_C.cv_results_)
cv_results


plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')




folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
gamma = {'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

model = SVR()

model_cv_g = GridSearchCV(estimator = model, param_grid = gamma, cv = folds , verbose = 1 , return_train_score=True)
model_cv_g.fit(x_train, y_train) 



folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
kernels = {'kernel': ['rbf' , 'poly' , 'sigmoid']}  

model = SVR()

model_cv_k = GridSearchCV(estimator = model, param_grid = kernels, cv = folds , return_train_score=True)
model_cv_k.fit(x_train, y_train)



print(model_cv_C.best_params_)
print(model_cv_g.best_params_)
print(model_cv_k.best_params_)


sv = SVR(C = 1000 , gamma = 1 , kernel = 'poly')
sv.fit(x_train , y_train)

y_pred_svm = sv.predict(x_test)

ac_svr = r2_score(y_test , y_pred_svm)
print(ac_svr*100)



param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf' , 'poly' , 'sigmoid']}



folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

model = SVR()

model_cv_svr = GridSearchCV(estimator = model, param_grid = param_grid, cv = folds , return_train_score=True)
model_cv_svr.fit(x_train, y_train) 

cv_results = pd.DataFrame(model_cv_svr.cv_results_)

print(model_cv_svr.best_params_)



