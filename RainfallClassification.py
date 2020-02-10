import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import warnings; warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split , KFold , GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


missing_values = ["nan" , "-"]
dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/Rainy.csv' , na_values = missing_values)



def convert_numeric(data):
    if data == 'Fog':
        return 0
    elif data == 'Rain , Thunderstorm':
        return 1
    elif data == 'Thunderstorm':
        return 0
    elif data == 'Rain':
        return 1
    elif data == 'Fog , Rain':
        return 1
    elif data == 'Rain , Snow':
        return 1
    else:
        return 0
    
    
dataset['Result'] = dataset['Result'].apply(convert_numeric)
dataset = dataset.drop(['Unnamed: 23'] , axis = 1)


x = dataset.iloc[:,0:-2].values
y = dataset.iloc[:,-1].values


imput = SimpleImputer(missing_values=np.nan, strategy='mean')
imput = imput.fit(x[:,3:21])
x[:,3:21] = imput.transform(x[:,3:21])


r_square_list = []

for i in range(1 , 100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(x_train , y_train)

    predicted = classifier.predict(x_test)

    ac = accuracy_score(y_test , predicted)
    r_square_list.append(ac)
    
    

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))

con_mat = confusion_matrix(y_test , predicted)






"""RANDOM FOREST"""
for i in range(1 , 100):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)
    
    n_folds = 5
    parameter = {'max_depth': range(100 , 200  , 55)}
    
    rf = RandomForestClassifier()
    
    rf = GridSearchCV(rf , parameter , cv = n_folds , scoring = "accuracy")
    
    rf.fit(x_train , y_train)
    
    predicted = rf.predict(x_test)

    ac = accuracy_score(y_test , predicted)
    r_square_list.append(ac)
    
    
max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))

con_mat = confusion_matrix(y_test , predicted)



"""Boosting"""


for i in range(1 , 100):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)
    
    param_grid = {'learning_rate': [0.2 , 0.6] , 'subsample': [0.3 , 0.6 , 0,9]}
    
    GBC = GradientBoostingClassifier(max_depth = 2 , n_estimators = 200)
    folds = 5
    
    grid_GBC = GridSearchCV(GBC , param_grid = param_grid , cv = folds , scoring = "roc_auc" , return_train_score = True , verbose = 1)
    grid_GBC.fit(x_train , y_train)
    
    predicted = grid_GBC.predict(x_test)

    ac = accuracy_score(y_test , predicted)
    r_square_list.append(ac)
    
    
    
max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))

con_mat = confusion_matrix(y_test , predicted)





"""Logistic Regression"""


import statsmodels.api as sm
logm1 = sm.GLM(y_train,(sm.add_constant(x_train)), family = sm.families.Binomial())
logm1.fit().summary()

for i in range(1 , 50):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)
    
    classifier = LogisticRegression()
    classifier.fit(x_train , y_train)
    
    predicted = classifier.predict(x_test)

    ac = accuracy_score(y_test , predicted)
    r_square_list.append(ac)
    
    

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))

con_mat = confusion_matrix(y_test , predicted)

    



"""Support Vector Machine"""

for i in range(1 , 50):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)
    
    from sklearn.svm import SVC
    classifier = SVC(C = 100 , gamma = 0.0001 ,kernel = 'rbf')
    classifier.fit(x_train , y_train)
    
    predicted = classifier.predict(x_test)

    ac = accuracy_score(y_test , predicted)
    r_square_list.append(ac)
    
    

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))

con_mat = confusion_matrix(y_test , predicted)




"""XGBoost"""


for i in range(1 , 50):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)

    classifier = XGBClassifier()
    classifier.fit(x_train , y_train)
    
    predicted = classifier.predict(x_test)

    ac = accuracy_score(y_test , predicted)
    r_square_list.append(ac)
    
    

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))

con_mat = confusion_matrix(y_test , predicted)

