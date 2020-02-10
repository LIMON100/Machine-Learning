import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import warnings; warnings.simplefilter('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/Rainfall-2.csv')



#status = pd.get_dummies(dataset['Station'] , drop_first = True)

#dataset = pd.concat([dataset , status] , axis = 1)

#dataset.drop(['Station'] , axis = 1 , inplace = True)


x = dataset.iloc[:,0:5].values
y = dataset.iloc[:,-1].values


labelencoder_X = LabelEncoder()
x[:,0]= labelencoder_X.fit_transform(x[:,0])


x = np.array(x)
y = np.array(y)

#x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)
error = []
for i in range(1 , 100):
    
    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.20 , random_state = i)
    
    neigh = KNeighborsRegressor(n_neighbors=8)
    neigh.fit(x_train , y_train)

    predicted = neigh.predict(x_test)
    
    ac = neigh.score(y_test , y_test)
    error.append(ac)
    

max_r_square = max(error)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square))
    

for i in range(1,100):
    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.20,random_state = i)
    regressor = DecisionTreeRegressor(max_depth=5)
    regressor.fit(x_train,y_train)

    predicted = regressor.predict(x_test)
    print('i =',i)
    print('R-squared test score: {:.3f}'.format(regressor.score(x_test,y_test)))
    








for i in range(1 , 30):

    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.20,random_state = i)
    regressor = RandomForestRegressor(n_estimators=200, max_depth=None, max_features=1 , min_samples_leaf=2, min_samples_split=2, bootstrap=False)
    regressor.fit(x_train, y_train)

    predicted = regressor.predict(x_test)

    print("i = ",i)

    print('R-squared test score: {:.3f}'.format(regressor.score(x_test , y_test)))

sns.set(style="darkgrid")
sns.tsplot(y_test[0:50])
sns.tsplot(predicted[0:50],color="indianred")
plt.xlabel('Number of Observation')
plt.ylabel('Rainfall (mm)')
plt.title('Rainfall Prediction with Random Forest Regressor')
plt.show()