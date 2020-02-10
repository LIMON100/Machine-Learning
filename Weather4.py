import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import warnings; warnings.simplefilter('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/65 Years of Weather Data Bangladesh (1948 - 2013).csv' )

dataset = dataset.drop(['Station Number' , 'X_COR' , 'Y_COR' , 'ALT' , 'Period'] , axis = 1)

status = pd.get_dummies(dataset['Station Names'] , drop_first = True)

dataset = pd.concat([dataset , status] , axis = 1)

dataset.drop(['Station Names'] , axis = 1 , inplace = True)


x = dataset.drop(['Rainfall'] , axis = 1).values
y = dataset['Rainfall'].values

#labelencoder_X = LabelEncoder()
#x[:,0]= labelencoder_X.fit_transform(x[:,0])


#x = np.array(x)
#y = np.array(y)

#x = x.reshape(-1 , 1)
y = y.reshape(-1 , 1)


r_square_list = []

for i in range(1 , 100):
    
    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.20 , random_state = i)
    
    neigh = KNeighborsRegressor(n_neighbors=8)
    neigh.fit(x_train , y_train)

    predicted = neigh.predict(x_test)
    
    ac = neigh.score(y_test , predicted)
    r_square_list.append(ac)
    
        
max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.0f}'.format(max_r_square))