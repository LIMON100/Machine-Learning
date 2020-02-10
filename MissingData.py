import pandas as pd


dt = pd.read_csv('C:/Users/Mahmudur Limon/Downloads/MOSTOFA\Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')

X = dt.iloc[:,:-1].values
y = dt.iloc[:,3].values

#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values ='NAN', strategy='mean', axis=0)
 
#imputer = imputer.fit(X[:, 1:3])

#X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)



#feature_Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
 
