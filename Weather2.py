import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import MultinomialNB
import statsmodels.api as sm  
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

"""dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/daily_rainfall_data.csv')


status = pd.get_dummies(dataset['Station'] , drop_first = True)

dataset = pd.concat([dataset , status] , axis = 1)

dataset.drop(['Station'] , axis = 1 , inplace = True)


x = dataset.drop(['Rainfall'] , axis = 1)
y = dataset['Rainfall']


x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 100)

vect = CountVectorizer(stop_words = 'english')
vect.fit(x_train)

x_train_transformed = vect.transform(x_train)
x_test_transformed = vect.transform(x_test)

mnb = MultinomialNB()

mnb.fit(x_train_transformed , y_train)


y_pred = mnb.predict(x_test_transformed)
ac = accuracy_score(y_test , y_pred)
mat = confusion_matrix(y_test , y_pred)

print('Accuracy of Naive Bayes: ',ac)"""

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/65 Years of Weather Data Bangladesh (1948 - 2013).csv')
dataset = dataset.drop(['Station Names'] , axis = 1)


x = dataset.drop(['Rainfall'] , axis = 1)
y = dataset['Rainfall']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , random_state = 100)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)
ac = accuracy_score(y_test , y_pred)
print(ac)