import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('H:/Software/Machine learning/1/20. Ensembling/RF_forudemy/credit-card-default.csv')

x = df.drop('defaulted' , axis = 1)
y = df[['defaulted']]


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.30 , random_state = 0)


shallow_tree = DecisionTreeClassifier(max_depth = 2 , random_state = 100)
shallow_tree.fit(x_train , y_train)


y_pred = shallow_tree.predict(x_test)
matrix = accuracy_score(y_test , y_pred)
print(matrix)


estimator = list(range(1 , 50 , 3))


abc_score = []

for n_est in estimator:
    ABC = AdaBoostClassifier(base_estimator = shallow_tree , n_estimators = n_est)
    
    ABC.fit(x_train , y_train)
    y_pred = ABC.predict(x_test)
    score = accuracy_score(y_test , y_pred)
    abc_score.append(score)
    
    

plt.plot(estimator , abc_score)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.ylim([0.80 , 1])
plt.show()