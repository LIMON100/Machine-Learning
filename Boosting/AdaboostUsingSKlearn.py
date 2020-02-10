import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Breast_Cancer/breast-cancer.csv')

x = dataset.drop(['diagnosis'] , axis = 1)
y = dataset['diagnosis']


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20)


shallow_tree = DecisionTreeClassifier(max_depth = 2 , random_state = 10)

shallow_tree.fit(x_train , y_train)

pred = shallow_tree.predict(x_test)

score = accuracy_score(y_test , pred)
print(score)
print(confusion_matrix(y_test , pred))


estimators = list(range(1 , 50 , 3))

abc_score = []
con = []

for i in estimators:
    ABC = AdaBoostClassifier(base_estimator = shallow_tree , n_estimators = i)
    
    ABC.fit(x_train , y_train)
    y_pred = ABC.predict(x_test)
    score = accuracy_score(y_test , y_pred)
    abc_score.append(score)
    
print(abc_score)

plt.plot(estimators , abc_score)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.ylim([0.85 , 1])
plt.show()
