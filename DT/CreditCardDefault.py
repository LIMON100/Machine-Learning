import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv('H:/Software/Machine learning/Dataset/RF/credit-card-default.csv')

x = dataset.drop(['defaulted'] , axis = 1)
y = dataset['defaulted']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 29)



lg = LogisticRegression()
lg.fit(x_train , y_train)
y_pred_lg = lg.predict(x_test)
ac_lg = accuracy_score(y_test , y_pred_lg)
print(ac_lg*100)


rfc = RandomForestClassifier()
rfc.fit(x_train ,y_train)

predict = rfc.predict(x_test)

print(classification_report(y_test , predict))
print(confusion_matrix(y_test , predict))
print(accuracy_score(y_test , predict))




n_folds = 5
parameters = {
        'n_neighbors': range (2 , 100 , 1)
        }

knn = KNeighborsClassifier()

tree = GridSearchCV(estimator = knn , param_grid = parameters , cv = n_folds , n_jobs = 1)
tree.fit(x_train , y_train)

score1 = tree.cv_results_

print(pd.DataFrame(score1).head())
print(tree.best_params_)




knn = KNeighborsClassifier(n_neighbors =  45)
knn.fit(x_train , y_train)
y_pred = knn.predict(x_test)
ac = accuracy_score(y_test , y_pred)
print(ac*100)
