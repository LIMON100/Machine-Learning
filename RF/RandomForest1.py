import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('H:/Software/Machine learning/1/20. Ensembling/RF_forudemy/credit-card-default.csv')

from sklearn.model_selection import train_test_split

X = df.drop('defaulted',axis=1)

y = df['defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


predictions = rfc.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))





from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5

parameters = {'max_depth': range(2, 20, 5)}

rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="accuracy")
rf.fit(X_train, y_train)

scores = rf.cv_results_


plt.figure()
plt.plot(scores["param_max_depth"], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_max_depth"], scores["mean_test_score"], label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5

parameters = {'n_estimators': range(100, 1500, 400)}

rf = RandomForestClassifier(max_depth=4)

rf = GridSearchCV(rf, parameters,  cv=n_folds,  scoring="accuracy")
rf.fit(X_train, y_train)


scores = rf.cv_results_



plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5

parameters = {'max_features': [4, 8, 14, 20, 24]}

rf = RandomForestClassifier(max_depth=4)

rf = GridSearchCV(rf, parameters,  cv=n_folds,  scoring="accuracy")
rf.fit(X_train, y_train)


scores = rf.cv_results_


plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


n_folds = 5

parameters = {'min_samples_leaf': range(100, 400, 50)}

rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="accuracy")
rf.fit(X_train, y_train)

scores = rf.cv_results_



plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()





from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5

parameters = {'min_samples_split': range(200, 500, 50)}

rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters,  cv=n_folds, scoring="accuracy")
rf.fit(X_train, y_train)


scores = rf.cv_results_



plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()





param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,  cv = 3, n_jobs = -1,verbose = 1)




grid_search.fit(X_train, y_train)

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,max_depth=10,min_samples_leaf=100, min_samples_split=200,max_features=10,n_estimators=100)



rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
