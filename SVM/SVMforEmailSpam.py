import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

email_data = pd.read_csv('G:/Software/Machine learning/1/18. Support Vector Machine (SVM)/SVM/Spam.csv')

X = email_data.drop("spam" , axis = 1)
y = email_data.spam.values.astype(int)

sns.countplot(x = 'spam' , data = email_data)
print(email_data['spam'].describe())

from sklearn.preprocessing import scale
X = scale(X)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.27 , random_state = 0)

model = SVC(C = 1) #Here c does the control tradeoff between smooth decision boundary and classify trainning point correctly

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

from sklearn import metrics
metrics.confusion_matrix(y_true = y_test , y_pred = y_pred)

print("accuracy" , metrics.accuracy_score(y_pred , y_test))
print("precesion" , metrics.precision_score(y_pred , y_test))
print("reacall" , metrics.recall_score(y_pred , y_test))

folds = KFold(n_splits = 5 , shuffle = True , random_state = 4)
model = SVC(C=1)

cv_results = cross_val_score(model , X_train , y_train , cv = folds , scoring = None)
print(cv_results)


param = {"C": [0.1 , 1 , 10 , 100 , 1000]}
model = SVC()

grid_cv = GridSearchCV(estimator = model , param_grid = param , scoring = 'accuracy' , cv = folds , verbose = 1 , return_train_score = True)
grid_cv.fit(X_train , y_train)

cv_results = pd.DataFrame(grid_cv.cv_results_)

accuracy = grid_cv.best_score_

best = grid_cv.best_params_['C']

model = SVC(C=best)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

