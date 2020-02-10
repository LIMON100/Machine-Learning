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

email_data = pd.read_csv('H:/Software/Machine learning/1/18. Support Vector Machine (SVM)/SVM/Spam.csv')

X = email_data.drop("spam" , axis = 1)
y = email_data.spam.values.astype(int)

from sklearn.preprocessing import scale
X = scale(X)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.27 , random_state = 0)

model = SVC(C = 1 , kernel = 'rbf') #Here c does the control tradeoff between smooth decision boundary and classify trainning point correctly

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


param = [{ 'gamma': [1e-2, 1e-3, 1e-4] , 'C': [0.1 , 1 , 10 , 100 , 1000]}]
model = SVC(kernel = 'rbf')

grid_cv = GridSearchCV(estimator = model , param_grid = param , scoring = 'accuracy' , cv = folds , verbose = 1 , return_train_score = True)
grid_cv.fit(X_train , y_train)

cv_results = pd.DataFrame(grid_cv.cv_results_)


cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

accuracy = grid_cv.best_score_

best = grid_cv.best_params_['C']

model = SVC(C=best)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

