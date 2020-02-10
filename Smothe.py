import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


dataset = pd.read_csv('G:/Software/Machine learning/1/19. Decision Tree/DT_forudemy/adult_dataset.csv')

dataset.apply(lambda x: x=='?' , axis = 0).sum()


dataset = dataset[dataset['workclass'] != '?']
dataset = dataset[dataset['occupation'] != '?']
dataset = dataset[dataset['native.country'] != '?']


from sklearn import preprocessing

dataset_cate = dataset.select_dtypes(include=['object'])

l = preprocessing.LabelEncoder()
dataset_cate = dataset_cate.apply(l.fit_transform)


dataset = dataset.drop(dataset_cate.columns , axis = 1)
dataset = pd.concat([dataset , dataset_cate] , axis = 1)


x = dataset.drop(['income'] , axis = 1)
y = dataset['income']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 90)


unique, count = np.unique(y_train, return_counts=True)
y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
print(y_train_dict_value_count)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 12)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


unique, count = np.unique(y_train_res, return_counts=True)
y_train_smote_value_count = { k:v for (k,v) in zip(unique, count)}
print(y_train_smote_value_count)

target = 'income'
clf = LogisticRegression().fit(x_train_res, y_train_res)
Y_Test_Pred = clf.predict(x_test)
pd.crosstab(pd.Series(Y_Test_Pred, name = 'Predicted') , pd.Series(y, name = 'Actual'))




def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass



def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass


generate_model_report(y_test, Y_Test_Pred)
generate_auc_roc_curve(clf, x_test)




pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

weights = np.linspace(0.005, 0.25, 10)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'smote__ratio': weights
    },
    scoring='f1',
    cv=3
)

grid_result = gsc.fit(x_train, y_train)

print("Best parameters : %s" % grid_result.best_params_)
weight_f1_score_df = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                                   'weight': weights })
weight_f1_score_df.plot(x='weight')