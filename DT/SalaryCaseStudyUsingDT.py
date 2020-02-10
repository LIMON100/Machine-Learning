import pandas as pd
from sklearn.model_selection import train_test_split , KFold , GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/DT/adult_dataset.csv')

#dataset = dataset[dataset['workclass'] != '?']

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


#dataset['income'] = dataset['income'].astype('category')

x = dataset.drop(['income'] , axis = 1)
y = dataset['income']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 90)


dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(x_train , y_train)


y_pred = dt.predict(x_test)

print(classification_report(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(confusion_matrix(y_test , y_pred))


n_folds = 5
parameter = {'max_depth':range (1,40)}

dtree = DecisionTreeClassifier(criterion = "gini" , random_state = 100)

tree = GridSearchCV(dtree , parameter , scoring = 'accuracy')
tree.fit(x_train , y_train)




n_folds = 5
parameters = {'min_samples_leaf':range (5 , 200 , 20)}

dtree = DecisionTreeClassifier(criterion = "gini" , random_state = 100)

tree = GridSearchCV(dtree , parameters , cv = n_folds , scoring = 'accuracy')
tree.fit(x_train , y_train)

score = tree.cv_results_
print(pd.DataFrame(score).head())