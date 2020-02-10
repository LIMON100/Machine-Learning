import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")

dataset = pd.read_csv('H:/Software/Machine learning/1/19. Decision Tree/DT_forudemy/adult_dataset.csv')


df_1 = dataset[dataset.workclass == '?']
print(df_1)


dataset = dataset[dataset['workclass'] != '?' ]


data_cata = dataset.select_dtypes(include=['object'])

data_cata.apply(lambda x: x=='?' , axis = 0).sum()


dataset = dataset[dataset['native.country'] != '?' ]
dataset = dataset[dataset['occupation'] != '?' ]

data_cata = dataset.select_dtypes(include=['object'])

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data_cata = data_cata.apply(le.fit_transform)
data_cata.head()


dataset = dataset.drop(data_cata.columns , axis = 1)
dataset = pd.concat([dataset , data_cata] , axis = 1)


dataset['income'] = dataset['income'].astype('category')


from sklearn.model_selection import train_test_split

x = dataset.drop('income' , axis = 1)
y = dataset['income']


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.27 , random_state = 99)


from sklearn.tree import DecisionTreeClassifier
data_default = DecisionTreeClassifier(max_depth = 5)
data_default.fit(x_train , y_train)


from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
y_pred = data_default.predict(x_test)

print(classification_report(y_test , y_pred))
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))



from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

features = list(dataset.columns[1:])

dot_data = StringIO()
export_graphviz(data_default , out_file = dot_data , feature_names = features , filled = True , rounded = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())