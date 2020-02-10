import pandas as pd
from sklearn.metrics import accuracy_score , confusion_matrix

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Diabetis/diabetes.csv')

x = dataset.iloc[:,:-1]
y = dataset['Outcome']

from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test=train_test_split(x, y,test_size=0.2 , random_state=0)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train , y_train)

y_pred = model.predict(x_test)


ac = accuracy_score(y_test , y_pred)

cn = confusion_matrix(y_test , y_pred)