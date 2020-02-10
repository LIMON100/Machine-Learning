import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve , auc
import matplotlib.pyplot as plt

#data = pd.read_csv('H:/Software/Machine learning/1/16. Naive Bayes/NaiveBayes/SmallSpamData.csv')
data = pd.read_csv('H:/Software/Machine learning/Dataset/Diabetis/diabetes.csv')

#ham_spam = data.Class.value_counts()

#data['label'] = data.Class.map({'ham':0 , 'spam':1})

#x = data.sms
#y = data.label

x = data.iloc[:, :-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 1)


"""from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()


vec.fit(x_train)

vec.vocabulary_


x_train_transformed = vec.transform(x_train)
x_test_transformed = vec.transform(x_test)"""


from sklearn.naive_bayes import BernoulliNB

mnb = BernoulliNB()

mnb.fit(x_train , y_train)

y_proba_class = mnb.predict(x_test)
y_proba_pred = mnb.predict_proba(x_test)


from sklearn import metrics
accu = metrics.accuracy_score(y_test , y_proba_class)
print('Accuracy:',accu)

confusion = metrics.confusion_matrix(y_test , y_proba_class)



TN = confusion[0 , 0]
FP = confusion[0 , 1]
FN = confusion[1 , 0]
TP = confusion[1 , 1]



sensitivity = TP / float(FN + TP)
print(sensitivity)


specificity = TN / float(TN + FP)
print(specificity)


precision = TP / float(FP + TP)
print(precision)



false_positive_rate , true_positive_rate , threshold = roc_curve(y_test , y_proba_pred[:,1])
roc_auc = auc(false_positive_rate , true_positive_rate)



