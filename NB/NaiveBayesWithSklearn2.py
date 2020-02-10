import pandas as pd

dataset = pd.read_csv('G:/Software\Machine learning/1/16. Naive Bayes/NaiveBayes/example_train1.csv')

dataset['Class'] = dataset.Class.map({'cinema':0 , 'education':1})

'''Make the dataset as matrix format.'''
np_array = dataset.as_matrix()
x = np_array[:, 0]
y = np_array[:, 1]
y = y.astype('int')


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()

'''How many time each word is for.'''
vec.fit(x)
vec.vocabulary_

vec = CountVectorizer(stop_words = 'english')
vec.fit(x)
vec.vocabulary_

x_transformed = vec.transform(x)
x_transformed

x = x_transformed.toarray()

pd.DataFrame(x , columns = vec.get_feature_names())


dataset1 = pd.read_csv('G:/Software/Machine learning/1/16. Naive Bayes/NaiveBayes/example_test.csv')
dataset1['Class'] = dataset1.Class.map({'cinema':0 , 'education':1})
 
test_np_array = dataset1.as_matrix()
x_test = np_array[:, 0]
y_test = np_array[:, 1]
y_test = y_test.astype('int')


x_test_transformed = vec.transform(x_test)
x_test_transformed


x_test = x_test_transformed.toarray()



'''Multinomial NB'''
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(x,y)

proba = mnb.predict_proba(x_test)

print("Probability for Movie" , proba[:,0])
print("Probability for Education" , proba[:,1])
pd.DataFrame(proba , columns = ['Movie' , 'Education'])



'''Bernouli'''
from sklearn.naive_bayes import BernoulliNB

mnb = BernoulliNB()

mnb.fit(x,y)

proba = mnb.predict_proba(x_test)

print("Probability for Movie" , proba[:,0])
print("Probability for Education" , proba[:,1])
pd.DataFrame(proba , columns = ['Movie' , 'Education'])