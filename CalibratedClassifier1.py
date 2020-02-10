'''
    To obtain meaningful predicted probabilities we need conduct what is called calibration. 
    In scikit-learn we can use the CalibratedClassifierCV class to create well calibrated predicted probabilities using k-fold cross-validation.
    In CalibratedClassifierCV the training sets are used to train the model and the test sets is used to calibrate the predicted probabilities. 
    The returned predicted probabilities are the average of the k-folds.
'''

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV


iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = GaussianNB()

clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
clf_sigmoid.fit(X, y)


new_observation = [[ 2.6,  2.6,  2.6,  0.4]]

p = clf_sigmoid.predict_proba(new_observation)
print(p)