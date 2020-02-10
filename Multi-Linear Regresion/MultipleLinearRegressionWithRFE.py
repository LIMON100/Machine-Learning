import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/Housing.csv')


dataset['mainroad'] = dataset['mainroad'].map({'yes': 1, 'no': 0})
dataset['guestroom'] = dataset['guestroom'].map({'yes': 1, 'no': 0})
dataset['basement'] = dataset['basement'].map({'yes': 1, 'no': 0})
dataset['hotwaterheating'] = dataset['hotwaterheating'].map({'yes': 1, 'no': 0})
dataset['airconditioning'] = dataset['airconditioning'].map({'yes': 1, 'no': 0})
dataset['prefarea'] = dataset['prefarea'].map({'yes': 1, 'no': 0})


status = pd.get_dummies(dataset['furnishingstatus'],drop_first=True)
dataset = pd.concat([dataset,status],axis=1)
dataset.drop(['furnishingstatus'] , axis=1 , inplace=True)


dataset = (dataset-np.min(dataset)) / (np.max(dataset) - np.min(dataset))


x = dataset.drop(['price'] , axis = 1)
y = dataset['price']



x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)


x_train = sm.add_constant(x_train) 
lm_1 = sm.OLS(y_train , x_train).fit()
lm_1.params
lm_1.summary()



from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
rfe = RFE(lm, 10)             
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)           #
print(rfe.ranking_) 

col = x_train.columns[rfe.support_]

X_train_rfe = x_train[col]

import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit() 


X_test_rfe = x_test[col]

X_test_rfe = sm.add_constant(X_test_rfe)

y_pred = lm.predict(X_test_rfe)



c = [i for i in range(1,165,1)] # generating index 
fig = plt.figure() 
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)


fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16) 



fig = plt.figure()
sns.distplot((y_test-y_pred),bins=50)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)  




print("The R squared Error: ",r2_score(y_test , y_pred))