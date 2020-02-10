import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/advertising.csv')

sns.pairplot(dataset)

x = dataset.iloc[:,:-1]
y = dataset['Sales']


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)

model = LinearRegression()
model.fit(x_train , y_train)

print(model.intercept_)
cof = pd.DataFrame(model.coef_ , x_test.columns , columns = ['Coeficient'])
print(cof)


y_pred = model.predict(x_test)


mse = mean_squared_error(y_test , y_pred)
rsquared = r2_score(y_test , y_pred)

import statsmodels.api as sm

x_train_sm = x_train
x_train_sm = sm.add_constant(x_train_sm) # Add an extra columns for constant W0

lm_1 = sm.OLS(y_train , x_train_sm).fit()
lm_1.params

"""
coef = coefficient values of trainning features

std_error = Measure of the Variability in the estimate for the Co-efficient

t = Ratio of the Estimated Co-efficint to the standard-Deviation (Co-efficient/std_error)

p = if it is greater than 0.05 than reject the feature beacuse it is not significant.

"""

x_train_new = x_train[['TV' , 'Radio']]
x_test_new = x_test[['TV' , 'Radio']]

model.fit(x_train_new , y_train)


y_pred = model.predict(x_test_new)
rsquared = r2_score(y_test , y_pred)


x_train_sm = x_train_new
x_train_sm = sm.add_constant(x_train_sm) 

lm_1 = sm.OLS(y_train , x_train_sm).fit()
lm_1.params
lm_1.summary()
