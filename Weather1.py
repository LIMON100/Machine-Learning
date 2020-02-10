import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  
from sklearn.metrics import accuracy_score
from sklearn import metrics

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/daily_rainfall_data(work).csv')


status = pd.get_dummies(dataset['Station'] , drop_first = True)

dataset = pd.concat([dataset , status] , axis = 1)

dataset.drop(['Station'] , axis = 1 , inplace = True)


x = dataset.drop(['Rainfall'] , axis = 1)
y = dataset['Rainfall']


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , random_state = 0)

def vif_cal(input_data, dependent_col):
    
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    
    for i in range(0,xvar_names.shape[0]):
        y = x_vars[xvar_names[i]] 
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = sm.OLS(y,x).fit().rsquared  
        vif = round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)


lm = LinearRegression()
rfe = RFE(lm, 9)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_) 

col = x_train.columns[rfe.support_]

X_train_rfe = x_train[col]
X_train_rfe = sm.add_constant(X_train_rfe)



lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())

X_test_rfe = x_test[col]

X_test_rfe = sm.add_constant(X_test_rfe)

y_pred = lm.predict(X_test_rfe)
ac = metrics.r2_score(y_test , y_pred)
print(ac*100)