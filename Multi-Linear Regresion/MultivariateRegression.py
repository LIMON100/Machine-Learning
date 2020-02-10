import pandas as pd
import seaborn as sns
import numpy as np

#df = pd.read_csv('C:/Users/Mahmudur Limon/Downloads/Fentech Driver/taxi.csv')
df = pd.read_csv('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/Book2.csv')


#sns.pairplot(df)

#sns.heatmap(df.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='red',annot=True)


#features =  df.iloc[:,0:-1].values
#labels =  df.iloc[:,-1].values

features =  df.iloc[:,:-1]
labels =  df['price']



from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test =  train_test_split(features , labels , test_size = 0.2 , random_state = 0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features , labels)

y_pred = regressor.predict(x_test)


print(regressor.coef_)
print(regressor.intercept_)


print('Train Data:',regressor.score(x_train , y_train))
print('Test Data:',regressor.score(x_test , y_test))



#y_output0 = regressor.intercept_ + regressor.coef_[0]*x_test[0][0] + regressor.coef_[1]*x_test[0][1] + regressor.coef_[2]*x_test[0][2] + regressor.coef_[3]*x_test[0][3] 

#y_output1 = regressor.intercept_ + regressor.coef_[0]*x_test[1][0] + regressor.coef_[1]*x_test[1][1] + regressor.coef_[2]*x_test[1][2] + regressor.coef_[3]*x_test[1][3] 




from sklearn import metrics

print('MSE :', metrics.mean_squared_error(y_test,y_pred))

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test,y_pred))) 
