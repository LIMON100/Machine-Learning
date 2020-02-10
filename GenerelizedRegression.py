import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline


df = pd.read_csv('G:/Software/Machine learning/1/24. Advanced Machine Learning Algorithms/AdvanceReg/total-electricity-consumption-us.csv' , sep = ',' , header = 0)

print(df.isnull().sum())


size = len(df.index)
index = range(0 , size , 5)


train = df[~df.index.isin(index)]
test = df[df.index.isin(index)]


x_train = train.Year.values.reshape(-1,1)
y_train = train.Consumption

x_test = test.Year.values.reshape(-1,1)
y_test = test.Consumption


'''
x = df['Year']
y = df['Consumption']

x_train , x_test , y_train , y_test = train_test_split(x, y , test_size = 0.2 , random_state = 3)

x_train = x_train.Year.reshape(-1,1)
x_test = x_test.Year.reshape(-1,1)
'''


t1 = []
t2 = []
degrees = [1,2,3]

for d in degrees:
    pipeline = Pipeline([('poly_features' , PolynomialFeatures(degree = d)) , ('model' , LinearRegression())])
    pipeline.fit(x_train , y_train)
    y_pred = pipeline.predict(x_test)
    
    t2.append(metrics.r2_score(y_test , y_pred))
    
    y_pred_train = pipeline.predict(x_train)
    t1.append(metrics.r2_score(y_train , y_pred_train))
    
    
    fig , ax = plt.subplots()
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption')
    ax.set_title('Degree = ' + str(d))
    
    ax.scatter(x_train , y_train)
    ax.plot(x_train , y_pred_train)
    
    ax.scatter(x_train , y_train)
    ax.plot(x_test , y_pred)
    
    plt.show()
    
    

print(degrees)
print(t1)
print(t2)